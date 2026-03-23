## Component 5: Order Management System (OMS)

### Responsibility

- Place limit orders paired with mandatory server-side stop-loss orders on the Dhan broker API
- Fan out orders to N accounts (multi-account replication): one signal produces N independent order lifecycles
- Track order status via Dhan's live order update WebSocket (primary) with REST fallback
- Execute per-order fill management loops with per-order locking and sequence numbers
- Manage SL lifecycle as a first-class concern: pairing, qty sync on partial fills, overnight conversion, verification
- Rate-limit API calls via per-account priority token buckets (10 OPS per Dhan account)
- Flatten all intraday positions at EOD via batched parallel IOC-escalating exit sequence
- Operate in degraded mode when Redis is unavailable
- Enforce a global kill switch that cancels everything and flattens all positions across all accounts
- **Does NOT** generate signals, resolve instruments, allocate capital, or manage positions (those are upstream components)

---

### Dhan API Integration

All broker interaction goes through the Dhan HTTP REST API and WebSocket API. This section documents the exact endpoints, request/response schemas, and integration patterns.

**Base URL:** `https://api.dhan.co`
**Auth header:** `access-token: {dhan_access_token}` (per-account, obtained during daily login)
**Content-Type:** `application/json`
**Rate limit:** 10 orders per second per API key (hard limit, enforced server-side)

#### Order Placement: POST /v2/orders

Places a new order on the exchange.

```python
class DhanOrderRequest(pydantic.BaseModel):
    """Exact request body for POST /v2/orders."""
    dhanClientId: str                # Dhan client ID for this account
    transactionType: Literal["BUY", "SELL"]
    exchangeSegment: Literal["NSE_EQ", "NSE_FNO", "BSE_EQ", "BSE_FNO"]
    productType: Literal["INTRADAY", "CNC", "MARGIN", "MTF", "CO", "BO"]
    orderType: Literal["LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_MARKET"]
    validity: Literal["DAY", "IOC"]
    securityId: str                  # Dhan's internal security ID (from instrument master CSV)
    quantity: int                    # total order quantity in units (not lots)
    price: float                    # limit price (required for LIMIT and STOP_LOSS)
    triggerPrice: float | None      # required for STOP_LOSS and STOP_LOSS_MARKET
    disclosedQuantity: int | None   # portion visible to market (0 = full visibility)
    afterMarketOrder: bool          # True for AMO (placed after market hours, executed at open)
    amoTime: Literal["OPEN", "OPEN_30", "OPEN_60"] | None  # when AMO activates
    boProfitValue: float | None     # bracket order profit target (not used in v1)
    boStopLossValue: float | None   # bracket order SL (not used in v1)
    correlationId: str | None       # our internal reference (max 25 chars)
    # NOTE: correlationId is returned in order updates but NOT used by Dhan
    # for dedup. Two identical orders with the same correlationId = two orders.

class DhanOrderResponse(pydantic.BaseModel):
    """Response from POST /v2/orders on success (HTTP 200)."""
    orderId: str                    # Dhan's order ID (unique, used for all subsequent operations)
    orderStatus: Literal["TRANSIT", "PENDING", "REJECTED"]
    # TRANSIT = sent to exchange, awaiting acknowledgement
    # PENDING = accepted by exchange, waiting in order book
    # REJECTED = rejected by Dhan's risk checks or exchange

class DhanErrorResponse(pydantic.BaseModel):
    """Response on failure (HTTP 4xx/5xx)."""
    status: str                     # "failure"
    remarks: str                    # human-readable error
    errorCode: str | None           # e.g., "DH-906" (insufficient margin)
    errorType: str | None           # e.g., "Input_Exception", "Order_Exception"
```

**Example: Place a LIMIT BUY for 650 qty of NIFTY CE option:**

```python
request = DhanOrderRequest(
    dhanClientId="1000000001",
    transactionType="BUY",
    exchangeSegment="NSE_FNO",
    productType="INTRADAY",
    orderType="LIMIT",
    validity="DAY",
    securityId="43925",              # NIFTY 24500 CE weekly
    quantity=650,                     # 10 lots × 65 per lot
    price=245.50,
    triggerPrice=None,
    disclosedQuantity=0,
    afterMarketOrder=False,
    amoTime=None,
    boProfitValue=None,
    boStopLossValue=None,
    correlationId="S1_a3f8c1d2e4b6",
)

# POST https://api.dhan.co/v2/orders
# Headers: {"access-token": "...", "Content-Type": "application/json"}
# Body: request.model_dump_json()
```

#### Order Modification: PUT /v2/orders/{order-id}

Modifies price, quantity, order type, or validity of a pending order.

```python
class DhanModifyRequest(pydantic.BaseModel):
    """Request body for PUT /v2/orders/{orderId}."""
    dhanClientId: str
    orderId: str
    orderType: Literal["LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_MARKET"]
    legName: str | None              # for multi-leg orders (not used in v1)
    quantity: int                    # CRITICAL: this is the ORIGINAL TOTAL quantity,
                                     # NOT the remaining quantity. If you placed 650
                                     # and 325 filled, you still send quantity=650
                                     # to keep the remaining 325 alive.
    price: float
    triggerPrice: float | None
    disclosedQuantity: int | None
    validity: Literal["DAY", "IOC"]

class DhanModifyResponse(pydantic.BaseModel):
    """Response from PUT /v2/orders/{orderId}."""
    orderId: str
    orderStatus: str                 # typically "TRANSIT" or "PENDING"
```

**The quantity=original_total gotcha:**

Dhan's modify endpoint interprets `quantity` as the total order quantity, not the remaining unfilled quantity. This is a critical integration detail:

```
Scenario:
  1. Place order: qty=650
  2. Partial fill: 325 filled, 325 remaining
  3. Want to reprice the remaining 325

  CORRECT:   PUT /v2/orders/{id}  quantity=650  price=new_price
             → Dhan keeps remaining_qty=325, updates price

  WRONG:     PUT /v2/orders/{id}  quantity=325  price=new_price
             → Dhan interprets this as "reduce total qty to 325"
             → Since 325 already filled, remaining becomes 0
             → Order cancelled! 325 shares bought at old price, no more fills.

  CRITICAL:  Always pass the ORIGINAL quantity from the initial placement.
             Store original_qty in OrderState at creation time. Never change it.
```

**25-modification limit:**

Dhan allows a maximum of 25 modifications per order. After 25 mods, further modify requests are rejected. The OMS tracks `modification_count` per order and switches to cancel-replace at modification 24 (one before the limit, as a safety margin):

```python
MAX_MODIFICATIONS = 24  # switch to cancel-replace at 24, not 25

async def reprice_order(self, state: OrderState, order: ResolvedOrder,
                        params: FillParams, account: Account) -> None:
    new_price = self.compute_new_limit_price(order, params)
    rate_limiter = self._account_rate_limiters[account.account_id]

    if state.modification_count >= MAX_MODIFICATIONS:
        # Cancel-replace: cancel the old order, place a new one
        await rate_limiter.acquire("MODIFY")
        try:
            await self._dhan_cancel_order(account, state.order_id)
        except OrderAlreadyFilled:
            return  # race: filled between check and cancel — fine

        await rate_limiter.acquire("MODIFY")
        new_resp = await self._dhan_place_order(account, DhanOrderRequest(
            dhanClientId=account.dhan_client_id,
            transactionType=order.transaction_type,
            exchangeSegment=order.exchange_segment,
            productType=self._product_type(order),
            orderType="LIMIT",
            validity="DAY",
            securityId=order.instrument_id,
            quantity=state.remaining_qty,  # NEW order: use remaining, not original
            price=new_price,
            triggerPrice=None,
            disclosedQuantity=0,
            afterMarketOrder=False,
            amoTime=None,
            boProfitValue=None,
            boStopLossValue=None,
            correlationId=f"CR_{order.signal.strategy_id}_{order.signal.signal_id[:12]}",
        ))

        if new_resp.orderStatus == "REJECTED":
            logger.error("cancel_replace_rejected",
                        account_id=account.account_id,
                        old_order=state.order_id)
            return

        # Update state to track the new order
        old_order_id = state.order_id
        state.order_id = new_resp.orderId
        state.modification_count = 0
        state.original_qty = state.remaining_qty  # new order, new original
        state.current_limit_price = new_price

        # Update SL pairing to reference new entry order ID
        await self.sl_manager.reparent_sl(old_order_id, state.order_id, account)

        logger.info("cancel_replace_completed",
                   account_id=account.account_id,
                   old_order=old_order_id,
                   new_order=state.order_id,
                   new_price=new_price)
    else:
        # Normal modify
        await rate_limiter.acquire("MODIFY")
        try:
            await self._dhan_modify_order(account, DhanModifyRequest(
                dhanClientId=account.dhan_client_id,
                orderId=state.order_id,
                orderType="LIMIT",
                legName=None,
                quantity=state.original_qty,  # ALWAYS original total qty
                price=new_price,
                triggerPrice=None,
                disclosedQuantity=0,
                validity="DAY",
            ))
            state.modification_count += 1
            state.current_limit_price = new_price

        except OrderNotModifiable:
            # Race: filled or cancelled between our check and modify
            await rate_limiter.acquire("POLL")
            fresh = await self._dhan_get_order_status(account, state.order_id)
            state.apply_update(fresh)
            # Next loop iteration handles the new status

        # Post-modify verification: confirm the modify was actually applied
        await rate_limiter.acquire("POLL")
        verified = await self._dhan_get_order_status(account, state.order_id)
        state.apply_update(verified)
```

#### Order Cancellation: DELETE /v2/orders/{order-id}

```python
# DELETE https://api.dhan.co/v2/orders/{orderId}
# Headers: {"access-token": "..."}
# No request body.

class DhanCancelResponse(pydantic.BaseModel):
    orderId: str
    orderStatus: str  # "CANCELLED" on success
```

**Cancellation edge cases:**
- Order already filled → Dhan returns error. Catch and treat as filled.
- Order already cancelled → Dhan returns error. Idempotent — ignore.
- Order in TRANSIT (not yet on exchange) → Cancellation may fail. Retry after 500ms.

#### Order Slicing: POST /v2/orders/slicing

Used when order quantity exceeds the exchange's freeze quantity limit. NSE sets per-instrument freeze limits (e.g., NIFTY options: 1800 qty = ~27 lots). Orders above this limit must be sliced into multiple child orders.

```python
class DhanSlicingRequest(pydantic.BaseModel):
    """Request body for POST /v2/orders/slicing.
    Same fields as regular order, but Dhan auto-splits into children."""
    dhanClientId: str
    transactionType: Literal["BUY", "SELL"]
    exchangeSegment: str
    productType: str
    orderType: str
    validity: str
    securityId: str
    quantity: int                    # total qty (may exceed freeze limit)
    price: float
    triggerPrice: float | None
    disclosedQuantity: int | None
    afterMarketOrder: bool
    amoTime: str | None
    boProfitValue: float | None
    boStopLossValue: float | None
    correlationId: str | None

class DhanSlicingResponse(pydantic.BaseModel):
    """Response: list of child order IDs."""
    orderId: str                    # parent order ID
    # Note: Dhan creates child orders automatically.
    # Each child appears as a separate order in order updates.
    # The parent orderId is referenced in child updates.
```

**When to use slicing:**

```python
def should_slice(self, quantity: int, freeze_qty: int) -> bool:
    """Check if order needs slicing based on exchange freeze limits."""
    return quantity > freeze_qty

# freeze_qty comes from the instrument master CSV
# Examples:
#   NIFTY options: freeze_qty = 1800 (27.7 lots at 65/lot)
#   BANKNIFTY options: freeze_qty = 900 (60 lots at 15/lot)
#   NIFTY futures: freeze_qty = 1800
```

**OMS handling of sliced orders:** When the OMS detects `quantity > freeze_qty`, it uses the slicing endpoint instead of the regular order endpoint. The fill management loop then tracks each child order independently. SL is placed for the TOTAL quantity (SL orders are not subject to freeze limits on Dhan).

#### Order Status: GET /v2/orders/{order-id}

REST endpoint for polling order status. Used as fallback when WS is unhealthy and for post-modify verification.

```python
# GET https://api.dhan.co/v2/orders/{orderId}
# Headers: {"access-token": "..."}

class DhanOrderDetail(pydantic.BaseModel):
    """Response from GET /v2/orders/{orderId}."""
    orderId: str
    correlationId: str | None
    orderStatus: Literal[
        "TRANSIT",        # sent to exchange, not yet acknowledged
        "PENDING",        # on exchange, waiting to match
        "TRADED",         # fully filled
        "PART_TRADED",    # partially filled, remaining still pending
        "CANCELLED",      # cancelled (by us or exchange)
        "REJECTED",       # rejected by Dhan risk or exchange
        "EXPIRED",        # DAY order expired at session close
    ]
    transactionType: str
    exchangeSegment: str
    productType: str
    orderType: str
    validity: str
    securityId: str
    quantity: int                    # original total quantity
    filledQty: int                   # how many units filled so far
    remainingQuantity: int           # quantity - filledQty
    price: float                     # current limit price
    triggerPrice: float | None
    averageTradedPrice: float        # volume-weighted avg fill price
    exchangeOrderId: str | None      # exchange's internal order ID
    exchangeTime: str | None         # exchange timestamp (ISO 8601)
    createTime: str                  # order creation time
    updateTime: str                  # last update time
    legName: str | None
    drvExpiryDate: str | None        # derivative expiry (YYYY-MM-DD)
    drvOptionType: str | None        # "CALL" | "PUT"
    drvStrikePrice: float | None
```

**PART_TRADED handling:**

When `orderStatus == "PART_TRADED"`, the order is partially filled with remaining quantity still pending on the exchange. Critical actions:

1. `filledQty` tells us how many units have been filled
2. The paired SL order qty MUST be updated to match `filledQty` (not the original qty)
3. The fill management loop decides whether to continue waiting (within patience) or abandon (cancel remaining, keep partial fill + adjusted SL)
4. `averageTradedPrice` gives the weighted average across all partial fills so far

#### Trades of Order: GET /v2/trades/{order-id}

Returns individual trade executions for an order. A single order may fill in multiple trades at different prices and quantities.

```python
# GET https://api.dhan.co/v2/trades/{orderId}
# Headers: {"access-token": "..."}

class DhanTradeDetail(pydantic.BaseModel):
    """One trade execution within an order."""
    orderId: str
    exchangeOrderId: str
    tradingSymbol: str
    securityId: str
    transactionType: str
    exchangeSegment: str
    productType: str
    orderType: str
    tradedQuantity: int              # qty filled in THIS trade
    tradedPrice: float               # price of THIS trade
    exchangeTime: str                # exchange timestamp for this trade
    drvExpiryDate: str | None
    drvOptionType: str | None
    drvStrikePrice: float | None

class DhanTradesResponse(pydantic.BaseModel):
    """Response: list of trades for this order."""
    trades: list[DhanTradeDetail]
```

**Why GET /v2/trades matters:** The `averageTradedPrice` in the order detail endpoint is the volume-weighted average. But for accurate cost analysis and audit trail, we need each individual trade (price × qty). The Position Tracker calls this endpoint after every order completion to record per-trade details.

#### Live Order Update WebSocket: wss://api-order-update.dhan.co

Primary channel for real-time order status updates. One WebSocket connection per account.

**Connection and authentication:**

```python
class DhanOrderWS:
    """
    WebSocket client for Dhan live order updates.

    One instance per account. Authenticates with the account's access token.
    Receives real-time updates for all orders on that account.
    """

    WS_URL = "wss://api-order-update.dhan.co"

    def __init__(self, account: "Account"):
        self.account = account
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect and authenticate."""
        self._ws = await websockets.connect(
            self.WS_URL,
            extra_headers={
                "access-token": self.account.api_key,
            },
            ping_interval=30,        # send ping every 30s to keep alive
            ping_timeout=10,
            close_timeout=5,
        )

        # Send authentication message
        auth_msg = {
            "LoginReq": {
                "MsgCode": 42,
                "ClientId": self.account.dhan_client_id,
                "Token": self.account.api_key,
            }
        }
        await self._ws.send(orjson.dumps(auth_msg))

        # Wait for auth response
        resp = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
        auth_resp = orjson.loads(resp)
        if auth_resp.get("type") == "order_alert":
            # Dhan sends existing pending orders as first messages after auth
            # Process them to sync state
            pass
        self._connected = True
        logger.info("dhan_order_ws_connected",
                   account_id=self.account.account_id)
```

**Order update message schema (from Dhan WS):**

```python
class DhanWSOrderUpdate(pydantic.BaseModel):
    """
    Schema of order update messages received on the Dhan order WS.

    Dhan sends these in real-time when any order status changes:
    placed, modified, partially filled, fully filled, cancelled, rejected.
    """
    type: str                        # "order_alert"
    orderId: str
    correlationId: str | None
    orderStatus: Literal[
        "TRANSIT", "PENDING", "TRADED",
        "PART_TRADED", "CANCELLED", "REJECTED", "EXPIRED"
    ]
    transactionType: str
    exchangeSegment: str
    productType: str
    orderType: str
    validity: str
    securityId: str
    quantity: int                    # original total quantity
    filledQty: int
    remainingQuantity: int
    price: float
    triggerPrice: float | None
    averageTradedPrice: float
    exchangeOrderId: str | None
    exchangeTime: str | None
    updateTime: str
```

**WS Demuxer design:**

The demuxer receives all order updates on a per-account WS connection and routes them to the correct per-order fill management loop.

```python
class OrderUpdateDemuxer:
    """
    Receives order updates from Dhan WS and routes them to the correct
    OrderState object. One demuxer per account.

    Architecture:
    1. WS reader task continuously reads from the WebSocket
    2. For each update, look up the OrderState by orderId
    3. Acquire the per-order lock
    4. Apply the update (increment sequence, update fields)
    5. Signal the fill management loop via update_event
    """

    def __init__(self, account: "Account"):
        self.account = account
        self._order_states: dict[str, "OrderState"] = {}

    def register_order(self, order_id: str, state: "OrderState") -> None:
        """Register a new order for update routing."""
        self._order_states[order_id] = state

    def unregister_order(self, order_id: str) -> None:
        """Unregister a completed/cancelled order."""
        self._order_states.pop(order_id, None)

    async def run(self, ws: DhanOrderWS) -> None:
        """
        Main demuxer loop. Reads from WS, routes to OrderState.

        This runs as an asyncio task for the lifetime of the WS connection.
        If the WS disconnects, this task exits and is restarted by
        _restart_on_error wrapper.
        """
        while True:
            raw = await ws.recv()
            try:
                update = DhanWSOrderUpdate.model_validate_json(raw)
            except pydantic.ValidationError:
                logger.warning("ws_update_parse_error",
                             account_id=self.account.account_id,
                             raw=raw[:200])
                continue

            if update.type != "order_alert":
                continue

            state = self._order_states.get(update.orderId)
            if state is None:
                # Order we're not tracking (e.g., manual order placed in Dhan app)
                logger.debug("ws_update_unknown_order",
                           account_id=self.account.account_id,
                           order_id=update.orderId)
                continue

            # Apply update under per-order lock
            async with state.lock:
                state.apply_ws_update(update)
                state.update_event.set()

            logger.debug("ws_update_routed",
                       account_id=self.account.account_id,
                       order_id=update.orderId,
                       status=update.orderStatus,
                       filled=update.filledQty)
```

**WS health monitoring:**

```python
class WSHealthMonitor:
    """Tracks WS connection health per account."""

    def __init__(self, account_id: str):
        self.account_id = account_id
        self.is_healthy: bool = False
        self.last_message_ts: int = 0
        self.reconnect_count: int = 0
        self.disconnect_ts: int | None = None

    def on_message(self) -> None:
        self.last_message_ts = now_ms()
        self.is_healthy = True

    def on_disconnect(self) -> None:
        self.is_healthy = False
        self.disconnect_ts = now_ms()
        self.reconnect_count += 1

    def on_reconnect(self) -> None:
        self.is_healthy = True
        gap_ms = now_ms() - (self.disconnect_ts or now_ms())
        logger.info("ws_reconnected",
                   account_id=self.account_id,
                   gap_ms=gap_ms,
                   reconnect_count=self.reconnect_count)
        self.disconnect_ts = None
```

#### Forever Orders: POST /v2/forever/orders

Forever Orders (also called GTT — Good Till Triggered) persist across trading sessions. They are not cancelled at EOD like DAY orders. This is the mechanism for overnight SL protection for strategies S2, S6 (multi-day), and S7.

```python
class DhanForeverOrderRequest(pydantic.BaseModel):
    """
    Request body for POST /v2/forever/orders.

    Forever Orders have TWO legs:
    - Leg 1: The trigger condition (price crosses trigger)
    - Leg 2: The order to place when triggered (LIMIT order)

    For SL purposes:
    - orderFlag = "SINGLE" (one-leg SL, not OCO)
    - triggerType = "STOP_LOSS" (trigger when price crosses DOWN for long, UP for short)
    """
    dhanClientId: str
    orderFlag: Literal["SINGLE", "OCO"]  # SINGLE for SL, OCO for target+SL
    transactionType: Literal["BUY", "SELL"]
    exchangeSegment: str
    productType: Literal["CNC", "MARGIN"]  # NOT INTRADAY — forever orders are multi-day
    orderType: Literal["LIMIT", "MARKET"]
    securityId: str
    quantity: int
    price: float                    # limit price for the order when triggered
    triggerPrice: float             # price at which the order activates
    correlationId: str | None

class DhanForeverOrderResponse(pydantic.BaseModel):
    orderId: str                    # forever order ID (different namespace from regular orders)
    orderStatus: str
```

**Forever Order for overnight SL (S2 example):**

```python
# S2 holds a NIFTY futures LONG overnight.
# Entry at 24500, SL at 24400 (100 points stop)

forever_sl = DhanForeverOrderRequest(
    dhanClientId="1000000001",
    orderFlag="SINGLE",
    transactionType="SELL",          # exit a LONG position
    exchangeSegment="NSE_FNO",
    productType="MARGIN",            # overnight position = MARGIN, not INTRADAY
    orderType="LIMIT",
    securityId="13",                 # NIFTY futures security ID
    quantity=75,                     # 1 lot NIFTY futures = 75
    price=24390.0,                   # limit price = trigger - 2 ticks (₹0.05 × 2)
    triggerPrice=24400.0,            # SL trigger
    correlationId="SL_S2_forever_abc123",
)

# POST https://api.dhan.co/v2/forever/orders
```

**Forever Order lifecycle:**
1. Created via API → stored on Dhan servers
2. Persists across market sessions (survives EOD, holidays, weekends)
3. When market price crosses triggerPrice → Dhan auto-places a regular LIMIT order
4. The triggered regular order then fills at the exchange
5. Modification: PUT /v2/forever/orders/{orderId}
6. Cancellation: DELETE /v2/forever/orders/{orderId}
7. Max validity: 1 year (Dhan's limit)

**Differences from DAY SL orders:**

| Aspect | DAY SL | Forever Order SL |
|--------|--------|-----------------|
| Validity | Expires at 15:30 IST session close | Persists until triggered, cancelled, or 1-year expiry |
| Product type | INTRADAY | CNC or MARGIN |
| Order type | STOP_LOSS (regular order) | Forever order (separate API) |
| WS updates | Yes, on regular order WS | Limited — trigger notifications only |
| Modification | PUT /v2/orders/{id} | PUT /v2/forever/orders/{id} |
| Use case | Intraday strategies (S1, S3, S5) | Overnight/multi-day (S2, S6, S7) |
| OPS cost | Counts against 10 OPS | Counts against 10 OPS |

---

### Mandatory Server-Side Stop-Loss

Every entry order placed by the OMS is paired with a server-side stop-loss order. This is the sole protection during system outage (process crash, EC2 reboot, Redis failure, network partition). It is not optional. There is no code path that places an entry without an SL.

#### Pairing Logic

```python
class SLPairing(pydantic.BaseModel):
    """Tracks the entry-SL pair for one position leg, one account."""
    entry_order_id: str
    sl_order_id: str
    account_id: str
    strategy_id: str
    instrument_id: str
    sl_type: Literal["DAY", "FOREVER"]
    sl_qty: int                      # MUST match entry filled_qty at all times
    sl_trigger: float
    sl_limit: float
    sl_status: Literal["PENDING", "TRIGGERED", "CANCELLED", "REJECTED", "EXPIRED", "UNKNOWN"]
    last_verified_ts: int            # epoch ms of last REST verification
    created_ts: int

async def place_entry_with_sl(
    self,
    order: "ResolvedOrder",
    account: "Account",
) -> tuple[str | None, str | None]:
    """
    Place entry order + paired SL order for one account.

    Returns (entry_order_id, sl_order_id) on success.
    Returns (None, None) if either fails.

    The SL is placed IMMEDIATELY after the entry. If SL placement fails,
    the entry is cancelled. A position without SL is NEVER acceptable.
    """
    rate_limiter = self._account_rate_limiters[account.account_id]

    # 1. Place entry order
    await rate_limiter.acquire("NEW")
    entry_resp = await self._dhan_place_order(account, DhanOrderRequest(
        dhanClientId=account.dhan_client_id,
        transactionType=order.transaction_type,
        exchangeSegment=order.exchange_segment,
        productType=self._product_type(order),
        orderType="LIMIT",
        validity="DAY",
        securityId=order.instrument_id,
        quantity=order.quantity,
        price=order.limit_price,
        triggerPrice=None,
        disclosedQuantity=0,
        afterMarketOrder=False,
        amoTime=None,
        boProfitValue=None,
        boStopLossValue=None,
        correlationId=self._make_correlation_id(order, account),
    ))

    if entry_resp.orderStatus == "REJECTED":
        logger.warning("entry_rejected",
                      account_id=account.account_id,
                      strategy_id=order.signal.strategy_id,
                      reason="broker_rejection")
        return (None, None)

    entry_order_id = entry_resp.orderId

    # 2. Register entry in demuxer for WS updates
    entry_state = OrderState(
        order_id=entry_order_id,
        account_id=account.account_id,
        original_qty=order.quantity,
        current_limit_price=order.limit_price,
    )
    self._demuxers[account.account_id].register_order(entry_order_id, entry_state)
    self._order_states[(account.account_id, entry_order_id)] = entry_state

    # 3. Place paired SL
    sl_txn = "SELL" if order.transaction_type == "BUY" else "BUY"

    if order.sl_type == "FOREVER":
        # Overnight/multi-day: use Forever Order API
        await rate_limiter.acquire("SL")
        sl_resp = await self._dhan_place_forever_order(account, DhanForeverOrderRequest(
            dhanClientId=account.dhan_client_id,
            orderFlag="SINGLE",
            transactionType=sl_txn,
            exchangeSegment=order.exchange_segment,
            productType="MARGIN",     # overnight = MARGIN
            orderType="LIMIT",
            securityId=order.instrument_id,
            quantity=order.quantity,
            price=order.sl_limit_price,
            triggerPrice=order.sl_trigger_price,
            correlationId=f"SL_{self._make_correlation_id(order, account)}",
        ))
    else:
        # Intraday: use regular SL order with DAY validity
        await rate_limiter.acquire("SL")
        sl_resp = await self._dhan_place_order(account, DhanOrderRequest(
            dhanClientId=account.dhan_client_id,
            transactionType=sl_txn,
            exchangeSegment=order.exchange_segment,
            productType="INTRADAY",
            orderType="STOP_LOSS",
            validity="DAY",
            securityId=order.instrument_id,
            quantity=order.quantity,
            price=order.sl_limit_price,
            triggerPrice=order.sl_trigger_price,
            disclosedQuantity=0,
            afterMarketOrder=False,
            amoTime=None,
            boProfitValue=None,
            boStopLossValue=None,
            correlationId=f"SL_{self._make_correlation_id(order, account)}",
        ))

    if sl_resp.orderStatus == "REJECTED":
        # SL failed — cancel entry immediately.
        # Position without SL is NEVER acceptable.
        logger.critical("sl_placement_failed_cancelling_entry",
                       account_id=account.account_id,
                       entry_id=entry_order_id,
                       strategy_id=order.signal.strategy_id)
        await rate_limiter.acquire("EXIT")
        try:
            await self._dhan_cancel_order(account, entry_order_id)
        except Exception:
            # Entry may have filled while we tried to cancel.
            # Query status to find out.
            await rate_limiter.acquire("POLL")
            status = await self._dhan_get_order_status(account, entry_order_id)
            if status.orderStatus in ("TRADED", "PART_TRADED"):
                # Entry filled but SL failed — EMERGENCY
                # Place an immediate market exit
                logger.critical("entry_filled_sl_failed_emergency_exit",
                              account_id=account.account_id,
                              entry_id=entry_order_id,
                              filled_qty=status.filledQty)
                await rate_limiter.acquire("EXIT")
                await self._dhan_place_order(account, DhanOrderRequest(
                    dhanClientId=account.dhan_client_id,
                    transactionType=sl_txn,
                    exchangeSegment=order.exchange_segment,
                    productType=self._product_type(order),
                    orderType="MARKET",
                    validity="IOC",
                    securityId=order.instrument_id,
                    quantity=status.filledQty,
                    price=0,
                    triggerPrice=None,
                    disclosedQuantity=0,
                    afterMarketOrder=False,
                    amoTime=None,
                    boProfitValue=None,
                    boStopLossValue=None,
                    correlationId=f"EMRG_{order.signal.strategy_id}",
                ))
                await telegram.send(CRITICAL,
                    f"EMERGENCY EXIT: {account.account_id} entry filled but SL rejected. "
                    f"Market exit placed for {status.filledQty} qty.")
        return (None, None)

    sl_order_id = sl_resp.orderId

    # 4. Register SL pairing
    pairing = SLPairing(
        entry_order_id=entry_order_id,
        sl_order_id=sl_order_id,
        account_id=account.account_id,
        strategy_id=order.signal.strategy_id,
        instrument_id=order.instrument_id,
        sl_type=order.sl_type,
        sl_qty=order.quantity,
        sl_trigger=order.sl_trigger_price,
        sl_limit=order.sl_limit_price,
        sl_status="PENDING",
        last_verified_ts=now_ms(),
        created_ts=now_ms(),
    )
    self.sl_manager.register(pairing)

    # 5. Register SL in demuxer if it's a regular (DAY) order
    if order.sl_type != "FOREVER":
        sl_state = OrderState(
            order_id=sl_order_id,
            account_id=account.account_id,
            original_qty=order.quantity,
            current_limit_price=order.sl_limit_price,
            is_sl=True,
        )
        self._demuxers[account.account_id].register_order(sl_order_id, sl_state)
        self._order_states[(account.account_id, sl_order_id)] = sl_state

    logger.info("entry_sl_paired",
               account_id=account.account_id,
               strategy_id=order.signal.strategy_id,
               entry_id=entry_order_id,
               sl_id=sl_order_id,
               sl_type=order.sl_type)

    return (entry_order_id, sl_order_id)
```

#### SL Validity by Strategy

| Strategy | Position Duration | SL Type | SL Validity | Rationale |
|----------|------------------|---------|-------------|-----------|
| S1 (ORB) | Intraday | Regular SL | DAY | Position closed by EOD flatten |
| S2 (Overnight) | Multi-day (overnight) | **Forever Order** | Until triggered/cancelled | **CRITICAL-1 fix**: DAY SL expires at 15:30, leaving overnight position unprotected |
| S3 (VWAP MR) | Intraday | Regular SL | DAY | Position closed by EOD flatten |
| S4 (Momentum) | Multi-day (monthly hold) | **Forever Order** | Until triggered/cancelled | Equity positions held across sessions |
| S5 (Expiry Day) | Intraday (0-DTE) | Regular SL | DAY | Expires worthless at EOD anyway |
| S6 (Vol Premium) | Variable: intraday or multi-day | **Forever Order** if multi-day, Regular SL if intraday | Depends on config | Short options may be held overnight |
| S7 (Pairs) | Multi-day | **Forever Order** | Until triggered/cancelled | Pairs positions held across sessions |

#### SL Quantity Adjustment on Every Partial Fill (CRITICAL-2 Fix)

When an entry order is partially filled, the SL order quantity MUST be immediately adjusted to match the filled quantity. If not adjusted, the SL covers more quantity than the actual position, creating a naked short/long if it triggers.

**The problem:**
```
1. Place entry: BUY 650 qty at 245.50
2. Place SL: SELL 650 qty trigger at 240.00
3. Partial fill: 325 filled, 325 remaining
4. SL still set at 650 qty
5. If SL triggers: SELL 650 — but we only hold 325
6. We're now SHORT 325 that we don't own → catastrophic
```

**The fix:** On EVERY `PART_TRADED` update, immediately modify the SL order quantity.

```python
class SLLifecycleManager:
    """
    First-class SL lifecycle management. Every SL state transition is tracked,
    verified, and acted upon.
    """

    _pairings: dict[tuple[str, str], SLPairing]
    # Key: (account_id, entry_order_id) → SLPairing

    def register(self, pairing: SLPairing) -> None:
        key = (pairing.account_id, pairing.entry_order_id)
        self._pairings[key] = pairing

    async def on_entry_partial_fill(
        self,
        account: "Account",
        entry_order_id: str,
        filled_qty: int,
    ) -> None:
        """
        IMMEDIATELY modify SL qty to match filled qty.

        Called on EVERY PART_TRADED update — not just on the timeout/abandon path.
        This is the CRITICAL-2 fix: the SL must always reflect the actual position.

        Args:
            account: The account this order belongs to
            entry_order_id: The entry order that was partially filled
            filled_qty: How many units have been filled so far (cumulative)
        """
        key = (account.account_id, entry_order_id)
        pairing = self._pairings.get(key)
        if pairing is None:
            logger.error("sl_pairing_not_found", entry_id=entry_order_id,
                        account_id=account.account_id)
            return

        if pairing.sl_qty == filled_qty:
            return  # already synced

        rate_limiter = self._oms._account_rate_limiters[account.account_id]

        if pairing.sl_type == "FOREVER":
            # Modify Forever Order
            await rate_limiter.acquire("SL")
            await self._oms._dhan_modify_forever_order(account, pairing.sl_order_id,
                quantity=filled_qty,
                price=pairing.sl_limit,
                trigger_price=pairing.sl_trigger)
        else:
            # Modify regular SL order
            # NOTE: For SL modify, quantity = NEW total (not original total like entry modify)
            # This is because the SL is being REDUCED, not repriced.
            # Dhan interprets SL modify qty as the new desired qty.
            await rate_limiter.acquire("SL")
            await self._oms._dhan_modify_order(account, DhanModifyRequest(
                dhanClientId=account.dhan_client_id,
                orderId=pairing.sl_order_id,
                orderType="STOP_LOSS",
                legName=None,
                quantity=filled_qty,    # new SL qty = filled qty
                price=pairing.sl_limit,
                triggerPrice=pairing.sl_trigger,
                disclosedQuantity=0,
                validity="DAY",
            ))

        old_qty = pairing.sl_qty
        pairing.sl_qty = filled_qty
        logger.info("sl_qty_synced_on_partial",
                   account_id=account.account_id,
                   entry_id=entry_order_id,
                   sl_id=pairing.sl_order_id,
                   old_qty=old_qty,
                   new_qty=filled_qty)

    async def on_entry_fully_filled(
        self,
        account: "Account",
        entry_order_id: str,
        filled_qty: int,
    ) -> None:
        """
        Entry fully filled. SL qty should already match (from partial fill updates).
        Verify via REST — don't trust WS alone for SL state.
        """
        key = (account.account_id, entry_order_id)
        pairing = self._pairings.get(key)
        if pairing is None:
            return

        # Verify SL is still active
        await self._verify_sl_active(account, pairing)

        # Ensure qty matches
        if pairing.sl_qty != filled_qty:
            logger.warning("sl_qty_mismatch_on_full_fill",
                         account_id=account.account_id,
                         sl_qty=pairing.sl_qty,
                         filled_qty=filled_qty)
            await self.on_entry_partial_fill(account, entry_order_id, filled_qty)

    async def on_normal_exit(
        self,
        account: "Account",
        entry_order_id: str,
    ) -> None:
        """
        Strategy exits normally (target hit, signal reversal, EOD flatten).
        Cancel the SL order — we're managing the exit ourselves.

        Race condition handled: SL may trigger between our exit decision and
        the cancel attempt. If cancel fails because SL already triggered,
        the position is already closed by the SL — skip the exit order.
        """
        key = (account.account_id, entry_order_id)
        pairing = self._pairings.get(key)
        if pairing is None:
            return

        rate_limiter = self._oms._account_rate_limiters[account.account_id]

        try:
            if pairing.sl_type == "FOREVER":
                await rate_limiter.acquire("SL")
                await self._oms._dhan_cancel_forever_order(account, pairing.sl_order_id)
            else:
                await rate_limiter.acquire("SL")
                await self._oms._dhan_cancel_order(account, pairing.sl_order_id)

            pairing.sl_status = "CANCELLED"

        except OrderNotCancellable:
            # SL may have already triggered — check
            await self._verify_sl_active(account, pairing)
            if pairing.sl_status == "TRIGGERED":
                logger.warning("sl_triggered_during_exit",
                             account_id=account.account_id,
                             entry_id=entry_order_id)
                # Position already closed by SL. Caller must skip exit order.
                return
            if pairing.sl_status == "CANCELLED":
                # Already cancelled (race with EOD flatten)
                return
            # Unknown state — log and continue
            logger.error("sl_cancel_failed_unknown_state",
                        account_id=account.account_id,
                        sl_status=pairing.sl_status)

    async def on_eod_for_overnight(
        self,
        account: "Account",
        entry_order_id: str,
    ) -> None:
        """
        EOD processing for an overnight position.

        DAY SL orders expire at 15:30 (session close). For overnight positions
        (S2, S6 multi-day, S7), we MUST place a Forever Order SL before the
        DAY SL expires. This is the CRITICAL-1 fix.

        Sequence:
        1. Place Forever Order SL (survives overnight)
        2. If Forever Order succeeds: cancel the DAY SL (or let it expire)
        3. If Forever Order fails: place AMO SL (After Market Order, active at next open)
        4. If AMO also fails: Telegram CRITICAL — position is unprotected overnight
        """
        key = (account.account_id, entry_order_id)
        pairing = self._pairings.get(key)
        if pairing is None:
            return

        if pairing.sl_type == "FOREVER":
            # Already a Forever Order — nothing to convert
            return

        rate_limiter = self._oms._account_rate_limiters[account.account_id]
        sl_txn = "SELL" if pairing.sl_trigger < self._get_current_price(pairing.instrument_id) else "BUY"

        # 1. Place Forever Order SL
        await rate_limiter.acquire("SL")
        try:
            forever_resp = await self._oms._dhan_place_forever_order(
                account,
                DhanForeverOrderRequest(
                    dhanClientId=account.dhan_client_id,
                    orderFlag="SINGLE",
                    transactionType=sl_txn,
                    exchangeSegment=self._get_exchange_segment(pairing.instrument_id),
                    productType="MARGIN",
                    orderType="LIMIT",
                    securityId=pairing.instrument_id,
                    quantity=pairing.sl_qty,
                    price=pairing.sl_limit,
                    triggerPrice=pairing.sl_trigger,
                    correlationId=f"FRVR_SL_{pairing.strategy_id}_{entry_order_id[:8]}",
                ))

            if forever_resp.orderStatus != "REJECTED":
                # Success — update pairing to reference Forever Order
                old_sl_id = pairing.sl_order_id
                pairing.sl_order_id = forever_resp.orderId
                pairing.sl_type = "FOREVER"
                pairing.last_verified_ts = now_ms()

                # Cancel old DAY SL (it will expire anyway, but cancel to be clean)
                try:
                    await rate_limiter.acquire("SL")
                    await self._oms._dhan_cancel_order(account, old_sl_id)
                except Exception:
                    pass  # old SL will expire at EOD anyway

                logger.info("sl_converted_to_forever",
                           account_id=account.account_id,
                           entry_id=entry_order_id,
                           old_sl=old_sl_id,
                           new_sl=forever_resp.orderId)
                return

        except Exception as e:
            logger.error("forever_order_placement_failed",
                        account_id=account.account_id,
                        error=str(e))

        # 2. Forever Order failed — try AMO SL
        await rate_limiter.acquire("SL")
        try:
            amo_resp = await self._oms._dhan_place_order(account, DhanOrderRequest(
                dhanClientId=account.dhan_client_id,
                transactionType=sl_txn,
                exchangeSegment=self._get_exchange_segment(pairing.instrument_id),
                productType="MARGIN",
                orderType="STOP_LOSS",
                validity="DAY",
                securityId=pairing.instrument_id,
                quantity=pairing.sl_qty,
                price=pairing.sl_limit,
                triggerPrice=pairing.sl_trigger,
                disclosedQuantity=0,
                afterMarketOrder=True,
                amoTime="OPEN",       # active immediately at next market open
                boProfitValue=None,
                boStopLossValue=None,
                correlationId=f"AMO_SL_{pairing.strategy_id}_{entry_order_id[:8]}",
            ))

            if amo_resp.orderStatus != "REJECTED":
                old_sl_id = pairing.sl_order_id
                pairing.sl_order_id = amo_resp.orderId
                pairing.sl_type = "DAY"  # AMO becomes DAY order at open
                pairing.last_verified_ts = now_ms()
                logger.info("sl_converted_to_amo",
                           account_id=account.account_id,
                           entry_id=entry_order_id)
                return

        except Exception as e:
            logger.error("amo_sl_placement_failed",
                        account_id=account.account_id,
                        error=str(e))

        # 3. Both failed — CRITICAL: position unprotected overnight
        logger.critical("overnight_sl_placement_failed_all_methods",
                       account_id=account.account_id,
                       entry_id=entry_order_id,
                       strategy_id=pairing.strategy_id,
                       instrument_id=pairing.instrument_id)
        await telegram.send(CRITICAL,
            f"OVERNIGHT POSITION UNPROTECTED: "
            f"Account {account.account_id}, "
            f"Strategy {pairing.strategy_id}, "
            f"Instrument {pairing.instrument_id}. "
            f"Forever Order AND AMO SL both failed. MANUAL INTERVENTION REQUIRED.")

    async def _verify_sl_active(self, account: "Account", pairing: SLPairing) -> None:
        """REST verification — don't trust WS alone for SL state."""
        rate_limiter = self._oms._account_rate_limiters[account.account_id]

        if pairing.sl_type == "FOREVER":
            await rate_limiter.acquire("POLL")
            resp = await self._oms._dhan_get_forever_order_status(
                account, pairing.sl_order_id)
            pairing.sl_status = resp.orderStatus
        else:
            await rate_limiter.acquire("POLL")
            resp = await self._oms._dhan_get_order_status(account, pairing.sl_order_id)
            pairing.sl_status = resp.orderStatus

        pairing.last_verified_ts = now_ms()

    async def periodic_verification(self) -> None:
        """
        Every 60 seconds, verify all active SL orders via REST.
        Catches silent rejections, status lag, and broker-side cancellations.

        Runs as a background task in the OMS process.
        """
        for key, pairing in self._pairings.items():
            account_id, entry_order_id = key
            if pairing.sl_status not in ("PENDING",):
                continue  # only check active SLs

            account = self._oms._accounts[account_id]
            await self._verify_sl_active(account, pairing)

            if pairing.sl_status not in ("PENDING", "TRIGGERED"):
                logger.critical("sl_unexpectedly_inactive",
                              account_id=account_id,
                              entry_id=entry_order_id,
                              sl_id=pairing.sl_order_id,
                              sl_status=pairing.sl_status)

                # Immediately re-place SL
                await self._re_place_sl(account, pairing)

    async def _re_place_sl(self, account: "Account", pairing: SLPairing) -> None:
        """Re-place a missing/cancelled SL. Emergency path."""
        rate_limiter = self._oms._account_rate_limiters[account.account_id]
        sl_txn = "SELL" if pairing.sl_trigger < self._get_current_price(pairing.instrument_id) else "BUY"

        if pairing.sl_type == "FOREVER":
            await rate_limiter.acquire("SL")
            resp = await self._oms._dhan_place_forever_order(account, DhanForeverOrderRequest(
                dhanClientId=account.dhan_client_id,
                orderFlag="SINGLE",
                transactionType=sl_txn,
                exchangeSegment=self._get_exchange_segment(pairing.instrument_id),
                productType="MARGIN",
                orderType="LIMIT",
                securityId=pairing.instrument_id,
                quantity=pairing.sl_qty,
                price=pairing.sl_limit,
                triggerPrice=pairing.sl_trigger,
                correlationId=f"REPL_SL_{pairing.strategy_id}",
            ))
        else:
            await rate_limiter.acquire("SL")
            resp = await self._oms._dhan_place_order(account, DhanOrderRequest(
                dhanClientId=account.dhan_client_id,
                transactionType=sl_txn,
                exchangeSegment=self._get_exchange_segment(pairing.instrument_id),
                productType="INTRADAY",
                orderType="STOP_LOSS",
                validity="DAY",
                securityId=pairing.instrument_id,
                quantity=pairing.sl_qty,
                price=pairing.sl_limit,
                triggerPrice=pairing.sl_trigger,
                disclosedQuantity=0,
                afterMarketOrder=False,
                amoTime=None,
                boProfitValue=None,
                boStopLossValue=None,
                correlationId=f"REPL_SL_{pairing.strategy_id}",
            ))

        pairing.sl_order_id = resp.orderId
        pairing.sl_status = "PENDING"
        pairing.last_verified_ts = now_ms()
        logger.info("sl_re_placed",
                   account_id=account.account_id,
                   entry_id=pairing.entry_order_id,
                   new_sl_id=resp.orderId)
```

#### Race Conditions

**Race 1: Entry fills before SL is placed**

Between entry placement (step 1) and SL placement (step 3) in `place_entry_with_sl`, there is a 100-400ms window where the entry may fill. If the entry fills instantly:

- The SL is still placed with the original quantity (correct — matches filled qty)
- The SL is now protecting the full position
- No issue unless the SL placement itself fails (handled by the cancellation/emergency path above)

At ₹50L scale, worst-case exposure during a flash crash in this 200ms window (65 lots × 200pt move × 200ms): ~₹130. Acceptable. Flag for review at ₹5Cr+.

**Race 2: SL triggers while we're placing an exit order**

When a strategy signals an exit (or EOD flatten begins):

```
1. We decide to exit the position
2. We call sl_manager.on_normal_exit() to cancel the SL
3. Between step 1 and step 2, the SL triggers
4. SL fills: position is already closed
5. Our exit order would create a new opposite position!
```

**Handling:**

```python
async def place_exit_order(
    self,
    account: "Account",
    position: "Position",
    exit_order: "ResolvedOrder",
) -> "OrderResult":
    """
    Place an exit order for an existing position.

    MUST cancel SL first. If SL has triggered, skip the exit.
    """
    entry_order_id = position.entry_order_id

    # 1. Cancel the SL
    await self.sl_manager.on_normal_exit(account, entry_order_id)

    # 2. Check if SL triggered (on_normal_exit updates pairing.sl_status)
    key = (account.account_id, entry_order_id)
    pairing = self.sl_manager._pairings.get(key)
    if pairing and pairing.sl_status == "TRIGGERED":
        logger.info("sl_triggered_before_exit_skip",
                   account_id=account.account_id,
                   entry_id=entry_order_id)
        return OrderResult(
            outcome="SKIPPED_SL_TRIGGERED",
            account_id=account.account_id,
            filled_qty=0,
        )

    # 3. Place exit order
    rate_limiter = self._account_rate_limiters[account.account_id]
    await rate_limiter.acquire("EXIT")
    exit_resp = await self._dhan_place_order(account, DhanOrderRequest(
        dhanClientId=account.dhan_client_id,
        transactionType=exit_order.transaction_type,
        exchangeSegment=exit_order.exchange_segment,
        productType=self._product_type(exit_order),
        orderType="LIMIT",
        validity="DAY",
        securityId=exit_order.instrument_id,
        quantity=position.quantity,
        price=exit_order.limit_price,
        triggerPrice=None,
        disclosedQuantity=0,
        afterMarketOrder=False,
        amoTime=None,
        boProfitValue=None,
        boStopLossValue=None,
        correlationId=f"EXIT_{exit_order.signal.strategy_id}_{account.account_id[:8]}",
    ))

    if exit_resp.orderStatus == "REJECTED":
        # Exit rejected — check if SL triggered in the meantime
        await self.sl_manager._verify_sl_active(account, pairing)
        if pairing.sl_status == "TRIGGERED":
            return OrderResult(outcome="SKIPPED_SL_TRIGGERED", ...)
        # Real rejection — Telegram CRITICAL
        logger.critical("exit_order_rejected",
                       account_id=account.account_id)
        return OrderResult(outcome="REJECTED", ...)

    # 4. Run fill management loop for the exit
    return await self.manage_order(exit_resp.orderId, exit_order, account)
```

---

### Fill Management Loop

The fill management loop is the core execution engine. One loop runs per order per account. It monitors order status, reprices when the market moves away, handles partial fills, and decides when to abandon.

#### Order State Machine

```
                  place_order()
                       │
                       ▼
                 ┌───────────┐
                 │  TRANSIT   │  (sent to exchange, awaiting ack)
                 └─────┬─────┘
                       │ exchange acks
                       ▼
                 ┌───────────┐
          ┌─────│  PENDING   │◄────────────────────┐
          │     └──┬──┬──┬───┘                      │
          │        │  │  │                          │
          │        │  │  │ partial fill             │
          │        │  │  ▼                          │
          │        │  │ ┌─────────────┐             │
          │        │  │ │ PART_TRADED │─────────────┘
          │        │  │ │             │  reprice (modify or cancel-replace)
          │        │  │ └──┬──────────┘
          │        │  │    │
          │        │  │    │ fully filled
          │        │  │    ▼
          │        │  │ ┌───────────┐
          │        │  └►│  TRADED   │  terminal: order complete
          │        │    └───────────┘
          │        │
          │        │ timeout or abandon
          │        ▼
          │  ┌───────────┐
          │  │ CANCELLED  │  terminal: order cancelled by us
          │  └───────────┘
          │
          │ rejected by exchange
          ▼
    ┌───────────┐
    │ REJECTED   │  terminal: order rejected
    └───────────┘

    ┌───────────┐
    │  EXPIRED   │  terminal: DAY order expired at session close
    └───────────┘
```

**Terminal states:** TRADED, CANCELLED, REJECTED, EXPIRED. Once terminal, no further actions.

**Non-terminal states:** TRANSIT, PENDING, PART_TRADED. The fill management loop runs while the order is in any non-terminal state.

#### OrderState

```python
class OrderState:
    """
    Mutable per-order state. One instance per order per account.

    This is a process-local object, NOT a Pydantic DTO. It is never serialized
    directly. It contains asyncio primitives (Lock, Event) that cannot be serialized.

    Thread safety: all mutations go through the per-order Lock.
    Ordering: a monotonic sequence number prevents stale-update races.
    """

    def __init__(
        self,
        order_id: str,
        account_id: str,
        original_qty: int,
        current_limit_price: float,
        is_sl: bool = False,
    ):
        self.order_id = order_id
        self.account_id = account_id
        self.lock = asyncio.Lock()
        self.sequence: int = 0
        self.update_event = asyncio.Event()

        self.status: str = "TRANSIT"
        self.original_qty = original_qty
        self.filled_qty: int = 0
        self.remaining_qty: int = original_qty
        self.avg_fill_price: float = 0.0
        self.current_limit_price = current_limit_price
        self.modification_count: int = 0
        self.placed_at_ms: int = now_ms()
        self.last_update_ms: int = now_ms()
        self.is_sl = is_sl

        # For cancel-replace tracking
        self._previous_order_ids: list[str] = []

    def apply_ws_update(self, update: DhanWSOrderUpdate) -> None:
        """
        Apply a WS update to this state. Called UNDER LOCK by the demuxer.

        Sequence number prevents stale updates: if a WS message arrives
        after a REST poll that contained newer data, the WS message is
        discarded.
        """
        self.sequence += 1
        self.status = update.orderStatus
        self.filled_qty = update.filledQty
        self.remaining_qty = update.remainingQuantity
        self.avg_fill_price = update.averageTradedPrice
        self.current_limit_price = update.price
        self.last_update_ms = now_ms()
        self.update_event.set()

    def apply_rest_update(self, detail: DhanOrderDetail) -> None:
        """Apply a REST poll update. Same as WS update but from REST source."""
        self.sequence += 1
        self.status = detail.orderStatus
        self.filled_qty = detail.filledQty
        self.remaining_qty = detail.remainingQuantity
        self.avg_fill_price = detail.averageTradedPrice
        self.current_limit_price = detail.price
        self.last_update_ms = now_ms()
        self.update_event.set()
```

#### Fill Management Loop Implementation

```python
class OrderResult(pydantic.BaseModel):
    """Result of a fill management loop."""
    outcome: Literal[
        "FILLED",             # fully filled
        "PARTIAL_FILL",       # partially filled, remainder abandoned
        "TIMEOUT",            # not filled within patience, cancelled
        "REJECTED",           # rejected by broker/exchange
        "SKIPPED_SL_TRIGGERED",  # SL triggered before exit could be placed
        "CANCELLED",          # cancelled by system (global kill, EOD)
    ]
    account_id: str
    strategy_id: str
    signal_id: str
    order_id: str
    filled_qty: int
    remaining_qty: int
    avg_fill_price: float
    modifications: int
    elapsed_ms: int

async def manage_order(
    self,
    entry_order_id: str,
    order: "ResolvedOrder",
    account: "Account",
) -> OrderResult:
    """
    Run the fill management loop for a single order on a single account.

    This is the core execution engine. It:
    1. Waits for WS updates (or polls REST if WS unhealthy)
    2. On PART_TRADED: syncs SL qty, decides whether to reprice or wait
    3. On timeout: reprices or abandons based on elapsed time vs patience
    4. On TRADED: returns success
    5. On REJECTED: returns rejection
    """
    state = self._order_states.get((account.account_id, entry_order_id))
    if state is None:
        return OrderResult(outcome="REJECTED", filled_qty=0, ...)

    params = order.fill_params
    rate_limiter = self._account_rate_limiters[account.account_id]
    ws_health = self._ws_health[account.account_id]
    start_ms = now_ms()

    while True:
        # Wait for WS update or timeout
        try:
            await asyncio.wait_for(
                state.update_event.wait(),
                timeout=params.reprice_interval_s,
            )
            state.update_event.clear()
        except asyncio.TimeoutError:
            # No WS update within reprice interval
            if not ws_health.is_healthy:
                # WS down — poll REST
                await rate_limiter.acquire("POLL")
                detail = await self._dhan_get_order_status(account, state.order_id)
                async with state.lock:
                    state.apply_rest_update(detail)

        elapsed_ms = now_ms() - start_ms
        elapsed_s = elapsed_ms / 1000

        # Per-order lock: atomic read + decide + act
        async with state.lock:
            match state.status:
                case "TRADED":
                    # Fully filled
                    await self.sl_manager.on_entry_fully_filled(
                        account, entry_order_id, state.filled_qty)
                    return OrderResult(
                        outcome="FILLED",
                        account_id=account.account_id,
                        strategy_id=order.signal.strategy_id,
                        signal_id=order.signal.signal_id,
                        order_id=state.order_id,
                        filled_qty=state.filled_qty,
                        remaining_qty=0,
                        avg_fill_price=state.avg_fill_price,
                        modifications=state.modification_count,
                        elapsed_ms=elapsed_ms,
                    )

                case "PART_TRADED":
                    # Partial fill — IMMEDIATELY sync SL qty
                    await self.sl_manager.on_entry_partial_fill(
                        account, entry_order_id, state.filled_qty)

                    # Notify position tracker
                    await self._notify_partial_fill(account, state, order)

                    if elapsed_s >= params.max_patience_s:
                        # Patience exhausted — cancel remaining
                        await rate_limiter.acquire("MODIFY")
                        try:
                            await self._dhan_cancel_order(account, state.order_id)
                        except OrderAlreadyFilled:
                            # Race: filled between check and cancel
                            await rate_limiter.acquire("POLL")
                            detail = await self._dhan_get_order_status(
                                account, state.order_id)
                            state.apply_rest_update(detail)
                            continue  # re-evaluate in next loop

                        # SL already synced to filled_qty
                        return OrderResult(
                            outcome="PARTIAL_FILL",
                            account_id=account.account_id,
                            strategy_id=order.signal.strategy_id,
                            signal_id=order.signal.signal_id,
                            order_id=state.order_id,
                            filled_qty=state.filled_qty,
                            remaining_qty=state.remaining_qty,
                            avg_fill_price=state.avg_fill_price,
                            modifications=state.modification_count,
                            elapsed_ms=elapsed_ms,
                        )

                    elif self._price_has_moved(state, order, account):
                        # Price moved — reprice
                        await self.reprice_order(state, order, params, account)

                case "PENDING":
                    if elapsed_s >= params.max_patience_s:
                        # Patience exhausted, not filled at all — cancel
                        await rate_limiter.acquire("MODIFY")
                        try:
                            await self._dhan_cancel_order(account, state.order_id)
                        except OrderAlreadyFilled:
                            await rate_limiter.acquire("POLL")
                            detail = await self._dhan_get_order_status(
                                account, state.order_id)
                            state.apply_rest_update(detail)
                            continue

                        # Cancel SL too (no position to protect)
                        await self.sl_manager.on_normal_exit(account, entry_order_id)
                        return OrderResult(
                            outcome="TIMEOUT",
                            account_id=account.account_id,
                            strategy_id=order.signal.strategy_id,
                            signal_id=order.signal.signal_id,
                            order_id=state.order_id,
                            filled_qty=0,
                            remaining_qty=state.original_qty,
                            avg_fill_price=0.0,
                            modifications=state.modification_count,
                            elapsed_ms=elapsed_ms,
                        )

                    elif elapsed_s >= params.reprice_interval_s and \
                         self._price_has_moved(state, order, account):
                        await self.reprice_order(state, order, params, account)

                case "REJECTED":
                    await self.sl_manager.on_normal_exit(account, entry_order_id)
                    return OrderResult(
                        outcome="REJECTED",
                        account_id=account.account_id,
                        strategy_id=order.signal.strategy_id,
                        signal_id=order.signal.signal_id,
                        order_id=state.order_id,
                        filled_qty=0,
                        remaining_qty=state.original_qty,
                        avg_fill_price=0.0,
                        modifications=state.modification_count,
                        elapsed_ms=elapsed_ms,
                    )

                case "CANCELLED" | "EXPIRED":
                    # Cancelled externally or expired
                    if state.filled_qty > 0:
                        # Had partial fills — SL already synced
                        return OrderResult(outcome="PARTIAL_FILL", ...)
                    else:
                        await self.sl_manager.on_normal_exit(account, entry_order_id)
                        return OrderResult(outcome="CANCELLED", ...)

                case "TRANSIT":
                    # Still in transit to exchange — wait
                    if elapsed_s > 10:
                        # 10s in transit is unusual — poll REST
                        await rate_limiter.acquire("POLL")
                        detail = await self._dhan_get_order_status(
                            account, state.order_id)
                        state.apply_rest_update(detail)
```

#### Reprice Logic

```python
def compute_new_limit_price(
    self,
    order: "ResolvedOrder",
    params: FillParams,
    account: "Account",
) -> float:
    """
    Compute a new limit price based on current market data and pricing mode.

    Market data source:
    - Primary: Redis LASTTICK:{symbol} (updated per tick by Data Ingester)
    - Fallback (Redis down): Dhan option chain API direct call

    Pricing modes:
    - PASSIVE:     BUY at best_ask,            SELL at best_bid
    - MIDPOINT:    BUY at ceil(mid/tick)*tick,  SELL at floor(mid/tick)*tick
    - AGGRESSIVE:  BUY at best_ask + 1 tick,   SELL at best_bid - 1 tick
    """
    # Get current market data
    market = self._get_current_market_data(order.instrument_id, account)
    tick_size = order.tick_size

    if order.transaction_type == "BUY":
        match params.pricing_mode:
            case "PASSIVE":
                raw = market.best_ask
            case "MIDPOINT":
                mid = (market.best_bid + market.best_ask) / 2
                raw = math.ceil(mid / tick_size) * tick_size
            case "AGGRESSIVE":
                raw = market.best_ask + tick_size
    else:  # SELL
        match params.pricing_mode:
            case "PASSIVE":
                raw = market.best_bid
            case "MIDPOINT":
                mid = (market.best_bid + market.best_ask) / 2
                raw = math.floor(mid / tick_size) * tick_size
            case "AGGRESSIVE":
                raw = market.best_bid - tick_size

    # Round to tick size
    return round(raw / tick_size) * tick_size

def _price_has_moved(
    self,
    state: OrderState,
    order: "ResolvedOrder",
    account: "Account",
) -> bool:
    """Check if market price has moved away from our limit price enough to reprice."""
    market = self._get_current_market_data(order.instrument_id, account)
    tick_size = order.tick_size

    if order.transaction_type == "BUY":
        # If best ask has moved up by >1 tick from our limit, reprice
        return market.best_ask > state.current_limit_price + tick_size
    else:
        # If best bid has moved down by >1 tick from our limit, reprice
        return market.best_bid < state.current_limit_price - tick_size

def _get_current_market_data(
    self,
    instrument_id: str,
    account: "Account",
) -> "MarketSnapshot":
    """
    Get current bid/ask for an instrument.

    Primary: Redis LASTTICK (updated per tick, <50ms latency)
    Fallback: Dhan option chain API (costs 1 OPS, 200-400ms latency)
    """
    try:
        tick_data = self._redis_sync.get(f"LASTTICK:{instrument_id}")
        if tick_data:
            tick = Tick.model_validate_json(tick_data)
            return MarketSnapshot(
                best_bid=tick.bid,
                best_ask=tick.ask,
                ltp=tick.ltp,
                ts=tick.receive_ts,
            )
    except Exception:
        pass

    # Redis down or key missing — fallback to Dhan API
    # This is the OMS degraded mode market data path
    logger.debug("market_data_fallback_to_api",
               instrument_id=instrument_id,
               account_id=account.account_id)
    return self._fetch_market_from_dhan(instrument_id, account)


class MarketSnapshot(pydantic.BaseModel):
    best_bid: float
    best_ask: float
    ltp: float
    ts: int
```

#### Fill Parameters Per Strategy

| Strategy | `reprice_interval_s` | `max_patience_s` | `pricing_mode` | Rationale |
|----------|---------------------|-------------------|----------------|-----------|
| S1 (ORB) | 5 | 15 | AGGRESSIVE | Time-sensitive breakout — need fast fills |
| S2 (Overnight) | 10 | 30 | MIDPOINT | Patient entry at 15:15, not rushing |
| S3 (VWAP MR) | 5 | 20 | AGGRESSIVE | Mean reversion — entry level matters but speed > precision |
| S4 (Momentum) | 30 | 120 | PASSIVE | Monthly rebalance, large order, minimize impact |
| S5 (Expiry Day) | 3 | 10 | AGGRESSIVE | 0-DTE, every second counts, fast fills critical |
| S6 (Vol Premium) | 15 | 60 | PASSIVE | Selling premium, patient entry |
| S7 (Pairs) | 10 | 45 | MIDPOINT | Pairs — moderate urgency, spread matters |

**Pricing modes explained:**

| Mode | BUY price | SELL price | When to use |
|------|-----------|------------|-------------|
| PASSIVE | best_ask | best_bid | Large orders, no urgency, minimize market impact |
| MIDPOINT | ceil(mid/tick) × tick | floor(mid/tick) × tick | Moderate urgency, balanced fill quality |
| AGGRESSIVE | best_ask + 1 tick | best_bid - 1 tick | Time-sensitive signals, fast fills |

---

### Multi-Account Order Fan-Out

When multi-account mode is active (`len(accounts) > 1`), every signal produces N independent order lifecycles — one per enabled account. Signal generation is shared; everything from sizing onward is per-account.

#### Fan-Out Flow

```
StrategySignal arrives at Signal Router
       │
       ▼
Signal Dedup (shared — one check)
       │
       ▼
For EACH enabled account in config.accounts:
       │
       ├── Does this account run this strategy?
       │     (account.enabled_strategies contains signal.strategy_id)
       │     NO → skip
       │     YES ↓
       │
       ├── Capital Allocator (per-account)
       │     → Uses THIS account's capital, weights, Kelly fraction
       │     → Computes lot count for THIS account
       │     → Checks existing position for THIS account
       │     → Returns AllocationResponse per-account
       │
       ├── Instrument Resolver (shared)
       │     → Same security_id for all accounts (same contract)
       │     → Limit price computed once (shared market data)
       │     → Returns ResolvedOrder (same for all accounts)
       │
       ├── Risk Manager (per-account)
       │     → Checks THIS account's margin
       │     → Checks THIS account's position limits
       │     → Checks THIS account's drawdown
       │     → ALSO checks aggregate exposure across all accounts
       │
       ├── OMS.place_entry_with_sl (per-account)
       │     → Uses THIS account's API key
       │     → Uses THIS account's rate limiter (10 OPS)
       │     → Places entry + SL on THIS account
       │     → Returns (entry_id, sl_id) for THIS account
       │
       └── Fill Management Loop (per-account)
             → Runs independently for THIS account
             → Fills may differ (timing, price, partial fills)
             → SL qty sync per-account
             → OrderResult per-account
```

```python
class MultiAccountFanOut:
    """
    Orchestrates order fan-out across N accounts.

    One signal → N parallel order placements + fill management loops.
    Each account operates independently after the fan-out point.
    """

    def __init__(
        self,
        accounts: list["Account"],
        oms: "OMS",
        capital_allocator: "CapitalAllocator",
        risk_manager: "RiskManager",
        instrument_resolver: "InstrumentResolver",
    ):
        self._accounts = {a.account_id: a for a in accounts}
        self._oms = oms
        self._allocator = capital_allocator
        self._risk = risk_manager
        self._resolver = instrument_resolver

    async def fan_out_signal(self, signal: "StrategySignal") -> list["AccountOrderResult"]:
        """
        Fan out a single signal to all enabled accounts.

        Returns a list of per-account results. Accounts that reject the signal
        (insufficient margin, position limits, not enabled for this strategy)
        are included with rejection reasons.
        """
        results: list[AccountOrderResult] = []

        # Resolve instrument ONCE (shared across accounts — same security_id)
        resolved = await self._resolver.resolve(signal)
        if resolved is None:
            logger.warning("instrument_resolution_failed",
                         strategy_id=signal.strategy_id)
            return results

        # Create per-account tasks for parallel execution
        tasks = []
        for account_id, account in self._accounts.items():
            if not account.enabled:
                continue
            if signal.strategy_id not in account.enabled_strategies:
                continue

            task = asyncio.create_task(
                self._process_for_account(signal, resolved, account),
                name=f"fanout_{account_id}_{signal.signal_id[:8]}",
            )
            tasks.append((account_id, task))

        # Execute all accounts in parallel
        # Each account has its own rate limiter — no OPS contention
        for account_id, task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception:
                logger.exception("fanout_account_error",
                               account_id=account_id,
                               signal_id=signal.signal_id)
                results.append(AccountOrderResult(
                    account_id=account_id,
                    outcome="ERROR",
                    error="unexpected_exception",
                ))

        # Log divergence
        outcomes = {r.account_id: r.outcome for r in results}
        if len(set(outcomes.values())) > 1:
            logger.warning("account_divergence_detected",
                         signal_id=signal.signal_id,
                         outcomes=outcomes)

        return results

    async def _process_for_account(
        self,
        signal: "StrategySignal",
        resolved: "ResolvedOrder",
        account: "Account",
    ) -> "AccountOrderResult":
        """Process a signal for one specific account."""

        # 1. Allocate capital for THIS account
        allocation = await self._allocator.request_allocation(
            AllocationRequest(
                account_id=account.account_id,
                strategy_id=signal.strategy_id,
                direction=signal.direction,
                underlying=signal.underlying,
                instrument_type=resolved.instrument_hint,
                estimated_premium=resolved.limit_price,
                lot_size=resolved.lot_size,
                existing_position_qty=self._get_existing_position(
                    account.account_id, signal.strategy_id),
            ),
            account=account,
        )

        if not allocation.approved:
            return AccountOrderResult(
                account_id=account.account_id,
                outcome="ALLOCATION_REJECTED",
                error=allocation.rejection_reason,
            )

        # 2. Adjust resolved order for this account's sizing
        account_order = resolved.copy(deep=True)
        account_order.quantity = allocation.max_lots * resolved.lot_size
        # Limit price and SL levels remain the same (same market, same instrument)

        # 3. Risk check for THIS account
        risk_ok = await self._risk.pre_trade_check(account_order, account)
        if not risk_ok.passed:
            return AccountOrderResult(
                account_id=account.account_id,
                outcome="RISK_REJECTED",
                error=risk_ok.rejection_reason,
            )

        # 4. Place entry + SL for THIS account
        entry_id, sl_id = await self._oms.place_entry_with_sl(account_order, account)
        if entry_id is None:
            return AccountOrderResult(
                account_id=account.account_id,
                outcome="PLACEMENT_FAILED",
                error="entry_or_sl_rejected",
            )

        # 5. Run fill management loop for THIS account
        fill_result = await self._oms.manage_order(entry_id, account_order, account)

        return AccountOrderResult(
            account_id=account.account_id,
            outcome=fill_result.outcome,
            order_id=fill_result.order_id,
            filled_qty=fill_result.filled_qty,
            avg_fill_price=fill_result.avg_fill_price,
            elapsed_ms=fill_result.elapsed_ms,
        )


class AccountOrderResult(pydantic.BaseModel):
    account_id: str
    outcome: str
    order_id: str | None = None
    filled_qty: int = 0
    remaining_qty: int = 0
    avg_fill_price: float = 0.0
    elapsed_ms: int = 0
    error: str | None = None
```

#### Per-Account Rate Limiters

Each Dhan API key gets its own 10 OPS rate limit. With N accounts, the system has N × 10 OPS total. Each account's rate limiter is independent.

```python
class OMS:
    def __init__(self, accounts: list["Account"]):
        # One rate limiter per account — they don't share OPS budget
        self._account_rate_limiters: dict[str, PriorityRateLimiter] = {
            account.account_id: PriorityRateLimiter(rate=10)
            for account in accounts
        }
```

#### Per-Account WS Connections

Each account has its own order update WebSocket connection and demuxer.

```python
class OMS:
    def __init__(self, accounts: list["Account"]):
        # One WS connection per account
        self._ws_connections: dict[str, DhanOrderWS] = {
            account.account_id: DhanOrderWS(account)
            for account in accounts
        }

        # One demuxer per account
        self._demuxers: dict[str, OrderUpdateDemuxer] = {
            account.account_id: OrderUpdateDemuxer(account)
            for account in accounts
        }

        # One WS health monitor per account
        self._ws_health: dict[str, WSHealthMonitor] = {
            account.account_id: WSHealthMonitor(account.account_id)
            for account in accounts
        }
```

#### Account Divergence Handling

Accounts can diverge when one account fills and another rejects.

**Causes of divergence:**
1. Account B has insufficient margin → allocation rejected
2. Account B's API key expired mid-session → all orders fail
3. Account B's order rejected by exchange (position limit, circuit limit)
4. Account A fills at 245.50, Account C fills at 245.55 (normal price variation)
5. Account A fully fills 10 lots, Account B only fills 6 lots (partial fill, different liquidity)

**Divergence tracking:**

```python
class DivergenceTracker:
    """
    Tracks position divergence across accounts for the same strategy.

    Divergence is EXPECTED (lot rounding, partial fills, margin differences).
    This tracker quantifies it for monitoring and compliance reporting.
    """

    async def record_signal_outcome(
        self,
        signal_id: str,
        strategy_id: str,
        results: list[AccountOrderResult],
    ) -> None:
        divergence = DivergenceRecord(
            signal_id=signal_id,
            strategy_id=strategy_id,
            timestamp_ms=now_ms(),
            account_outcomes={
                r.account_id: {
                    "outcome": r.outcome,
                    "filled_qty": r.filled_qty,
                    "avg_fill_price": r.avg_fill_price,
                }
                for r in results
            },
        )

        # Check for actionable divergence
        filled_accounts = [r for r in results if r.outcome == "FILLED"]
        failed_accounts = [r for r in results
                          if r.outcome in ("REJECTED", "PLACEMENT_FAILED", "ERROR")]

        if filled_accounts and failed_accounts:
            logger.warning("divergence_fill_vs_reject",
                         signal_id=signal_id,
                         filled=[r.account_id for r in filled_accounts],
                         failed=[r.account_id for r in failed_accounts])
            await telegram.send(WARNING,
                f"Account divergence on signal {signal_id[:8]}: "
                f"{len(filled_accounts)} filled, {len(failed_accounts)} failed")

        # Price divergence across filled accounts
        if len(filled_accounts) >= 2:
            prices = [r.avg_fill_price for r in filled_accounts]
            max_price_diff = max(prices) - min(prices)
            if max_price_diff > 5.0:  # >₹5 divergence
                logger.warning("divergence_price",
                             signal_id=signal_id,
                             max_diff=max_price_diff)

        # Store for compliance reporting
        await self._store_divergence(divergence)

    async def get_divergence_report(
        self,
        strategy_id: str | None = None,
        lookback_days: int = 30,
    ) -> list["DivergenceRecord"]:
        """For PMS/AIF compliance: per-client divergence report."""
        ...
```

**Policy on divergence:**
- Divergence is monitored but NOT force-synced. Each account runs independently.
- If Account B consistently fails (>3 consecutive signals rejected), disable it and alert operator.
- Per-client NAV computation (SEBI PMS requirement) uses actual per-account fills, not theoretical fills.
- At no point does the system trade one account to "catch up" to another. Divergence is an operational reality.

#### OPS Budget Analysis: Multi-Account

With N accounts, each with 10 OPS, the system has N × 10 OPS total. This RELAXES the OPS constraint.

**Single account (current):**

```
Total OPS: 10

Normal operation (3 strategies with pending orders, WS healthy):
  Entry modifications:  0.73 OPS/s
  No REST polling:       0.00 OPS/s
  New entry burst:       2 OPS (entry + SL)
  Total typical:         ~1-3 OPS/s  →  well within 10

EOD flatten (5 positions):
  Phase 1 (cancel pending): 2×5 = 10 cancels = 1.0s
  Phase 2 (exit orders):    5 orders = 0.5s
  Phase 3 (fill management): shared 10 OPS
  Total: ~3-5 seconds
```

**3 accounts:**

```
Total OPS: 30 (10 per account, independent)

Normal operation:
  Per-account: same ~1-3 OPS/s
  Cross-account: NO contention — each account has its own bucket
  Signal fan-out: 3 entries + 3 SLs = 6 OPS total
    But: 2 OPS per account, all 3 accounts in parallel = 2 OPS time cost
  Total wall-clock time: SAME as single account

EOD flatten (5 positions × 3 accounts = 15 positions):
  Per-account (parallel):
    5 cancels = 0.5s per account
    5 exits = 0.5s per account
    Fill management on 10 OPS per account
  Wall-clock time: SAME as single account (3 accounts run in parallel)
  Total API calls: 3× but on 3× the OPS budget = no congestion
```

**OPS scaling table:**

| Accounts | Total OPS | Per-Signal OPS (entry+SL) | Wall-Clock Time | Concurrent Fill Loops |
|----------|-----------|--------------------------|-----------------|----------------------|
| 1 | 10 | 2 | 200ms | Up to 10 per second |
| 2 | 20 | 4 total, 2 per account | 200ms (parallel) | Up to 20 per second |
| 3 | 30 | 6 total, 2 per account | 200ms (parallel) | Up to 30 per second |
| 5 | 50 | 10 total, 2 per account | 200ms (parallel) | Up to 50 per second |
| 10 | 100 | 20 total, 2 per account | 200ms (parallel) | Up to 100 per second |

**The key insight:** Multi-account does NOT multiply latency. All accounts execute in parallel. The OPS budget scales linearly while wall-clock time stays constant.

---

### Priority Rate Limiter

One `PriorityRateLimiter` instance per account. Ensures high-priority operations (exits, SL) are never starved by low-priority operations (polling, new entries).

```python
import asyncio
import heapq
import time

class PriorityRateLimiter:
    """
    Token bucket with priority-based dispatch.

    When multiple operations compete for a limited OPS budget, higher-priority
    operations dequeue first. This prevents a burst of POLL requests from
    delaying an EXIT order.

    Token refill: 1 token per (1/rate) seconds. Tokens accumulate up to
    max_tokens (= rate). Burst capacity = rate tokens.
    """

    PRIORITIES: dict[str, int] = {
        "EXIT":   0,    # highest — closing positions, never delayed
        "SL":     1,    # SL placement/modification — safety-critical
        "MODIFY": 2,    # fill management repricing
        "NEW":    3,    # new entry orders
        "POLL":   4,    # REST status checks — lowest priority
    }

    def __init__(self, rate: int = 10):
        """
        Args:
            rate: Maximum operations per second (Dhan limit: 10 per API key).
        """
        self.rate = rate
        self.max_tokens = rate
        self._tokens: float = rate
        self._last_refill: float = time.monotonic()
        self._queue: list[tuple[int, int, asyncio.Event]] = []  # min-heap
        self._counter: int = 0  # tiebreaker for same-priority items
        self._lock = asyncio.Lock()
        self._dispatch_task: asyncio.Task | None = None

    async def acquire(self, priority: str, timeout: float = 10.0) -> None:
        """
        Acquire a rate limit token at the given priority level.

        Blocks until a token is available. Higher-priority requests dequeue first.
        If no token is available within timeout seconds, raises TimeoutError.

        Args:
            priority: One of EXIT, SL, MODIFY, NEW, POLL
            timeout: Max wait time in seconds (default 10s)

        Raises:
            asyncio.TimeoutError: if token not acquired within timeout
            ValueError: if priority is unknown
        """
        if priority not in self.PRIORITIES:
            raise ValueError(f"Unknown priority '{priority}'. "
                           f"Valid: {list(self.PRIORITIES.keys())}")

        event = asyncio.Event()
        self._counter += 1

        async with self._lock:
            heapq.heappush(
                self._queue,
                (self.PRIORITIES[priority], self._counter, event)
            )

        # Try to dispatch immediately
        await self._try_dispatch()

        # Wait for our turn
        await asyncio.wait_for(event.wait(), timeout=timeout)

    async def _try_dispatch(self) -> None:
        """Try to dispatch queued requests if tokens are available."""
        async with self._lock:
            self._refill_tokens()

            while self._queue and self._tokens >= 1.0:
                _, _, event = heapq.heappop(self._queue)
                self._tokens -= 1.0
                event.set()

            # If there are still items in the queue, schedule a delayed dispatch
            if self._queue and self._dispatch_task is None:
                self._dispatch_task = asyncio.create_task(self._delayed_dispatch())

    async def _delayed_dispatch(self) -> None:
        """Wait for token refill, then dispatch."""
        await asyncio.sleep(1.0 / self.rate)
        self._dispatch_task = None
        await self._try_dispatch()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.max_tokens,
            self._tokens + elapsed * self.rate
        )
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        """Current available tokens (for monitoring)."""
        return self._tokens

    @property
    def queue_depth(self) -> int:
        """Number of queued requests (for monitoring)."""
        return len(self._queue)
```

**Concrete OPS budget scenario with 3 accounts:**

```
Scenario: EOD flatten with 5 positions per account at 15:20 IST.

Account "prop" (10 OPS):
  15:20:00  Cancel 5 pending entries:        5 OPS  (EXIT priority)
  15:20:00  Cancel 5 paired SLs:             5 OPS  (SL priority)
            → 10 OPS consumed in 1.0 second
  15:20:01  Place 5 exit orders:             5 OPS  (EXIT priority)
            → 0.5 seconds
  15:20:02  Fill management (5 parallel):    shared 10 OPS for repricing
            → ~2 OPS/s for modifications
  15:22:00  IOC escalation (if needed):      5 OPS  (EXIT priority)
            → 0.5 seconds

Account "client_001" (10 OPS — independent, runs in parallel):
  15:20:00  Same sequence, same timing, different API key
            → All 10 OPS consumed independently

Account "client_002" (10 OPS — independent, runs in parallel):
  15:20:00  Same sequence, same timing, different API key

Wall-clock time: ~2-3 seconds for all 3 accounts (parallel)
If single account: same 2-3 seconds
Multi-account adds ZERO latency to the flatten sequence.
```

---

### EOD Flatten

The EOD flatten sequence closes all intraday positions before market close. It runs independently and in parallel across all accounts.

```python
class EODFlatten:
    """
    End-of-day position flattening.

    Triggered at 15:20 IST for intraday strategies.
    Overnight positions (S2, S6 multi-day, S7) are NOT flattened — their SLs
    are converted to Forever Orders instead.

    The sequence is identical for each account but runs in parallel.
    """

    PHASE_1_TIME = "15:20:00"  # cancel pending
    PHASE_2_TIME = "15:20:01"  # place exits
    PHASE_3_TIME = "15:20:02"  # fill management
    PHASE_4_TIME = "15:22:00"  # IOC escalation
    PHASE_5_TIME = "15:25:00"  # hard deadline

    async def run(self, accounts: list["Account"]) -> dict[str, "FlattenResult"]:
        """Run EOD flatten for all accounts in parallel."""

        # Launch flatten for each account concurrently
        tasks = {
            account.account_id: asyncio.create_task(
                self._flatten_account(account),
                name=f"flatten_{account.account_id}",
            )
            for account in accounts
        }

        results = {}
        for account_id, task in tasks.items():
            try:
                results[account_id] = await task
            except Exception:
                logger.exception("flatten_account_error", account_id=account_id)
                results[account_id] = FlattenResult(
                    account_id=account_id, success=False, error="exception")

        return results

    async def _flatten_account(self, account: "Account") -> "FlattenResult":
        """
        5-phase flatten for one account.

        Phase 1: Cancel all pending entry orders + their SL pairs
        Phase 2: Place all exit orders in burst (AGGRESSIVE pricing)
        Phase 3: Parallel fill management for all exits
        Phase 4: IOC escalation for unfilled exits
        Phase 5: Hard deadline — alert, broker auto-squares at 15:30
        """
        rate_limiter = self._oms._account_rate_limiters[account.account_id]
        position_tracker = self._oms._position_tracker

        # ---- Phase 1: Cancel all pending entries ----
        logger.info("flatten_phase_1_cancel_pending", account_id=account.account_id)

        pending_orders = self._oms.get_pending_orders(account.account_id)
        cancel_tasks = []
        for order_state in pending_orders:
            if order_state.is_sl:
                continue  # SLs cancelled in Phase 2 as part of exit

            async def cancel_one(os=order_state):
                await rate_limiter.acquire("EXIT")
                try:
                    await self._oms._dhan_cancel_order(account, os.order_id)
                except OrderAlreadyFilled:
                    pass  # filled between check and cancel — fine
                # Cancel the paired SL too
                pairing = self._oms.sl_manager.get_pairing(
                    account.account_id, os.order_id)
                if pairing:
                    await self._oms.sl_manager.on_normal_exit(
                        account, os.order_id)

            cancel_tasks.append(asyncio.create_task(cancel_one()))

        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)

        # ---- Phase 2: Place all exit orders ----
        logger.info("flatten_phase_2_place_exits", account_id=account.account_id)

        intraday_positions = position_tracker.get_intraday_positions(
            account.account_id)
        # Filter out overnight positions (S2, S7, and S6 if multi-day)
        intraday_positions = [
            p for p in intraday_positions
            if p.strategy_id not in self._overnight_strategies(account)
        ]

        exit_tasks = []
        for pos in intraday_positions:
            exit_order = self._build_exit_order(pos, urgency="URGENT")

            # Cancel the entry's SL — we're managing the exit now
            await self._oms.sl_manager.on_normal_exit(
                account, pos.entry_order_id)

            # Check if SL triggered
            pairing = self._oms.sl_manager.get_pairing(
                account.account_id, pos.entry_order_id)
            if pairing and pairing.sl_status == "TRIGGERED":
                logger.info("flatten_skip_sl_triggered",
                           account_id=account.account_id,
                           instrument=pos.trading_symbol)
                continue  # position already closed by SL

            await rate_limiter.acquire("EXIT")
            resp = await self._oms._dhan_place_order(account, exit_order)

            if resp.orderStatus != "REJECTED":
                exit_state = OrderState(
                    order_id=resp.orderId,
                    account_id=account.account_id,
                    original_qty=pos.quantity,
                    current_limit_price=exit_order.price,
                )
                self._oms._demuxers[account.account_id].register_order(
                    resp.orderId, exit_state)
                exit_tasks.append((pos, exit_state, exit_order))

        # ---- Phase 3: Parallel fill management ----
        logger.info("flatten_phase_3_fill_management",
                   account_id=account.account_id,
                   exit_count=len(exit_tasks))

        fill_loops = []
        for pos, state, exit_order in exit_tasks:
            fill_params = FillParams(
                reprice_interval_s=3,     # fast repricing for EOD
                max_patience_s=110,       # until Phase 4 at 15:22
                pricing_mode="AGGRESSIVE",
            )
            loop = asyncio.create_task(
                self._oms.manage_order(state.order_id, exit_order, account))
            fill_loops.append((pos, state, loop))

        # Wait until Phase 4 time or all loops complete
        phase_4_deadline = self._time_until("15:22:00")
        done, pending = await asyncio.wait(
            [loop for _, _, loop in fill_loops],
            timeout=phase_4_deadline,
        )

        # ---- Phase 4: IOC escalation ----
        unfilled = [(pos, state) for pos, state, loop in fill_loops
                    if loop in pending]

        if unfilled:
            logger.warning("flatten_phase_4_ioc_escalation",
                         account_id=account.account_id,
                         unfilled_count=len(unfilled))

            for pos, state in unfilled:
                # Cancel the pending exit
                await rate_limiter.acquire("EXIT")
                try:
                    await self._oms._dhan_cancel_order(account, state.order_id)
                except Exception:
                    pass

                # Re-place as IOC at aggressive price (+2 ticks)
                market = self._oms._get_current_market_data(
                    pos.instrument_id, account)
                tick_size = pos.tick_size

                if pos.direction == "LONG":
                    # Sell to close: bid - 2 ticks
                    ioc_price = market.best_bid - 2 * tick_size
                else:
                    # Buy to close: ask + 2 ticks
                    ioc_price = market.best_ask + 2 * tick_size

                # +2 ticks, not +5: SEBI limit-order rule means our limit must
                # still be a reasonable limit, not a de facto market order
                ioc_price = round(ioc_price / tick_size) * tick_size

                remaining_qty = state.remaining_qty if state.filled_qty > 0 else pos.quantity
                await rate_limiter.acquire("EXIT")
                await self._oms._dhan_place_order(account, DhanOrderRequest(
                    dhanClientId=account.dhan_client_id,
                    transactionType="SELL" if pos.direction == "LONG" else "BUY",
                    exchangeSegment=pos.exchange_segment,
                    productType="INTRADAY",
                    orderType="LIMIT",
                    validity="IOC",          # Immediate or Cancel
                    securityId=pos.instrument_id,
                    quantity=remaining_qty,
                    price=ioc_price,
                    triggerPrice=None,
                    disclosedQuantity=0,
                    afterMarketOrder=False,
                    amoTime=None,
                    boProfitValue=None,
                    boStopLossValue=None,
                    correlationId=f"IOC_FLAT_{account.account_id[:8]}",
                ))

        # ---- Phase 5: Hard deadline (15:25) ----
        phase_5_deadline = self._time_until("15:25:00")
        await asyncio.sleep(max(0, phase_5_deadline))

        # Check for any remaining positions
        remaining_positions = position_tracker.get_intraday_positions(
            account.account_id)
        remaining_intraday = [
            p for p in remaining_positions
            if p.strategy_id not in self._overnight_strategies(account)
        ]

        if remaining_intraday:
            logger.critical("flatten_incomplete_at_hard_deadline",
                          account_id=account.account_id,
                          remaining=[p.trading_symbol for p in remaining_intraday])
            await telegram.send(CRITICAL,
                f"EOD FLATTEN INCOMPLETE: Account {account.account_id} has "
                f"{len(remaining_intraday)} positions at 15:25. "
                f"Dhan auto-squares at ~15:30 at market price.")

        # ---- Convert overnight SLs to Forever Orders ----
        overnight_positions = position_tracker.get_overnight_positions(
            account.account_id)
        for pos in overnight_positions:
            await self._oms.sl_manager.on_eod_for_overnight(
                account, pos.entry_order_id)

        return FlattenResult(
            account_id=account.account_id,
            success=len(remaining_intraday) == 0,
            positions_closed=len(intraday_positions) - len(remaining_intraday),
            positions_remaining=len(remaining_intraday),
            overnight_sl_converted=len(overnight_positions),
        )


class FlattenResult(pydantic.BaseModel):
    account_id: str
    success: bool
    positions_closed: int = 0
    positions_remaining: int = 0
    overnight_sl_converted: int = 0
    error: str | None = None
```

**OPS budget for N-account flatten:**

```
Per-account (5 positions, 10 OPS):
  Phase 1: 5 cancel entry + 5 cancel SL = 10 OPS → 1.0 second
  Phase 2: 5 exit orders = 5 OPS → 0.5 second
  Phase 3: fill management, ~2-3 OPS/s for repricing → 2 minutes
  Phase 4: 5 cancel + 5 IOC = 10 OPS → 1.0 second (if needed)
  Total per account: ~3-5 seconds of OPS-intensive work

With 3 accounts (parallel):
  Wall-clock = max(per-account) = same 3-5 seconds
  Total API calls = 3x but on 3x the OPS budget = no congestion

With 5 accounts (parallel):
  Wall-clock = same 3-5 seconds
  Total API calls = 5x but on 5x the OPS budget = no congestion
```

**Dhan auto-square backstop:** Dhan auto-squares all INTRADAY positions at approximately 15:30 IST at market price. This is the last line of defense if our EOD flatten fails completely. The auto-square is undesirable (market price = potentially worse than our limit) but prevents carrying unintended overnight risk.

---

### OMS Degraded Mode (Redis Down)

When Redis is unavailable, the OMS enters degraded mode. It continues managing existing orders and positions using broker WS + REST directly. No new signals arrive (strategy processes can't publish without Redis).

#### What Still Works

| Function | Status | Mechanism |
|----------|--------|-----------|
| Dhan order update WS | WORKS | Direct WebSocket connection, no Redis dependency |
| Dhan REST API (place, modify, cancel, status) | WORKS | Direct HTTPS, no Redis dependency |
| Fill management loops (running orders) | WORKS | OrderState is in-memory, WS updates route to it |
| SL lifecycle management | WORKS | SLPairing objects are in-memory |
| Per-order locking | WORKS | asyncio.Lock is in-memory |
| Rate limiters | WORKS | Token bucket state is in-memory |
| EOD flatten | WORKS | Uses broker API directly |
| Global kill switch | PARTIAL | OMS checks an in-memory flag; the Redis `HALT:global` check fails but the CLI can set the in-memory flag directly |
| Server-side SL orders | WORKS | SL orders are on broker infra, independent of our system |

#### What Breaks

| Function | Status | Impact |
|----------|--------|--------|
| Strategy signal delivery | BROKEN | Strategies can't publish to `STREAM:SIGNAL` |
| New entry orders | BLOCKED | No signals arriving → no new orders |
| LASTTICK for repricing | BROKEN | Can't read `LASTTICK:{symbol}` from Redis |
| Position state updates | BROKEN | Can't write `POSITION:strategy:{sid}` |
| Heartbeats | BROKEN | Strategies appear dead (heartbeat keys expire) |
| State snapshots | BROKEN | Can't save strategy state to Redis |
| Config hot-reload | BROKEN | Can't read `CONFIG:strategy:{sid}` |

#### Fallback for Market Prices

When Redis `LASTTICK` is unavailable, the OMS falls back to Dhan's option chain API for market data:

```python
def _fetch_market_from_dhan(
    self,
    instrument_id: str,
    account: "Account",
) -> MarketSnapshot:
    """
    Fallback market data source when Redis is down.

    Uses Dhan's quote API to get current bid/ask for an instrument.
    Costs 1 OPS per call. Used ONLY for repricing active fill management loops.

    Endpoint: GET https://api.dhan.co/v2/marketfeed/ltp
    or: POST https://api.dhan.co/v2/marketfeed/quote

    This is expensive (1 OPS per reprice check) but acceptable in degraded mode
    because new entries are blocked (no signals) so OPS headroom is available.
    """
    rate_limiter = self._account_rate_limiters[account.account_id]
    # Don't acquire rate limit here — caller already acquired
    resp = self._dhan_client.get_quote(
        account.api_key,
        security_id=instrument_id,
    )
    return MarketSnapshot(
        best_bid=resp.get("bestBidPrice", 0),
        best_ask=resp.get("bestOfferPrice", 0),
        ltp=resp.get("lastTradedPrice", 0),
        ts=now_ms(),
    )
```

#### Recovery When Redis Reconnects

```python
async def on_redis_reconnect(self) -> None:
    """
    Called when Redis connection is re-established.

    Sequence:
    1. Publish accumulated state changes (order results, position updates)
    2. Re-enable signal reception (unblock STREAM:SIGNAL consumer)
    3. Strategy processes will auto-reconnect their consumer groups
    4. Normal operation resumes
    """
    logger.info("redis_reconnected_oms_recovery")

    # 1. Sync all current order states to Redis
    for (account_id, order_id), state in self._order_states.items():
        await redis.hset(f"ORDER:{account_id}:{order_id}", mapping={
            "status": state.status,
            "filled_qty": str(state.filled_qty),
            "remaining_qty": str(state.remaining_qty),
            "avg_fill_price": str(state.avg_fill_price),
        })

    # 2. Sync all SL pairings
    for key, pairing in self.sl_manager._pairings.items():
        account_id, entry_id = key
        await redis.hset(f"SL:{account_id}:{entry_id}", mapping={
            "sl_order_id": pairing.sl_order_id,
            "sl_qty": str(pairing.sl_qty),
            "sl_status": pairing.sl_status,
            "sl_type": pairing.sl_type,
        })

    # 3. Clear HALT flag if it was set due to Redis outage
    # (but NOT if global kill was manually triggered)
    if not self._manual_kill_active:
        await redis.delete("HALT:redis_outage")

    # 4. Resume normal mode
    self._degraded_mode = False
    logger.info("oms_degraded_mode_exited")
```

---

### Global Kill Switch

```python
async def global_kill(self) -> None:
    """
    One command: cancel everything, flatten everything, NOW.
    Runs across ALL accounts in parallel.

    Triggered by:
    - CLI: python -m live.cli kill
    - Redis: SET HALT:global 1
    - Telegram: /kill command (authenticated)
    - Risk Manager: portfolio drawdown > 10%
    """
    logger.critical("GLOBAL_KILL_ACTIVATED")
    self._manual_kill_active = True

    # 1. Block all new signals immediately
    try:
        await redis.set("HALT:global", "1")
    except Exception:
        pass  # Redis may be down — in-memory flag is the backup

    # 2. Cancel ALL pending orders across ALL accounts
    cancel_tasks = []
    for account_id, account in self._accounts.items():
        rate_limiter = self._account_rate_limiters[account_id]
        for key, state in self._order_states.items():
            if key[0] != account_id:
                continue
            if state.status in ("PENDING", "PART_TRADED", "TRANSIT"):
                async def cancel_one(a=account, s=state, rl=rate_limiter):
                    try:
                        await rl.acquire("EXIT")
                        await self._dhan_cancel_order(a, s.order_id)
                    except Exception:
                        pass  # best effort
                cancel_tasks.append(asyncio.create_task(cancel_one()))

    if cancel_tasks:
        await asyncio.gather(*cancel_tasks, return_exceptions=True)

    # 3. Flatten ALL positions across ALL accounts with AGGRESSIVE pricing
    flatten_tasks = []
    for account_id, account in self._accounts.items():
        positions = self._position_tracker.get_all_open(account_id)
        rate_limiter = self._account_rate_limiters[account_id]

        for pos in positions:
            async def flatten_one(a=account, p=pos, rl=rate_limiter):
                exit_order = self._build_exit_order(p, urgency="URGENT")
                await rl.acquire("EXIT")
                await self._dhan_place_order(a, exit_order)
            flatten_tasks.append(asyncio.create_task(flatten_one()))

    if flatten_tasks:
        await asyncio.gather(*flatten_tasks, return_exceptions=True)

    # 4. Alert
    await telegram.send(CRITICAL,
        f"GLOBAL KILL: all orders cancelled across {len(self._accounts)} accounts, "
        f"flattening all positions")

    logger.critical("GLOBAL_KILL_COMPLETE")
```

**Resuming after global kill:** requires manual intervention:
1. `DEL HALT:global` in Redis
2. Full system restart
3. Operator verifies all positions are flat across all accounts
4. Operator reviews kill reason in audit log

---

### State Table

| State Item | Storage | Scope | Lifecycle | Writer | Readers |
|-----------|---------|-------|-----------|--------|---------|
| OrderState objects | In-memory | Per-account, per-order | Created on placement, removed on terminal state | OMS (demuxer applies updates) | Fill management loop |
| SLPairing objects | In-memory | Per-account, per-entry | Entry → SL mapping, lifetime of position | SL Lifecycle Manager | OMS, EOD Flatten |
| Rate limiter tokens | In-memory | Per-account | Session | PriorityRateLimiter | All OMS operations |
| Rate limiter queue | In-memory | Per-account | Transient (drain on each dispatch) | PriorityRateLimiter | All OMS operations |
| WS connection | In-memory (socket) | Per-account | Session (reconnects on drop) | DhanOrderWS | OrderUpdateDemuxer |
| WS health state | In-memory | Per-account | Continuous | WSHealthMonitor | Fill management loops |
| Demuxer routing table | In-memory | Per-account | Order lifetime | OMS (register/unregister) | OrderUpdateDemuxer |
| Placed signal_ids (dedup) | In-memory set | Global (across accounts) | Session | OMS | OMS (before placement) |
| Daily order count | Redis `OMS:daily_count:{account_id}` | Per-account | Reset 08:30 | OMS | Risk Manager, Monitoring |
| Order state mirror | Redis `ORDER:{account_id}:{order_id}` | Per-account, per-order | Updated on every state change | OMS | Position Tracker, Monitoring |
| SL pairing mirror | Redis `SL:{account_id}:{entry_id}` | Per-account | Updated on SL changes | SL Lifecycle Manager | Monitoring |
| Degraded mode flag | In-memory | Global | Set on Redis loss, cleared on reconnect | OMS | OMS (all operations) |
| Manual kill flag | In-memory + Redis `HALT:global` | Global | Set by kill, cleared by operator | OMS, CLI, Telegram | All components |
| Account auth tokens | In-memory | Per-account | Session (refreshed daily) | Auth Manager | DhanOrderWS, REST calls |
| Divergence records | DuckDB | Per-signal | Persistent | DivergenceTracker | Compliance reporting |

---

### Failure Modes

| # | Failure | Detection | Impact | Recovery |
|---|---------|-----------|--------|----------|
| 1 | **Dhan order WS disconnects (one account)** | `ConnectionClosed` exception in demuxer | That account's fill management loops lose real-time updates. Other accounts unaffected. | Exponential backoff reconnect: 1s, 2s, 4s, 8s, max 30s. Fill management loops switch to REST polling (costs OPS). |
| 2 | **Dhan order WS disconnects (all accounts)** | All WSHealthMonitor.is_healthy = False | All fill management loops switch to REST polling. OPS consumption increases significantly. | Same reconnect logic. If all WS down for >60s: Telegram CRITICAL, suppress new entries. |
| 3 | **Dhan REST API returns 5xx** | HTTP status code | That specific API call fails. | Retry with backoff: 500ms, 1s, 2s. Max 3 retries. If persistent: mark account as degraded. |
| 4 | **Dhan REST API returns 429 (rate limited)** | HTTP 429 | Server-side rate limit hit (we exceeded 10 OPS). | Back off for 1 second. Should not happen if our client-side rate limiter is working correctly. If it does: log ERROR, investigate. |
| 5 | **Account API key expires mid-session** | HTTP 401 from any Dhan call | All operations for that account fail. Other accounts continue. | Mark account as AUTH_FAILED. Stop all new orders for that account. Existing positions protected by server-side SLs. Telegram CRITICAL: "Account {id} auth expired — manual re-auth required." |
| 6 | **Entry order rejected by exchange** | `orderStatus: "REJECTED"` | No position taken. SL is cancelled (if already placed) or never placed. | Log rejection reason. Signal is consumed (not retried — market may have moved). |
| 7 | **SL order rejected by exchange** | `orderStatus: "REJECTED"` for SL | Entry order may be pending or filled WITHOUT SL protection. | CRITICAL: Cancel entry immediately. If entry already filled: place emergency market exit. Telegram CRITICAL. |
| 8 | **SL triggers while modifying entry** | SL WS update shows TRADED while entry modify is in-flight | Position is closed by SL. Any pending modify is irrelevant. | Detect in fill management loop: if SL has triggered, stop managing the entry. Position already flat. |
| 9 | **Partial fill → SL qty mismatch** | SL qty != entry filled_qty after PART_TRADED | If SL triggers with wrong qty: over-exit (short shares we don't own) or under-exit (residual position). | CRITICAL-2 fix: `on_entry_partial_fill` called on EVERY PART_TRADED update, not just on abandon. |
| 10 | **DAY SL expires at EOD for overnight position** | 15:30 IST session close | Overnight position has zero SL protection until next morning. | CRITICAL-1 fix: `on_eod_for_overnight` converts to Forever Order before DAY SL expires. Falls back to AMO if Forever fails. |
| 11 | **Order modify fails silently** | Post-modify REST verification shows price didn't change | Order is at stale price. May not fill or fill at wrong level. | Every modify is followed by a REST verification poll. If price mismatch: re-send modify. |
| 12 | **Cancel-replace creates duplicate position** | After cancel, old order fills before we detect the fill. New order also placed. | Double position: 2x intended exposure. | Check if old order filled before placing new order. If old order is TRADED: skip new placement. Use the order's correlation_id to detect cancel-replace pairs in reconciliation. |
| 13 | **Redis down during fill management** | `aioredis.ConnectionError` | Can't update LASTTICK for repricing. Can't publish results. | OMS enters degraded mode: use Dhan API for market data, hold state in-memory, sync to Redis on reconnect. |
| 14 | **OMS process crashes** | systemd detects exit | All order tracking state lost. Fill management loops stop. | Restart: query Dhan for all active orders + positions. Rebuild OrderState from Dhan API. SL orders are on broker — still active. |
| 15 | **Dhan daily order limit (5,000) reached** | HTTP rejection with order-limit error | No more orders for that account for the day. | Telegram CRITICAL. Account enters read-only mode. Existing positions manage via SL only. Should not happen in normal operation (7 strategies × 2 trades/day × 2 orders/trade = 28 orders/day). |
| 16 | **Account A fills, Account B rejects (divergence)** | MultiAccountFanOut compares outcomes | Accounts hold different positions for the same strategy signal. | Log divergence. Do NOT force-sync (that would require additional trades). Track for compliance reporting. If Account B fails >3 consecutive signals: disable it, Telegram CRITICAL. |
| 17 | **Freeze qty exceeded (large order)** | `quantity > freeze_qty` check before placement | Exchange rejects orders above freeze limit. | Auto-switch to slicing endpoint (POST /v2/orders/slicing). Track child orders independently. |
| 18 | **Network partition between OMS and Dhan** | Connection timeouts on all REST + WS | Complete broker blackout. Cannot place, modify, or cancel. | Server-side SLs are the sole protection. Telegram CRITICAL (if Telegram is reachable). Wait for network restoration. If >5 min: operator intervention required. |
| 19 | **Dhan WS sends duplicate update** | Sequence number in OrderState is already >= update's implied sequence | State could be applied twice, corrupting filled_qty. | Sequence number check: if WS update would decrease sequence, discard it. |
| 20 | **Global kill during active fill management** | `HALT:global` flag checked on every loop iteration | Fill loops must abort immediately. | Set in-memory `_halt` flag. All fill loops check this flag and exit cleanly on next iteration. Cancel remaining orders. |

---

### Edge Cases

#### 1. Entry fills in 2ms, before SL API call starts

The entry LIMIT order is placed and fills within 2ms (happens with aggressive pricing in liquid instruments). By the time we place the SL:
- Position is already filled at known price
- SL placement uses the filled price (not the limit price) for trigger calculation
- SL is placed with exact filled_qty
- No issue — the position is protected once SL is confirmed

#### 2. Modify returns success but price doesn't change

Dhan may accept a modify request but the exchange hasn't processed it yet (TRANSIT state). Our post-modify REST poll may show the OLD price.

**Handling:** If post-modify poll shows the old price, wait 500ms and poll again. If still old price after 3 polls: assume modify failed, consider cancel-replace.

#### 3. SL triggers during EOD flatten's cancel-SL step

```
1. EOD flatten calls sl_manager.on_normal_exit() to cancel SL
2. While our cancel request is in-flight, the market moves and SL triggers
3. Dhan fills the SL (position closed)
4. Our cancel request returns "order not cancellable"
5. We place an exit order (but position is already flat!)
6. Exit order creates a NEW position (wrong direction)
```

**Handling:** After cancel-SL fails with "not cancellable", verify SL status. If TRIGGERED: skip exit order. The `place_exit_order` method checks this (see Race Condition 2 above).

#### 4. Sliced order: some children fill, others don't

A 1950-qty order (above 1800 freeze limit) is sliced into two children: 1800 + 150. The 1800-qty child fills, the 150-qty child is rejected (insufficient lot).

**Handling:** Track each child independently. SL is placed for total intended qty. On partial (child rejection): adjust SL qty to match total filled across all children. Treat as PARTIAL_FILL in the fill management result.

#### 5. Account added mid-session (hot-add)

v1 does not support hot-adding accounts mid-session. All accounts must be configured at startup. The system restart required to add an account is the same restart required to add a strategy (see Strategy Manager hot-add decision).

#### 6. Two signals arrive for the same strategy within 100ms

Signal deduplication (in Signal Router) handles this upstream. By the time signals reach the OMS, they are already deduplicated. Additionally, the OMS tracks `placed_signal_ids` and rejects any `signal_id` it has already processed.

#### 7. Dhan changes modify qty semantics

If Dhan changes the modify API to interpret `quantity` as remaining (not original total), our orders would be cancelled unexpectedly.

**Mitigation:** Paper trading validation: place a 650-qty order, get 325 filled, modify with qty=650, verify remaining is still 325. If behavior changes: flip a config flag `MODIFY_QTY_IS_REMAINING = True` and adjust the reprice logic to send `remaining_qty` instead of `original_qty`.

```python
# In reprice_order:
if config.MODIFY_QTY_IS_REMAINING:
    modify_qty = state.remaining_qty
else:
    modify_qty = state.original_qty  # default: Dhan current behavior
```

#### 8. Forever Order triggers overnight, WS not connected

Our system is shut down overnight (08:00 startup). A Forever Order SL triggers at 03:00 AM (unlikely for equity derivatives, but possible for currency futures or if the trigger is very close to the closing price and a gap opens).

**Handling:** The Forever Order triggers on Dhan's infrastructure without our WS connected. The fill happens on the exchange. At our 08:40 startup (Phase 4: Position Recovery), we query broker positions and discover the position is flat. We reconcile: mark the SL as TRIGGERED, mark the position as closed, log the event. No action needed — the SL did its job.

#### 9. Multi-account: different lot sizes due to rounding

Account A (₹50L, 0.25 Kelly) allocates 3.7 lots → rounds to 3 lots.
Account B (₹1Cr, 0.25 Kelly) allocates 7.4 lots → rounds to 7 lots.
Account C (₹2Cr, 0.25 Kelly) allocates 14.8 lots → rounds to 15 lots.

Accounts are NOT proportional (3:7:15 vs the theoretical 1:2:4 capital ratio). This is inherent to discrete lot sizes.

**Rounding rule:** Always round DOWN (floor). Never round up — that would exceed the allocation.

```python
def compute_lots(capital: float, kelly: float, weight: float,
                premium: float, lot_size: int) -> int:
    raw_capital = capital * kelly * weight
    raw_qty = raw_capital / premium
    raw_lots = raw_qty / lot_size
    return max(1, int(raw_lots))  # floor, minimum 1 lot
```

#### 10. OMS places order, Dhan returns success, but order never appears in order book

Rare: Dhan REST returns 200 with orderId, but the order is silently dropped before reaching the exchange.

**Detection:** Order stays in TRANSIT for >10 seconds. Fill management loop's TRANSIT timeout triggers a REST poll. If REST also shows TRANSIT after 10s: cancel and re-place. If REST shows the order doesn't exist: treat as rejected.
