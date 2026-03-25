## Component 6: Multi-Broker Router

### Responsibility

- Maintain per-account broker client instances with independent auth, connections, and failover state
- Provide a unified `BrokerAdapter` interface that insulates the OMS from broker-specific API details
- Map canonical instrument identifiers (Dhan `security_id`) to equivalent identifiers on backup brokers (Upstox `instrument_key`)
- Authenticate all accounts in parallel at system startup (08:25 IST) and manage token lifecycle in Redis
- Detect per-account broker health degradation and execute independent failover: hedge open positions on backup broker while monitoring primary for recovery
- Enforce per-account rate limits via delegation to the OMS's `PriorityRateLimiter` (10 OPS per Dhan account, 50 OPS per Upstox account)
- **Does NOT** generate signals, manage positions, or track fills (those are upstream/downstream components)

---

### Per-Account Broker Clients

Each trading account gets its own broker client instance. Accounts do not share connections, tokens, or rate-limit budgets. If Account A's broker connection fails, Account B continues operating on its own independent connection.

#### Account Model

```python
class Account(pydantic.BaseModel):
    """
    Canonical Account model (defined in Component 11: Account Replication Layer).
    Reproduced here for reference — the single source of truth is in section 10.

    The Broker Router uses these fields:
    - account_id, dhan_client_id, dhan_access_token: for DhanClient creation
    - capital: Decimal (exact arithmetic at ₹10Cr scale)
    - primary_broker: which broker to use by default
    """
    account_id: str
    dhan_client_id: str
    dhan_access_token: str
    capital: Decimal
    enabled_strategies: list[str]
    strategy_weights: dict[str, float]
    kelly_fraction: float
    max_drawdown_pct: float
    status: Literal["ACTIVE", "SUSPENDED", "AUTH_FAILED", "MARGIN_CALL"] = "ACTIVE"
    # Broker Router extensions (not in canonical model):
    upstox_client_id: str | None = None
    upstox_api_key: str | None = None
    primary_broker: Literal["dhan", "upstox"] = "dhan"
```

#### DhanClient

One `DhanClient` per account. Handles authentication headers, base URL, and retry logic for all REST calls.

```python
import aiohttp
import asyncio
import time
import structlog

logger = structlog.get_logger()


class DhanClient:
    """
    HTTP client for the Dhan REST API. One instance per account.

    Handles:
    - Auth headers (per-account access token)
    - Base URL routing
    - Retry with exponential backoff on transient errors (5xx, timeout)
    - Request/response logging for audit trail
    - Connection pooling via aiohttp.ClientSession
    """

    BASE_URL = "https://api.dhan.co"
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 0.5   # seconds: 0.5, 1.0, 2.0
    REQUEST_TIMEOUT = 10.0     # seconds per request

    def __init__(self, account: Account):
        self.account = account
        self._client_id = account.dhan_client_id
        self._token: str | None = None
        self._session: aiohttp.ClientSession | None = None
        self._request_count: int = 0
        self._error_count: int = 0
        self._last_request_ts: float = 0.0

    @property
    def auth_headers(self) -> dict[str, str]:
        if self._token is None:
            raise RuntimeError(
                f"DhanClient for {self.account.account_id} not authenticated"
            )
        return {
            "access-token": self._token,
            "Content-Type": "application/json",
        }

    def set_token(self, token: str) -> None:
        """Set the access token after authentication."""
        self._token = token

    async def start(self) -> None:
        """Create the aiohttp session. Call once at startup."""
        timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
        self._session = aiohttp.ClientSession(
            base_url=self.BASE_URL,
            headers=self.auth_headers,
            timeout=timeout,
        )

    async def close(self) -> None:
        """Close the aiohttp session. Call at shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
    ) -> dict:
        """
        Make an authenticated request to Dhan API with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API path (e.g., "/v2/orders")
            body: JSON body for POST/PUT requests

        Returns:
            Parsed JSON response as dict.

        Raises:
            DhanAPIError: on non-retryable failure (4xx except 429)
            DhanRateLimitError: on HTTP 429
            DhanAuthError: on HTTP 401
            DhanTimeoutError: after MAX_RETRIES exhausted
        """
        if self._session is None:
            raise RuntimeError("DhanClient session not started")

        last_exception: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                self._request_count += 1
                self._last_request_ts = time.monotonic()

                async with self._session.request(
                    method,
                    path,
                    json=body,
                    headers=self.auth_headers,
                ) as resp:
                    response_body = await resp.json()

                    if resp.status == 200:
                        return response_body

                    if resp.status == 401:
                        self._error_count += 1
                        raise DhanAuthError(
                            account_id=self.account.account_id,
                            message=response_body.get("remarks", "auth_failed"),
                        )

                    if resp.status == 429:
                        self._error_count += 1
                        raise DhanRateLimitError(
                            account_id=self.account.account_id,
                        )

                    if 400 <= resp.status < 500:
                        self._error_count += 1
                        raise DhanAPIError(
                            account_id=self.account.account_id,
                            status=resp.status,
                            error_code=response_body.get("errorCode"),
                            message=response_body.get("remarks", "unknown"),
                        )

                    # 5xx — retryable
                    last_exception = DhanAPIError(
                        account_id=self.account.account_id,
                        status=resp.status,
                        error_code=response_body.get("errorCode"),
                        message=response_body.get("remarks", "server_error"),
                    )
                    logger.warning(
                        "dhan_api_5xx_retry",
                        account_id=self.account.account_id,
                        status=resp.status,
                        attempt=attempt + 1,
                        path=path,
                    )

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = DhanTimeoutError(
                    account_id=self.account.account_id,
                    message=str(e),
                )
                logger.warning(
                    "dhan_api_timeout_retry",
                    account_id=self.account.account_id,
                    attempt=attempt + 1,
                    path=path,
                    error=str(e),
                )

            # Exponential backoff before retry
            if attempt < self.MAX_RETRIES - 1:
                backoff = self.RETRY_BACKOFF_BASE * (2 ** attempt)
                await asyncio.sleep(backoff)

        # All retries exhausted
        self._error_count += 1
        raise last_exception or DhanTimeoutError(
            account_id=self.account.account_id,
            message=f"All {self.MAX_RETRIES} retries exhausted for {method} {path}",
        )

    # --- Convenience methods ---

    async def place_order(self, body: dict) -> dict:
        return await self.request("POST", "/v2/orders", body)

    async def modify_order(self, order_id: str, body: dict) -> dict:
        return await self.request("PUT", f"/v2/orders/{order_id}", body)

    async def cancel_order(self, order_id: str) -> dict:
        return await self.request("DELETE", f"/v2/orders/{order_id}")

    async def get_order(self, order_id: str) -> dict:
        return await self.request("GET", f"/v2/orders/{order_id}")

    async def get_trades(self, order_id: str) -> dict:
        return await self.request("GET", f"/v2/trades/{order_id}")

    async def place_forever_order(self, body: dict) -> dict:
        return await self.request("POST", "/v2/forever/orders", body)

    async def modify_forever_order(self, order_id: str, body: dict) -> dict:
        return await self.request("PUT", f"/v2/forever/orders/{order_id}", body)

    async def cancel_forever_order(self, order_id: str) -> dict:
        return await self.request("DELETE", f"/v2/forever/orders/{order_id}")

    async def get_positions(self) -> dict:
        return await self.request("GET", "/v2/positions")

    async def get_holdings(self) -> dict:
        return await self.request("GET", "/v2/holdings")

    async def place_slicing_order(self, body: dict) -> dict:
        return await self.request("POST", "/v2/orders/slicing", body)

    @property
    def stats(self) -> dict:
        return {
            "account_id": self.account.account_id,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._request_count
                if self._request_count > 0
                else 0.0
            ),
            "last_request_ts": self._last_request_ts,
        }


class DhanAPIError(Exception):
    def __init__(self, account_id: str, status: int,
                 error_code: str | None, message: str):
        self.account_id = account_id
        self.status = status
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{account_id}] Dhan {status}: {error_code} — {message}")


class DhanAuthError(DhanAPIError):
    def __init__(self, account_id: str, message: str):
        super().__init__(account_id, 401, "AUTH_FAILED", message)


class DhanRateLimitError(DhanAPIError):
    def __init__(self, account_id: str):
        super().__init__(account_id, 429, "RATE_LIMITED", "Too many requests")


class DhanTimeoutError(Exception):
    def __init__(self, account_id: str, message: str):
        self.account_id = account_id
        self.message = message
        super().__init__(f"[{account_id}] Dhan timeout: {message}")
```

#### Per-Account Failover State

Each account tracks its own broker health independently. Account A might be operating on Dhan while Account B has failed over to Upstox.

```python
from enum import Enum


class BrokerHealth(str, Enum):
    HEALTHY = "HEALTHY"             # primary broker operating normally
    DEGRADED = "DEGRADED"           # intermittent failures, still usable
    FAILED = "FAILED"               # primary broker unreachable
    FAILOVER_ACTIVE = "FAILOVER"    # using backup broker
    SUSPENDED = "SUSPENDED"         # account disabled (auth failure, operator action)


class AccountBrokerState(pydantic.BaseModel):
    """
    Per-account broker health and failover state. One instance per account.
    Stored in-memory, mirrored to Redis for monitoring.
    """
    account_id: str
    primary_broker: Literal["dhan", "upstox"] = "dhan"
    active_broker: Literal["dhan", "upstox"] = "dhan"
    health: BrokerHealth = BrokerHealth.HEALTHY

    # Failure tracking for failover trigger
    consecutive_failures: int = 0
    last_failure_ts: float = 0.0
    last_success_ts: float = 0.0

    # WS state
    ws_connected: bool = False
    ws_disconnect_ts: float | None = None

    # Failover tracking
    failover_ts: float | None = None
    failover_reason: str | None = None
    failback_eligible_ts: float | None = None   # earliest allowed failback

    # Hedging state (populated during failover)
    hedge_positions: list["HedgePosition"] = []

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.last_success_ts = time.monotonic()
        if self.health == BrokerHealth.DEGRADED:
            self.health = BrokerHealth.HEALTHY

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_ts = time.monotonic()
        if self.consecutive_failures >= 3:
            self.health = BrokerHealth.FAILED

    def record_ws_disconnect(self) -> None:
        self.ws_connected = False
        self.ws_disconnect_ts = time.monotonic()

    def record_ws_reconnect(self) -> None:
        self.ws_connected = True
        self.ws_disconnect_ts = None
```

#### Per-Account WS Connections

Each account maintains its own WebSocket connection for order updates. This is established in the OMS (Component 5) via `DhanOrderWS`. The broker router owns the lifecycle:

```python
class BrokerRouter:
    """
    Central broker routing layer. Manages per-account broker clients,
    failover, symbol mapping, and auth lifecycle.
    """

    def __init__(self, accounts: list[Account]):
        self._accounts = {a.account_id: a for a in accounts}

        # Per-account Dhan clients
        self._dhan_clients: dict[str, DhanClient] = {
            a.account_id: DhanClient(a) for a in accounts
        }

        # Per-account Upstox clients (created on demand during failover)
        self._upstox_clients: dict[str, "UpstoxClient"] = {}

        # Per-account broker state
        self._broker_state: dict[str, AccountBrokerState] = {
            a.account_id: AccountBrokerState(
                account_id=a.account_id,
                primary_broker=a.primary_broker,
                active_broker=a.primary_broker,
            )
            for a in accounts
        }

        # Per-account WS connections (for order updates)
        self._order_ws: dict[str, DhanOrderWS] = {
            a.account_id: DhanOrderWS(a) for a in accounts
        }

        # Symbol mapper (shared across all accounts)
        self._symbol_mapper: SymbolMapper | None = None

        # Auth manager
        self._auth_manager = AuthManager(self)

    def get_client(self, account_id: str) -> "BrokerAdapter":
        """
        Return the active broker adapter for this account.
        If failover is active, returns the backup broker's adapter.
        """
        state = self._broker_state[account_id]

        if state.active_broker == "dhan":
            return DhanAdapter(self._dhan_clients[account_id])
        elif state.active_broker == "upstox":
            if account_id not in self._upstox_clients:
                raise BrokerUnavailableError(
                    f"Upstox client not initialized for {account_id}"
                )
            return UpstoxAdapter(self._upstox_clients[account_id])

        raise BrokerUnavailableError(
            f"No active broker for {account_id}, state={state.health}"
        )

    def get_state(self, account_id: str) -> AccountBrokerState:
        return self._broker_state[account_id]
```

---

### BrokerAdapter Interface

The OMS interacts with brokers exclusively through the `BrokerAdapter` abstract base class. Broker-specific details (endpoint URLs, request schemas, field names) are encapsulated in concrete implementations.

```python
from abc import ABC, abstractmethod


class OrderResponse(pydantic.BaseModel):
    """Normalized order response from any broker."""
    order_id: str
    status: Literal["TRANSIT", "PENDING", "REJECTED"]
    broker: Literal["dhan", "upstox"]
    raw_response: dict                  # original broker response for audit


class OrderDetail(pydantic.BaseModel):
    """Normalized order status from any broker."""
    order_id: str
    status: Literal[
        "TRANSIT", "PENDING", "TRADED", "PART_TRADED",
        "CANCELLED", "REJECTED", "EXPIRED",
    ]
    filled_qty: int
    remaining_qty: int
    avg_fill_price: float
    exchange_order_id: str | None
    exchange_time: str | None
    broker: Literal["dhan", "upstox"]


class PositionDetail(pydantic.BaseModel):
    """Normalized position from any broker."""
    instrument_id: str                  # canonical (Dhan security_id)
    exchange_segment: str
    product_type: str
    quantity: int                       # signed: positive = long, negative = short
    avg_price: float
    pnl: float
    broker: Literal["dhan", "upstox"]


class BrokerAdapter(ABC):
    """
    Abstract interface for broker integration. The OMS calls these methods
    without knowing which broker is active for a given account.

    Every method accepts an Account and the canonical instrument identifiers.
    The adapter translates to broker-specific field names internally.
    """

    @abstractmethod
    async def place_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        validity: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        """Place an order. Returns normalized response."""
        ...

    @abstractmethod
    async def modify_order(
        self,
        account: Account,
        order_id: str,
        order_type: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        validity: str = "DAY",
    ) -> OrderResponse:
        """Modify a pending order."""
        ...

    @abstractmethod
    async def cancel_order(
        self, account: Account, order_id: str,
    ) -> OrderResponse:
        """Cancel a pending order."""
        ...

    @abstractmethod
    async def get_order_status(
        self, account: Account, order_id: str,
    ) -> OrderDetail:
        """Poll order status via REST."""
        ...

    @abstractmethod
    async def get_positions(self, account: Account) -> list[PositionDetail]:
        """Get all positions for this account."""
        ...

    @abstractmethod
    async def place_forever_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        """Place a GTT / Forever order."""
        ...

    @abstractmethod
    async def place_slicing_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        validity: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        """Place an order that auto-slices above freeze quantity."""
        ...

    @abstractmethod
    async def get_trades(self, account: Account, order_id: str) -> list[dict]:
        """Get individual trade executions for an order."""
        ...

    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Return broker identifier."""
        ...
```

#### DhanAdapter Implementation

```python
class DhanAdapter(BrokerAdapter):
    """
    Dhan broker adapter. Translates BrokerAdapter calls into
    Dhan REST API requests via the per-account DhanClient.
    """

    def __init__(self, client: DhanClient):
        self._client = client

    @property
    def broker_name(self) -> str:
        return "dhan"

    async def place_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        validity: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        body = {
            "dhanClientId": account.dhan_client_id,
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,
            "productType": product_type,
            "orderType": order_type,
            "validity": validity,
            "securityId": security_id,
            "quantity": quantity,
            "price": price,
            "triggerPrice": trigger_price,
            "disclosedQuantity": 0,
            "afterMarketOrder": False,
            "amoTime": None,
            "boProfitValue": None,
            "boStopLossValue": None,
            "correlationId": correlation_id,
        }

        raw = await self._client.place_order(body)

        return OrderResponse(
            order_id=raw["orderId"],
            status=raw["orderStatus"],
            broker="dhan",
            raw_response=raw,
        )

    async def modify_order(
        self,
        account: Account,
        order_id: str,
        order_type: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        validity: str = "DAY",
    ) -> OrderResponse:
        body = {
            "dhanClientId": account.dhan_client_id,
            "orderId": order_id,
            "orderType": order_type,
            "legName": None,
            "quantity": quantity,
            "price": price,
            "triggerPrice": trigger_price,
            "disclosedQuantity": 0,
            "validity": validity,
        }

        raw = await self._client.modify_order(order_id, body)

        return OrderResponse(
            order_id=raw["orderId"],
            status=raw.get("orderStatus", "PENDING"),
            broker="dhan",
            raw_response=raw,
        )

    async def cancel_order(
        self, account: Account, order_id: str,
    ) -> OrderResponse:
        raw = await self._client.cancel_order(order_id)
        return OrderResponse(
            order_id=raw["orderId"],
            status=raw.get("orderStatus", "CANCELLED"),
            broker="dhan",
            raw_response=raw,
        )

    async def get_order_status(
        self, account: Account, order_id: str,
    ) -> OrderDetail:
        raw = await self._client.get_order(order_id)
        return OrderDetail(
            order_id=raw["orderId"],
            status=raw["orderStatus"],
            filled_qty=raw["filledQty"],
            remaining_qty=raw["remainingQuantity"],
            avg_fill_price=raw["averageTradedPrice"],
            exchange_order_id=raw.get("exchangeOrderId"),
            exchange_time=raw.get("exchangeTime"),
            broker="dhan",
        )

    async def get_positions(self, account: Account) -> list[PositionDetail]:
        raw = await self._client.get_positions()
        positions = []
        for p in raw.get("data", []):
            net_qty = p.get("netQty", 0)
            if net_qty == 0:
                continue
            positions.append(PositionDetail(
                instrument_id=p["securityId"],
                exchange_segment=p["exchangeSegment"],
                product_type=p["productType"],
                quantity=net_qty,
                avg_price=p.get("averagePrice", 0.0),
                pnl=p.get("realizedProfit", 0.0) + p.get("unrealizedProfit", 0.0),
                broker="dhan",
            ))
        return positions

    async def place_forever_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        body = {
            "dhanClientId": account.dhan_client_id,
            "orderFlag": "SINGLE",
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,
            "productType": product_type,
            "orderType": order_type,
            "securityId": security_id,
            "quantity": quantity,
            "price": price,
            "triggerPrice": trigger_price,
            "correlationId": correlation_id,
        }

        raw = await self._client.place_forever_order(body)

        return OrderResponse(
            order_id=raw["orderId"],
            status=raw.get("orderStatus", "PENDING"),
            broker="dhan",
            raw_response=raw,
        )

    async def place_slicing_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        validity: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        body = {
            "dhanClientId": account.dhan_client_id,
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,
            "productType": product_type,
            "orderType": order_type,
            "validity": validity,
            "securityId": security_id,
            "quantity": quantity,
            "price": price,
            "triggerPrice": trigger_price,
            "disclosedQuantity": 0,
            "afterMarketOrder": False,
            "amoTime": None,
            "boProfitValue": None,
            "boStopLossValue": None,
            "correlationId": correlation_id,
        }

        raw = await self._client.place_slicing_order(body)

        return OrderResponse(
            order_id=raw["orderId"],
            status=raw.get("orderStatus", "PENDING"),
            broker="dhan",
            raw_response=raw,
        )

    async def get_trades(self, account: Account, order_id: str) -> list[dict]:
        raw = await self._client.get_trades(order_id)
        return raw.get("trades", raw.get("data", []))
```

#### UpstoxAdapter (Failover Stub)

```python
class UpstoxClient:
    """
    HTTP client for the Upstox REST API. Structurally identical to DhanClient.
    Created on demand during failover.

    Base URL: https://api.upstox.com/v2
    Auth header: Authorization: Bearer {access_token}
    Rate limit: 50 OPS per API key (server-side)
    """

    BASE_URL = "https://api.upstox.com/v2"
    HFT_BASE_URL = "https://api-hft.upstox.com/v2"
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 0.5
    REQUEST_TIMEOUT = 10.0

    def __init__(self, account: Account):
        self.account = account
        self._token: str | None = None
        self._session: aiohttp.ClientSession | None = None

    @property
    def auth_headers(self) -> dict[str, str]:
        if self._token is None:
            raise RuntimeError(
                f"UpstoxClient for {self.account.account_id} not authenticated"
            )
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def set_token(self, token: str) -> None:
        self._token = token

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
        self._session = aiohttp.ClientSession(
            base_url=self.BASE_URL,
            headers=self.auth_headers,
            timeout=timeout,
        )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def request(
        self, method: str, path: str, body: dict | None = None,
    ) -> dict:
        """Same retry logic as DhanClient."""
        # Implementation mirrors DhanClient.request with Upstox-specific
        # error codes and status handling.
        ...

    async def place_order(self, body: dict) -> dict:
        return await self.request("POST", "/order/place", body)

    async def modify_order(self, body: dict) -> dict:
        return await self.request("PUT", "/order/modify", body)

    async def cancel_order(self, order_id: str) -> dict:
        return await self.request("DELETE", f"/order/cancel?order_id={order_id}")

    async def get_order(self, order_id: str) -> dict:
        return await self.request("GET", f"/order/details?order_id={order_id}")

    async def get_positions(self) -> dict:
        return await self.request("GET", "/portfolio/short-term-positions")


class UpstoxAdapter(BrokerAdapter):
    """
    Upstox broker adapter for failover. Translates BrokerAdapter calls
    into Upstox REST API requests.

    Field mapping differences from Dhan:
    - security_id → instrument_key (via SymbolMapper)
    - exchangeSegment "NSE_FNO" → exchange "NSE", segment "FO"
    - productType "INTRADAY" → product "I"
    - orderType "LIMIT" → order_type "LIMIT" (same name, different casing)
    """

    PRODUCT_MAP = {
        "INTRADAY": "I",
        "CNC": "D",
        "MARGIN": "D",
    }

    def __init__(self, client: UpstoxClient, symbol_mapper: "SymbolMapper"):
        self._client = client
        self._mapper = symbol_mapper

    @property
    def broker_name(self) -> str:
        return "upstox"

    async def place_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        validity: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        # Translate canonical security_id to Upstox instrument_key
        instrument_key = self._mapper.dhan_to_upstox(security_id)
        if instrument_key is None:
            raise SymbolMappingError(
                f"No Upstox mapping for Dhan security_id {security_id}"
            )

        body = {
            "quantity": quantity,
            "product": self.PRODUCT_MAP.get(product_type, "I"),
            "validity": validity,
            "price": price,
            "tag": correlation_id or "",
            "instrument_token": instrument_key,
            "order_type": order_type.upper(),
            "transaction_type": transaction_type.upper(),
            "disclosed_quantity": 0,
            "trigger_price": trigger_price or 0,
            "is_amo": False,
        }

        raw = await self._client.place_order(body)
        data = raw.get("data", {})

        return OrderResponse(
            order_id=data.get("order_id", ""),
            status=self._normalize_status(data.get("status", "")),
            broker="upstox",
            raw_response=raw,
        )

    async def modify_order(
        self,
        account: Account,
        order_id: str,
        order_type: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        validity: str = "DAY",
    ) -> OrderResponse:
        body = {
            "order_id": order_id,
            "quantity": quantity,
            "validity": validity,
            "price": price,
            "order_type": order_type.upper(),
            "trigger_price": trigger_price or 0,
            "disclosed_quantity": 0,
        }

        raw = await self._client.modify_order(body)
        data = raw.get("data", {})

        return OrderResponse(
            order_id=data.get("order_id", order_id),
            status=self._normalize_status(data.get("status", "PENDING")),
            broker="upstox",
            raw_response=raw,
        )

    async def cancel_order(
        self, account: Account, order_id: str,
    ) -> OrderResponse:
        raw = await self._client.cancel_order(order_id)
        data = raw.get("data", {})
        return OrderResponse(
            order_id=data.get("order_id", order_id),
            status="CANCELLED",
            broker="upstox",
            raw_response=raw,
        )

    async def get_order_status(
        self, account: Account, order_id: str,
    ) -> OrderDetail:
        raw = await self._client.get_order(order_id)
        data = raw.get("data", {})

        return OrderDetail(
            order_id=data.get("order_id", order_id),
            status=self._normalize_status(data.get("status", "")),
            filled_qty=data.get("filled_quantity", 0),
            remaining_qty=data.get("pending_quantity", 0),
            avg_fill_price=data.get("average_price", 0.0),
            exchange_order_id=data.get("exchange_order_id"),
            exchange_time=data.get("exchange_timestamp"),
            broker="upstox",
        )

    async def get_positions(self, account: Account) -> list[PositionDetail]:
        raw = await self._client.get_positions()
        positions = []
        for p in raw.get("data", []):
            net_qty = p.get("quantity", 0)
            if net_qty == 0:
                continue
            # Reverse-map Upstox instrument_key to Dhan security_id
            canonical_id = self._mapper.upstox_to_dhan(
                p.get("instrument_token", "")
            )
            positions.append(PositionDetail(
                instrument_id=canonical_id or p.get("instrument_token", ""),
                exchange_segment=p.get("exchange", ""),
                product_type=p.get("product", ""),
                quantity=net_qty,
                avg_price=p.get("average_price", 0.0),
                pnl=p.get("pnl", 0.0),
                broker="upstox",
            ))
        return positions

    async def place_forever_order(self, account, **kwargs) -> OrderResponse:
        # Upstox uses GTT API: POST /v2/gtt/orders
        # Structurally similar to Dhan Forever Orders
        raise NotImplementedError("Upstox GTT not implemented in v1 failover")

    async def place_slicing_order(self, account, **kwargs) -> OrderResponse:
        # Upstox does not have an auto-slicing endpoint.
        # Manual slicing required at the OMS level.
        raise NotImplementedError("Manual slicing required for Upstox")

    async def get_trades(self, account: Account, order_id: str) -> list[dict]:
        raw = await self._client.request("GET", f"/order/trades?order_id={order_id}")
        return raw.get("data", [])

    @staticmethod
    def _normalize_status(upstox_status: str) -> str:
        """Map Upstox order status to canonical status."""
        mapping = {
            "open": "PENDING",
            "complete": "TRADED",
            "partial fill": "PART_TRADED",
            "cancelled": "CANCELLED",
            "rejected": "REJECTED",
            "after market order req received": "TRANSIT",
            "modify pending": "TRANSIT",
            "cancel pending": "TRANSIT",
            "not cancelled": "PENDING",
            "not modified": "PENDING",
        }
        return mapping.get(upstox_status.lower(), "TRANSIT")
```

---

### Symbol Mapping

#### Canonical Identifier

The system uses Dhan `security_id` as the canonical instrument identifier everywhere: in signals, order requests, position tracking, risk calculations, and audit logs. When interacting with a non-Dhan broker, the `SymbolMapper` translates to that broker's native identifier.

| Broker | Identifier Field | Example (NIFTY 24500 CE weekly) |
|--------|-----------------|--------------------------------|
| Dhan | `security_id` (string of integer) | `"43925"` |
| Upstox | `instrument_key` (exchange\|segment\|symbol) | `"NSE_FO\|NIFTY24MAR24500CE"` |
| Angel One | `symboltoken` (string of integer) | `"58127"` |

#### SymbolMapper Class

```python
import csv
import io
import aiohttp
import asyncio
from datetime import date


class SymbolMapper:
    """
    Cross-broker instrument identifier mapping.

    Builds mapping tables from broker CSV instrument masters at 08:30 IST daily.
    The Dhan security_id is the canonical key. Maps to Upstox instrument_key
    and Angel One symboltoken using a composite match key:
    (exchange, symbol, expiry, strike, option_type).

    The composite key avoids ambiguity: two brokers may assign different
    numeric IDs to the same instrument, but the exchange-level attributes
    (underlying, expiry date, strike price, CE/PE) uniquely identify it.
    """

    DHAN_CSV_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
    UPSTOX_CSV_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
    ANGEL_CSV_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

    def __init__(self):
        # Primary maps: canonical (Dhan security_id) → broker-specific ID
        self._dhan_to_upstox: dict[str, str] = {}
        self._upstox_to_dhan: dict[str, str] = {}
        self._dhan_to_angel: dict[str, str] = {}
        self._angel_to_dhan: dict[str, str] = {}

        # Metadata for the canonical ID
        self._dhan_meta: dict[str, InstrumentMeta] = {}

        # Build status
        self._built: bool = False
        self._build_date: date | None = None
        self._build_errors: list[str] = []

    async def build(self) -> None:
        """
        Download all three broker CSVs and build cross-mapping tables.
        Called at 08:30 IST during system startup (Phase 2: Data Prep).

        Downloads happen in parallel. If Upstox or Angel CSV fails,
        the system continues with Dhan-only mode (failover will be
        degraded but primary broker still works).
        """
        logger.info("symbol_mapper_build_start")

        # Download all CSVs in parallel
        dhan_task = asyncio.create_task(self._download_dhan_csv())
        upstox_task = asyncio.create_task(self._download_upstox_csv())

        dhan_instruments = await dhan_task

        try:
            upstox_instruments = await upstox_task
        except Exception as e:
            logger.error("upstox_csv_download_failed", error=str(e))
            upstox_instruments = {}
            self._build_errors.append(f"Upstox CSV failed: {e}")

        # Build Dhan metadata index (canonical)
        for inst in dhan_instruments:
            meta = InstrumentMeta(
                security_id=inst["SEM_SMST_SECURITY_ID"],
                trading_symbol=inst["SEM_TRADING_SYMBOL"],
                exchange=inst["SEM_EXM_EXCH_ID"],
                segment=inst.get("SEM_SEGMENT", ""),
                instrument_type=inst.get("SEM_INSTRUMENT_NAME", ""),
                underlying=inst.get("SEM_CUSTOM_SYMBOL", ""),
                expiry=inst.get("SEM_EXPIRY_DATE", ""),
                strike=float(inst.get("SEM_STRIKE_PRICE", 0)),
                option_type=inst.get("SEM_OPTION_TYPE", ""),
                lot_size=int(inst.get("SEM_LOT_UNITS", 1)),
                tick_size=float(inst.get("SEM_TICK_SIZE", 0.05)),
                freeze_qty=int(inst.get("FREEZE_QTY", 0)),
            )
            self._dhan_meta[meta.security_id] = meta

        # Build Upstox composite key index
        upstox_by_key: dict[str, str] = {}
        for inst in upstox_instruments:
            composite = self._make_composite_key(
                exchange=inst.get("exchange", ""),
                symbol=inst.get("tradingsymbol", ""),
                expiry=inst.get("expiry", ""),
                strike=float(inst.get("strike", 0)),
                option_type=inst.get("option_type", ""),
            )
            upstox_by_key[composite] = inst.get("instrument_key", "")

        # Cross-map: for each Dhan instrument, find Upstox equivalent
        matched = 0
        unmatched = 0
        for sec_id, meta in self._dhan_meta.items():
            composite = self._make_composite_key(
                exchange=meta.exchange,
                symbol=meta.trading_symbol,
                expiry=meta.expiry,
                strike=meta.strike,
                option_type=meta.option_type,
            )
            upstox_key = upstox_by_key.get(composite)
            if upstox_key:
                self._dhan_to_upstox[sec_id] = upstox_key
                self._upstox_to_dhan[upstox_key] = sec_id
                matched += 1
            else:
                unmatched += 1

        self._built = True
        self._build_date = date.today()

        logger.info(
            "symbol_mapper_build_complete",
            dhan_instruments=len(dhan_instruments),
            upstox_instruments=len(upstox_instruments),
            matched=matched,
            unmatched=unmatched,
        )

    @staticmethod
    def _make_composite_key(
        exchange: str,
        symbol: str,
        expiry: str,
        strike: float,
        option_type: str,
    ) -> str:
        """
        Composite key for cross-broker matching.
        Normalizes fields to handle formatting differences between brokers.
        """
        # Normalize exchange: NSE, BSE
        norm_exchange = exchange.upper().replace("_EQ", "").replace("_FNO", "")
        # Normalize expiry to YYYY-MM-DD
        norm_expiry = expiry[:10] if expiry else ""
        # Normalize strike: remove trailing zeros
        norm_strike = f"{strike:.2f}".rstrip("0").rstrip(".")
        # Normalize option type
        norm_opt = option_type.upper()[:2] if option_type else ""

        return f"{norm_exchange}|{symbol}|{norm_expiry}|{norm_strike}|{norm_opt}"

    async def _download_dhan_csv(self) -> list[dict]:
        """Download and parse Dhan instrument master CSV."""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.DHAN_CSV_URL, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Dhan CSV download failed: HTTP {resp.status}")
                text = await resp.text()

        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    async def _download_upstox_csv(self) -> list[dict]:
        """Download and parse Upstox instrument master CSV (gzipped)."""
        import gzip

        async with aiohttp.ClientSession() as session:
            async with session.get(self.UPSTOX_CSV_URL, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Upstox CSV download failed: HTTP {resp.status}")
                raw_bytes = await resp.read()

        text = gzip.decompress(raw_bytes).decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    # --- Lookup methods ---

    def dhan_to_upstox(self, security_id: str) -> str | None:
        """Translate Dhan security_id to Upstox instrument_key."""
        return self._dhan_to_upstox.get(security_id)

    def upstox_to_dhan(self, instrument_key: str) -> str | None:
        """Translate Upstox instrument_key to Dhan security_id."""
        return self._upstox_to_dhan.get(instrument_key)

    def dhan_to_angel(self, security_id: str) -> str | None:
        """Translate Dhan security_id to Angel One symboltoken."""
        return self._dhan_to_angel.get(security_id)

    def get_meta(self, security_id: str) -> "InstrumentMeta | None":
        """Get instrument metadata by canonical ID."""
        return self._dhan_meta.get(security_id)

    def validate_mapping(self, security_id: str, target_broker: str) -> bool:
        """Check if a mapping exists for the given security_id and broker."""
        if target_broker == "upstox":
            return security_id in self._dhan_to_upstox
        if target_broker == "angel":
            return security_id in self._dhan_to_angel
        return True  # Dhan-to-Dhan always valid

    @property
    def build_status(self) -> dict:
        return {
            "built": self._built,
            "build_date": str(self._build_date),
            "dhan_count": len(self._dhan_meta),
            "upstox_mapped": len(self._dhan_to_upstox),
            "angel_mapped": len(self._dhan_to_angel),
            "errors": self._build_errors,
        }


class InstrumentMeta(pydantic.BaseModel):
    """Instrument metadata from the canonical (Dhan) instrument master."""
    security_id: str
    trading_symbol: str
    exchange: str
    segment: str
    instrument_type: str              # OPTIDX, FUTIDX, EQUITY, etc.
    underlying: str                    # NIFTY, BANKNIFTY, etc.
    expiry: str                        # YYYY-MM-DD or empty for equity
    strike: float                      # 0 for futures/equity
    option_type: str                   # CE, PE, or empty
    lot_size: int
    tick_size: float
    freeze_qty: int


class SymbolMappingError(Exception):
    pass
```

#### Startup Validation

At 08:30, after the SymbolMapper builds, a validation pass runs to confirm all instruments the strategies will trade are mapped:

```python
async def validate_strategy_instruments(
    mapper: SymbolMapper,
    strategy_configs: list["StrategyConfig"],
) -> list[str]:
    """
    Validate that all instruments the strategies might trade have
    cross-broker mappings. Returns list of warnings for unmapped instruments.

    Called at startup, before trading begins.
    """
    warnings = []

    for config in strategy_configs:
        for underlying in config.underlyings:
            # Check that the underlying's FNO segment instruments
            # have Upstox mappings (for failover readiness)
            meta_list = [
                m for m in mapper._dhan_meta.values()
                if m.underlying == underlying
                and m.segment in ("NSE_FNO", "FNO")
            ]

            mapped = sum(
                1 for m in meta_list
                if mapper.validate_mapping(m.security_id, "upstox")
            )
            total = len(meta_list)

            if total > 0 and mapped / total < 0.95:
                msg = (
                    f"Strategy {config.strategy_id}: "
                    f"{underlying} has only {mapped}/{total} "
                    f"({100*mapped/total:.1f}%) Upstox mappings"
                )
                warnings.append(msg)
                logger.warning("symbol_mapping_coverage_low", message=msg)

    return warnings
```

---

### Auth Lifecycle

#### Parallel Account Authentication

All accounts authenticate in parallel at 08:25 IST (Phase 1 of the startup sequence). Each account's token is stored in Redis for cross-process access.

```python
class AuthManager:
    """
    Manages per-account authentication with all configured brokers.

    08:25 IST: authenticate all accounts in parallel
    Token storage: Redis AUTH:{broker}:{account_id}:token
    Failed auth: SUSPEND that account (no orders, existing SLs remain)
    Shutdown: logout all accounts via atexit
    """

    DHAN_LOGIN_URL = "https://api.dhan.co/v2/token"
    TOKEN_TTL_SECONDS = 86400          # 24 hours (Dhan tokens valid for 1 day)
    AUTH_TIMEOUT = 30.0                # seconds per auth attempt
    MAX_AUTH_RETRIES = 3

    def __init__(self, router: "BrokerRouter"):
        self._router = router
        self._redis: "aioredis.Redis | None" = None

    async def authenticate_all(
        self,
        accounts: list[Account],
        redis: "aioredis.Redis",
    ) -> dict[str, bool]:
        """
        Authenticate all accounts in parallel. Returns dict of
        account_id -> success.

        Failed accounts are set to SUSPENDED state. They receive no new
        orders but existing server-side SLs remain active on the broker.
        """
        self._redis = redis
        results: dict[str, bool] = {}

        tasks = [
            asyncio.create_task(
                self._auth_single_account(account),
                name=f"auth_{account.account_id}",
            )
            for account in accounts
        ]

        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for account, result in zip(accounts, completed):
            if isinstance(result, Exception):
                logger.critical(
                    "account_auth_failed",
                    account_id=account.account_id,
                    error=str(result),
                )
                self._router._broker_state[account.account_id].health = (
                    BrokerHealth.SUSPENDED
                )
                results[account.account_id] = False
                await telegram.send(
                    CRITICAL,
                    f"AUTH FAILED: Account {account.account_id}. "
                    f"Suspended for today. Error: {result}",
                )
            else:
                results[account.account_id] = True

        auth_ok = sum(1 for v in results.values() if v)
        auth_fail = sum(1 for v in results.values() if not v)
        logger.info(
            "auth_all_complete",
            total=len(accounts),
            success=auth_ok,
            failed=auth_fail,
        )

        if auth_ok == 0:
            raise SystemStartupError(
                "All accounts failed authentication. Cannot start trading."
            )

        return results

    async def _auth_single_account(self, account: Account) -> None:
        """
        Authenticate a single account with its primary broker.

        For Dhan:
        1. Use the API key to obtain an access token
        2. Store token in Redis with TTL
        3. Set token on the DhanClient instance
        4. Verify token by making a test API call (GET /v2/orders)
        """
        last_error: Exception | None = None

        for attempt in range(self.MAX_AUTH_RETRIES):
            try:
                # Dhan auth: API key is the access token (no separate login flow)
                # The api_key provided at account setup IS the access token
                token = account.dhan_access_token

                # Verify token is valid by making a lightweight call
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{DhanClient.BASE_URL}/v2/orders",
                        headers={
                            "access-token": token,
                            "Content-Type": "application/json",
                        },
                        timeout=aiohttp.ClientTimeout(total=self.AUTH_TIMEOUT),
                    ) as resp:
                        if resp.status == 401:
                            raise DhanAuthError(
                                account.account_id,
                                "Token rejected (HTTP 401)",
                            )
                        if resp.status >= 500:
                            raise DhanAPIError(
                                account.account_id,
                                resp.status,
                                None,
                                "Auth verification failed (server error)",
                            )
                        # 200 or other 2xx/4xx = token is valid (4xx might be
                        # "no orders" which is fine for verification)

                # Store in Redis
                redis_key = f"AUTH:dhan:{account.account_id}:token"
                await self._redis.set(redis_key, token, ex=self.TOKEN_TTL_SECONDS)

                # Set on the DhanClient instance
                client = self._router._dhan_clients[account.account_id]
                client.set_token(token)
                await client.start()

                # Update broker state
                state = self._router._broker_state[account.account_id]
                state.health = BrokerHealth.HEALTHY
                state.record_success()

                logger.info(
                    "account_auth_success",
                    account_id=account.account_id,
                    broker="dhan",
                    attempt=attempt + 1,
                )
                return

            except (DhanAuthError, DhanAPIError) as e:
                last_error = e
                logger.warning(
                    "account_auth_attempt_failed",
                    account_id=account.account_id,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.MAX_AUTH_RETRIES - 1:
                    await asyncio.sleep(2.0 * (attempt + 1))

        raise last_error or RuntimeError(
            f"Auth failed for {account.account_id} after {self.MAX_AUTH_RETRIES} attempts"
        )

    async def refresh_token(self, account: Account) -> bool:
        """
        Refresh an account's token mid-session. Called when a 401 is
        detected during normal operation.

        Returns True if refresh succeeded, False if account must be suspended.
        """
        try:
            await self._auth_single_account(account)
            logger.info("token_refresh_success", account_id=account.account_id)
            return True
        except Exception as e:
            logger.critical(
                "token_refresh_failed",
                account_id=account.account_id,
                error=str(e),
            )
            self._router._broker_state[account.account_id].health = (
                BrokerHealth.SUSPENDED
            )
            await telegram.send(
                CRITICAL,
                f"TOKEN REFRESH FAILED: Account {account.account_id}. "
                f"Suspended. Manual re-auth required.",
            )
            return False

    async def logout_all(self) -> None:
        """
        Clean shutdown: close all broker sessions. Registered via
        atexit.register(asyncio.run, auth_manager.logout_all).

        Tokens are NOT invalidated (Dhan tokens expire naturally after 24h).
        This only closes HTTP connections and cleans up resources.
        """
        for account_id, client in self._router._dhan_clients.items():
            try:
                await client.close()
                logger.info("client_closed", account_id=account_id, broker="dhan")
            except Exception:
                pass  # best-effort on shutdown

        for account_id, client in self._router._upstox_clients.items():
            try:
                await client.close()
                logger.info("client_closed", account_id=account_id, broker="upstox")
            except Exception:
                pass

        logger.info("logout_all_complete")
```

#### Token Redis Layout

```
AUTH:dhan:{account_id}:token        → access token string (TTL 86400s)
AUTH:dhan:{account_id}:auth_ts      → ISO timestamp of last auth
AUTH:dhan:{account_id}:status       → "active" | "expired" | "suspended"
AUTH:upstox:{account_id}:token      → access token string (TTL 86400s)
AUTH:upstox:{account_id}:auth_ts    → ISO timestamp of last auth
AUTH:upstox:{account_id}:status     → "active" | "expired" | "suspended"
```

#### atexit Registration

```python
import atexit
import asyncio


def register_shutdown(auth_manager: AuthManager) -> None:
    """Register clean shutdown handler at system startup."""

    def _sync_logout():
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(auth_manager.logout_all())
            else:
                loop.run_until_complete(auth_manager.logout_all())
        except Exception:
            pass  # best-effort

    atexit.register(_sync_logout)
```

---

### Failover Logic

#### Health Monitoring

The broker router runs a continuous health monitor per account. Health is assessed from two signals: REST API success/failure rates and WebSocket connection state.

```python
class BrokerHealthMonitor:
    """
    Continuous health monitor for per-account broker connections.
    Runs as a background asyncio task. Checks health every 10 seconds.
    """

    CHECK_INTERVAL = 10.0              # seconds
    CONSECUTIVE_FAILURE_THRESHOLD = 3   # failures before FAILED state
    WS_DISCONNECT_THRESHOLD = 30.0     # seconds of WS disconnect before FAILED
    FAILBACK_COOLDOWN = 300.0          # seconds: wait 5 min before failback

    def __init__(self, router: "BrokerRouter"):
        self._router = router
        self._running = False

    async def run(self) -> None:
        """Main monitoring loop. One loop covers all accounts."""
        self._running = True

        while self._running:
            for account_id, state in self._router._broker_state.items():
                if state.health == BrokerHealth.SUSPENDED:
                    continue  # skip suspended accounts

                await self._check_account_health(account_id, state)

            await asyncio.sleep(self.CHECK_INTERVAL)

    async def _check_account_health(
        self, account_id: str, state: AccountBrokerState,
    ) -> None:
        # Check WS disconnect duration
        if state.ws_disconnect_ts is not None:
            disconnect_duration = time.monotonic() - state.ws_disconnect_ts
            if disconnect_duration > self.WS_DISCONNECT_THRESHOLD:
                if state.health != BrokerHealth.FAILED:
                    logger.warning(
                        "ws_disconnect_threshold_exceeded",
                        account_id=account_id,
                        duration_s=disconnect_duration,
                    )
                    state.health = BrokerHealth.FAILED

        # Check consecutive REST failures
        if state.consecutive_failures >= self.CONSECUTIVE_FAILURE_THRESHOLD:
            if state.health not in (
                BrokerHealth.FAILED, BrokerHealth.FAILOVER_ACTIVE,
            ):
                state.health = BrokerHealth.FAILED

        # Trigger failover if FAILED and not already in failover
        if (
            state.health == BrokerHealth.FAILED
            and state.active_broker == state.primary_broker
        ):
            await self._initiate_failover(account_id, state)

        # Check for failback eligibility
        if state.health == BrokerHealth.FAILOVER_ACTIVE:
            await self._check_failback(account_id, state)

    async def _initiate_failover(
        self, account_id: str, state: AccountBrokerState,
    ) -> None:
        """
        Failover procedure for a single account:
        1. Hedge open positions on backup broker
        2. Switch active broker to backup
        3. Continue monitoring primary for recovery
        """
        logger.critical(
            "failover_initiated",
            account_id=account_id,
            primary=state.primary_broker,
            reason=f"consecutive_failures={state.consecutive_failures}, "
                   f"ws_connected={state.ws_connected}",
        )

        account = self._router._accounts[account_id]

        # Initialize backup broker client if needed
        backup_broker = "upstox" if state.primary_broker == "dhan" else "dhan"

        if backup_broker == "upstox" and account_id not in self._router._upstox_clients:
            if not account.upstox_client_id or not account.upstox_api_key:
                logger.critical(
                    "failover_no_backup_configured",
                    account_id=account_id,
                )
                await telegram.send(
                    CRITICAL,
                    f"FAILOVER FAILED: Account {account_id} has no Upstox "
                    f"credentials configured. Positions protected by server-side SLs only.",
                )
                return

            # Auth with backup broker
            try:
                upstox_client = UpstoxClient(account)
                upstox_client.set_token(account.upstox_api_key)
                await upstox_client.start()
                self._router._upstox_clients[account_id] = upstox_client
            except Exception as e:
                logger.critical(
                    "failover_backup_auth_failed",
                    account_id=account_id,
                    error=str(e),
                )
                await telegram.send(
                    CRITICAL,
                    f"FAILOVER FAILED: Upstox auth failed for {account_id}. "
                    f"Positions protected by server-side SLs only. Error: {e}",
                )
                return

        # Hedge open positions on backup broker
        await self._router._hedge_manager.hedge_account(account_id, backup_broker)

        # Switch active broker
        state.active_broker = backup_broker
        state.health = BrokerHealth.FAILOVER_ACTIVE
        state.failover_ts = time.monotonic()
        state.failover_reason = (
            f"consecutive_failures={state.consecutive_failures}, "
            f"ws_connected={state.ws_connected}"
        )
        state.failback_eligible_ts = time.monotonic() + self.FAILBACK_COOLDOWN

        await telegram.send(
            CRITICAL,
            f"FAILOVER ACTIVE: Account {account_id} switched from "
            f"{state.primary_broker} to {backup_broker}. "
            f"Hedge positions placed. Monitoring primary for recovery.",
        )

    async def _check_failback(
        self, account_id: str, state: AccountBrokerState,
    ) -> None:
        """
        Check if primary broker has recovered and failback is safe.

        Failback requirements:
        1. At least FAILBACK_COOLDOWN seconds since failover
        2. Primary broker responds to a health-check API call
        3. Primary broker WS reconnects successfully
        """
        if state.failback_eligible_ts and time.monotonic() < state.failback_eligible_ts:
            return  # cooldown not elapsed

        # Probe primary broker with a lightweight call
        try:
            account = self._router._accounts[account_id]

            if state.primary_broker == "dhan":
                client = self._router._dhan_clients[account_id]
                await client.request("GET", "/v2/orders")
            else:
                # Upstox health probe
                pass

            # Primary is back. Initiate failback.
            await self._initiate_failback(account_id, state)

        except Exception:
            # Primary still down. Check again next cycle.
            pass

    async def _initiate_failback(
        self, account_id: str, state: AccountBrokerState,
    ) -> None:
        """
        Failback to primary broker:
        1. Unwind hedge positions on backup broker
        2. Switch active broker back to primary
        3. Re-establish WS connections
        """
        logger.info("failback_initiated", account_id=account_id)

        # Unwind hedges on backup broker
        await self._router._hedge_manager.unwind_hedges(account_id)

        # Switch back to primary
        state.active_broker = state.primary_broker
        state.health = BrokerHealth.HEALTHY
        state.consecutive_failures = 0
        state.failover_ts = None
        state.failover_reason = None
        state.failback_eligible_ts = None

        # Re-establish WS
        ws = self._router._order_ws.get(account_id)
        if ws:
            try:
                await ws.connect()
                state.ws_connected = True
                state.ws_disconnect_ts = None
            except Exception as e:
                logger.warning(
                    "failback_ws_reconnect_failed",
                    account_id=account_id,
                    error=str(e),
                )

        await telegram.send(
            WARNING,
            f"FAILBACK COMPLETE: Account {account_id} returned to "
            f"{state.primary_broker}. Hedges unwound.",
        )
```

#### Failover State Machine

```
                       ┌──────────────────────────────────┐
                       │                                  │
                       ▼                                  │
                 ┌───────────┐                            │
          ┌─────│  HEALTHY   │◄───────────────────┐       │
          │     └─────┬──────┘                    │       │
          │           │                            │       │
          │           │ intermittent failure        │       │
          │           │ (1-2 consecutive)           │       │
          │           ▼                            │       │
          │     ┌───────────┐                      │       │
          │     │  DEGRADED  │                     │       │
          │     └─────┬──────┘                     │       │
          │           │                            │       │
          │           │ success                    │       │
          │           │ (resets count)              │       │
          │           └────────────────────────────┘       │
          │                                                │
          │  3 consecutive failures                        │
          │  OR WS disconnect > 30s                        │
          │                                                │
          ▼                                                │
    ┌───────────┐                                          │
    │   FAILED   │                                         │
    └─────┬──────┘                                         │
          │                                                │
          │ backup broker available                        │
          │ + hedge positions placed                       │
          ▼                                                │
    ┌───────────────┐                                      │
    │ FAILOVER_ACTIVE│                                     │
    └─────┬─────────┘                                      │
          │                                                │
          │ primary recovers                               │
          │ + cooldown elapsed (5 min)                     │
          │ + hedges unwound                               │
          │                                                │
          └────────────────────────────────────────────────┘

    ┌───────────┐
    │ SUSPENDED  │  ← auth failure or operator action
    └───────────┘    (no automatic recovery)
```

---

### Cross-Broker Hedging (Finding 4)

When Dhan fails for a specific account that has open positions, those positions must be hedged on the backup broker. Each account is hedged independently. Account A's Dhan failure does not affect Account B.

#### Hedge Strategy

| Position Type | Hedge Method | Hedge Instrument |
|--------------|-------------|------------------|
| Long futures | Short same future on Upstox | Same underlying, same expiry |
| Short futures | Long same future on Upstox | Same underlying, same expiry |
| Long call option | Short ATM call on Upstox (delta-adjusted) | Same underlying, nearest strike, same expiry |
| Short call option | Long ATM call on Upstox (delta-adjusted) | Same underlying, nearest strike, same expiry |
| Long put option | Short ATM put on Upstox (delta-adjusted) | Same underlying, nearest strike, same expiry |
| Short put option | Long ATM put on Upstox (delta-adjusted) | Same underlying, nearest strike, same expiry |

**Delta adjustment for options:** The hedge quantity is scaled by the position's delta. A 10-lot long call at delta 0.5 requires a 5-lot hedge (short call at delta ~0.5) or a 5-lot short futures hedge. The system uses the simpler option-for-option hedge where possible, falling back to futures for illiquid strikes.

#### HedgeManager Implementation

```python
class HedgePosition(pydantic.BaseModel):
    """Tracks a hedge position placed on the backup broker."""
    account_id: str
    original_security_id: str           # canonical ID of the position being hedged
    hedge_security_id: str              # canonical ID of the hedge instrument
    hedge_broker: Literal["dhan", "upstox"]
    hedge_order_id: str
    hedge_qty: int                      # signed: positive = long, negative = short
    hedge_price: float
    hedge_status: Literal["PENDING", "FILLED", "PARTIAL", "FAILED"]
    delta_ratio: float                  # 1.0 for futures, 0.0-1.0 for options
    placed_ts: float


class HedgeManager:
    """
    Places and manages cross-broker hedges during failover.

    Hedging rules:
    1. Futures: perfect hedge (1:1 opposite position on backup broker)
    2. Options: delta-adjusted hedge (scale qty by position delta)
    3. Each account hedged independently
    4. Hedge uses IOC validity to avoid hanging orders
    5. Failed hedges: Telegram CRITICAL, operator must intervene
    """

    def __init__(self, router: "BrokerRouter"):
        self._router = router
        self._hedges: dict[str, list[HedgePosition]] = {}  # account_id → hedges

    async def hedge_account(
        self, account_id: str, backup_broker: str,
    ) -> list[HedgePosition]:
        """
        Hedge all open positions for one account on the backup broker.

        Steps:
        1. Query primary broker for open positions (may fail — use cached)
        2. For each position, determine hedge instrument on backup broker
        3. Place hedge orders on backup broker
        4. Track hedge positions for later unwinding
        """
        account = self._router._accounts[account_id]
        hedges: list[HedgePosition] = []

        # Try to get positions from primary broker
        positions = await self._get_positions_best_effort(account_id)

        if not positions:
            logger.warning(
                "hedge_no_positions",
                account_id=account_id,
                reason="no_open_positions_or_query_failed",
            )
            return hedges

        # Get backup broker adapter
        adapter = self._router.get_client(account_id)
        if adapter.broker_name != backup_broker:
            # Force backup broker
            if backup_broker == "upstox":
                upstox_client = self._router._upstox_clients.get(account_id)
                if not upstox_client:
                    logger.critical(
                        "hedge_no_backup_client", account_id=account_id,
                    )
                    return hedges
                adapter = UpstoxAdapter(upstox_client, self._router._symbol_mapper)

        for position in positions:
            try:
                hedge = await self._hedge_single_position(
                    account, position, adapter, backup_broker,
                )
                if hedge:
                    hedges.append(hedge)
            except Exception as e:
                logger.critical(
                    "hedge_position_failed",
                    account_id=account_id,
                    security_id=position.instrument_id,
                    error=str(e),
                )
                await telegram.send(
                    CRITICAL,
                    f"HEDGE FAILED: Account {account_id}, "
                    f"instrument {position.instrument_id}. "
                    f"Position unhedged. Error: {e}",
                )

        self._hedges[account_id] = hedges

        # Update broker state with hedge info
        self._router._broker_state[account_id].hedge_positions = hedges

        logger.info(
            "hedge_account_complete",
            account_id=account_id,
            positions_found=len(positions),
            hedges_placed=len(hedges),
            hedges_failed=len(positions) - len(hedges),
        )

        return hedges

    async def _hedge_single_position(
        self,
        account: Account,
        position: PositionDetail,
        adapter: BrokerAdapter,
        backup_broker: str,
    ) -> HedgePosition | None:
        """Place a hedge for a single position."""
        meta = self._router._symbol_mapper.get_meta(position.instrument_id)
        if meta is None:
            logger.error(
                "hedge_no_meta",
                security_id=position.instrument_id,
            )
            return None

        # Determine hedge direction (opposite of position)
        hedge_txn = "SELL" if position.quantity > 0 else "BUY"
        hedge_qty = abs(position.quantity)

        # Delta adjustment for options
        delta_ratio = 1.0
        if meta.option_type in ("CE", "PE"):
            delta = self._estimate_delta(meta)
            delta_ratio = abs(delta)
            hedge_qty = max(
                meta.lot_size,
                int(hedge_qty * delta_ratio / meta.lot_size) * meta.lot_size,
            )

        # Translate security_id for backup broker
        if backup_broker == "upstox":
            hedge_security_id = self._router._symbol_mapper.dhan_to_upstox(
                position.instrument_id
            )
            if not hedge_security_id:
                logger.error(
                    "hedge_no_symbol_mapping",
                    security_id=position.instrument_id,
                    broker=backup_broker,
                )
                return None
        else:
            hedge_security_id = position.instrument_id

        # Place hedge order with MARKET + IOC for immediate fill
        try:
            resp = await adapter.place_order(
                account=account,
                transaction_type=hedge_txn,
                exchange_segment="NSE_FNO",
                product_type="INTRADAY",
                order_type="MARKET",
                validity="IOC",
                security_id=position.instrument_id,  # adapter translates internally
                quantity=hedge_qty,
                price=0,
                trigger_price=None,
                correlation_id=f"HEDGE_{account.account_id[:8]}_{position.instrument_id[:8]}",
            )

            return HedgePosition(
                account_id=account.account_id,
                original_security_id=position.instrument_id,
                hedge_security_id=hedge_security_id,
                hedge_broker=backup_broker,
                hedge_order_id=resp.order_id,
                hedge_qty=-hedge_qty if hedge_txn == "SELL" else hedge_qty,
                hedge_price=0.0,  # market order — actual price from fill
                hedge_status="PENDING",
                delta_ratio=delta_ratio,
                placed_ts=time.monotonic(),
            )

        except Exception as e:
            logger.critical(
                "hedge_order_placement_failed",
                account_id=account.account_id,
                security_id=position.instrument_id,
                error=str(e),
            )
            return None

    def _estimate_delta(self, meta: InstrumentMeta) -> float:
        """
        Estimate option delta for hedge sizing. Uses a rough approximation
        based on moneyness (ATM ~ 0.5, deep ITM ~ 0.9, deep OTM ~ 0.1).

        A full Black-Scholes delta calculation is not needed here because:
        1. Hedge is temporary (minutes to hours)
        2. Approximate hedge is better than no hedge
        3. Exact delta requires IV which may not be available during failover
        """
        # Without current spot price during failover, assume ATM
        # This is conservative: over-hedges OTM, under-hedges ITM
        if meta.option_type == "CE":
            return 0.5
        elif meta.option_type == "PE":
            return -0.5
        return 1.0  # futures

    async def _get_positions_best_effort(
        self, account_id: str,
    ) -> list[PositionDetail]:
        """
        Try to get positions from the primary broker. If that fails
        (which is likely during failover), fall back to the cached
        position state from the Position Tracker (Redis).
        """
        try:
            client = self._router._dhan_clients[account_id]
            adapter = DhanAdapter(client)
            account = self._router._accounts[account_id]
            return await adapter.get_positions(account)
        except Exception:
            # Primary is down — query cached positions from Redis
            return await self._get_cached_positions(account_id)

    async def _get_cached_positions(
        self, account_id: str,
    ) -> list[PositionDetail]:
        """Read cached positions from Redis POS:{account_id}:* keys."""
        # Position Tracker writes POS:{account_id}:{security_id} on every fill
        positions = []
        redis = self._router._auth_manager._redis
        if redis is None:
            return positions

        cursor = 0
        pattern = f"POS:{account_id}:*"
        while True:
            cursor, keys = await redis.scan(cursor, match=pattern, count=100)
            for key in keys:
                data = await redis.hgetall(key)
                if data and int(data.get(b"quantity", 0)) != 0:
                    positions.append(PositionDetail(
                        instrument_id=data.get(b"security_id", b"").decode(),
                        exchange_segment=data.get(b"exchange_segment", b"").decode(),
                        product_type=data.get(b"product_type", b"").decode(),
                        quantity=int(data.get(b"quantity", 0)),
                        avg_price=float(data.get(b"avg_price", 0)),
                        pnl=float(data.get(b"pnl", 0)),
                        broker="dhan",
                    ))
            if cursor == 0:
                break

        return positions

    async def unwind_hedges(self, account_id: str) -> None:
        """
        Unwind all hedge positions for an account after failback.
        Places opposite orders on the backup broker to close hedges.
        """
        hedges = self._hedges.get(account_id, [])
        if not hedges:
            return

        for hedge in hedges:
            try:
                # Close the hedge: opposite transaction
                close_txn = "BUY" if hedge.hedge_qty < 0 else "SELL"
                close_qty = abs(hedge.hedge_qty)

                if hedge.hedge_broker == "upstox":
                    client = self._router._upstox_clients.get(account_id)
                    if client:
                        adapter = UpstoxAdapter(client, self._router._symbol_mapper)
                        account = self._router._accounts[account_id]
                        await adapter.place_order(
                            account=account,
                            transaction_type=close_txn,
                            exchange_segment="NSE_FNO",
                            product_type="INTRADAY",
                            order_type="MARKET",
                            validity="IOC",
                            security_id=hedge.original_security_id,
                            quantity=close_qty,
                            price=0,
                            correlation_id=f"UNHEDGE_{account_id[:8]}",
                        )

                logger.info(
                    "hedge_unwound",
                    account_id=account_id,
                    security_id=hedge.original_security_id,
                    qty=close_qty,
                )

            except Exception as e:
                logger.critical(
                    "hedge_unwind_failed",
                    account_id=account_id,
                    security_id=hedge.original_security_id,
                    error=str(e),
                )
                await telegram.send(
                    CRITICAL,
                    f"HEDGE UNWIND FAILED: Account {account_id}, "
                    f"instrument {hedge.original_security_id}. "
                    f"Manual close required on {hedge.hedge_broker}.",
                )

        # Clear hedge tracking
        self._hedges.pop(account_id, None)
        self._router._broker_state[account_id].hedge_positions = []
```

---

### Rate Limiting

Per-account rate limits are enforced by the `PriorityRateLimiter` defined in the OMS (Component 5). The broker router does not implement its own rate limiter but delegates to the OMS's per-account instances.

| Broker | OPS Limit | Scope | Enforced By |
|--------|-----------|-------|-------------|
| Dhan | 10 per API key | Per-account | `PriorityRateLimiter` in OMS |
| Upstox | 50 per API key | Per-account | Separate limiter created on failover |

```python
# During failover, create a new rate limiter for the Upstox account
async def _initiate_failover(self, account_id, state):
    # ... (failover setup) ...

    # Upstox gets its own rate limiter at 50 OPS
    self._router._failover_rate_limiters[account_id] = PriorityRateLimiter(rate=50)
```

The OMS checks `broker_router.get_state(account_id).active_broker` to decide which rate limiter to use. If the account is on Dhan, it uses the 10 OPS limiter. If on Upstox failover, it uses the 50 OPS limiter.

---

### State Table

| State Item | Storage | Scope | Lifecycle | Writer | Readers |
|-----------|---------|-------|-----------|--------|---------|
| DhanClient instances | In-memory | Per-account | Session | BrokerRouter (startup) | OMS, HedgeManager |
| UpstoxClient instances | In-memory | Per-account | Created on failover, destroyed on failback | BrokerHealthMonitor | OMS (during failover) |
| AccountBrokerState | In-memory + Redis mirror | Per-account | Session | BrokerHealthMonitor | OMS, Risk Monitor, Monitoring |
| Auth tokens (Dhan) | Redis `AUTH:dhan:{account_id}:token` | Per-account | TTL 86400s | AuthManager | DhanClient |
| Auth tokens (Upstox) | Redis `AUTH:upstox:{account_id}:token` | Per-account | TTL 86400s | AuthManager | UpstoxClient |
| SymbolMapper tables | In-memory | Shared (all accounts) | Built once at 08:30, immutable for session | SymbolMapper | BrokerRouter, OMS, HedgeManager |
| Dhan instrument CSV | Downloaded at 08:30 | Shared | Session | SymbolMapper | SymbolMapper |
| Upstox instrument CSV | Downloaded at 08:30 | Shared | Session | SymbolMapper | SymbolMapper |
| HedgePosition records | In-memory | Per-account | Created on failover, cleared on failback | HedgeManager | BrokerHealthMonitor, Monitoring |
| Failover rate limiters | In-memory | Per-account | Created on failover, destroyed on failback | BrokerHealthMonitor | OMS |
| Broker state mirror | Redis `BROKER:{account_id}:state` | Per-account | Updated on every state change | BrokerHealthMonitor | Monitoring, Grafana |
| Symbol mapping stats | Redis `SYMBOLS:build_status` | Shared | Updated at 08:30 | SymbolMapper | Monitoring |

**Per-account vs shared:**

| Category | Per-Account | Shared |
|----------|------------|--------|
| Broker clients | DhanClient, UpstoxClient | -- |
| Auth tokens | Yes (separate credentials) | -- |
| Failover state | Yes (independent failover) | -- |
| WS connections | Yes (one per account) | -- |
| Rate limiters | Yes (10 OPS each) | -- |
| Symbol mapping | -- | Yes (one mapper, all accounts read) |
| Instrument CSVs | -- | Yes (downloaded once) |

---

### Failure Modes

| # | Failure | Detection | Impact | Recovery |
|---|---------|-----------|--------|----------|
| 1 | **Dhan auth failure at startup** | HTTP 401 during `_auth_single_account` | Account cannot trade. Other accounts unaffected. | SUSPEND account. Telegram CRITICAL. Operator provides fresh API key and restarts. |
| 2 | **Dhan token expires mid-session** | HTTP 401 on any REST call during trading hours | All operations for that account fail. Server-side SLs remain active. | `refresh_token()` attempt. If refresh fails: SUSPEND account, Telegram CRITICAL. |
| 3 | **Dhan API timeout (single request)** | `asyncio.TimeoutError` after 10s | That specific operation delayed. Retried automatically. | Exponential backoff retry (0.5s, 1s, 2s). Max 3 retries. |
| 4 | **Dhan API 5xx (server error)** | HTTP 5xx response | That specific operation fails. Retried automatically. | Same retry logic. If 3 retries fail: increment `consecutive_failures`. |
| 5 | **Dhan WS disconnect (one account)** | `ConnectionClosed` exception in WS reader | That account's fill management loops lose real-time updates. | Reconnect with backoff. If disconnect > 30s: trigger FAILED state. |
| 6 | **Dhan WS disconnect (all accounts)** | All `WSHealthMonitor.ws_connected = False` | All fill management loops switch to REST polling. OPS pressure increases. | Individual account reconnects. If all down > 60s: Telegram CRITICAL, suppress new entries. |
| 7 | **Broker maintenance window** | Dhan announces scheduled downtime or returns 503 | All API calls fail during window. | If pre-announced: suppress new entries for the window. If unexpected: treated as consecutive failures, may trigger failover. |
| 8 | **Cross-broker symbol mismatch** | `SymbolMapper.dhan_to_upstox()` returns None during failover hedge | Cannot place hedge on Upstox for that instrument. | Telegram CRITICAL with the specific instrument. Position protected only by Dhan server-side SL. Operator must manually hedge or accept the risk. |
| 9 | **Hedge order rejected on backup broker** | `OrderResponse.status == "REJECTED"` from Upstox | Position on primary broker is unhedged during failover. | Telegram CRITICAL. Try alternative hedge (futures instead of options). If all fail: operator intervention. |
| 10 | **Partial failover (some instruments hedge, others fail)** | Hedge loop tracks success/failure per position | Some positions hedged, others exposed. | Each position is independent. Hedged positions are safe. Unhedged positions rely on primary broker SLs. Log the specific unhedged instruments. |
| 11 | **Upstox auth fails during failover** | Exception in `_initiate_failover` when creating UpstoxClient | Cannot establish backup broker connection. Failover aborted. | Positions protected by Dhan server-side SLs only. Telegram CRITICAL. No automatic retry for Upstox auth. |
| 12 | **Failback causes double position** | Hedge unwind fails, primary broker recovers, orders resume | Account has positions on both brokers simultaneously. | Unwind verification: after failback, query both brokers for positions. If residual on backup: force-close with MARKET IOC. |
| 13 | **Symbol CSV download fails at 08:30** | HTTP error or timeout during `_download_dhan_csv` | System cannot start — Dhan CSV is mandatory for all operations. | Retry 3 times with 10s backoff. If Dhan CSV fails after retries: abort startup. If only Upstox CSV fails: start with degraded failover capability. |
| 14 | **Token stored in Redis but client not initialized** | Client `_token` is None despite Redis having the token | Process restart reads Redis token but does not restore client state. | Full re-auth on startup — never rely on stale Redis tokens. |
| 15 | **Rate limit server-side (HTTP 429)** | Dhan returns HTTP 429 | OPS limit exceeded despite client-side rate limiter. | Back off 1s. Log ERROR. Investigate why client-side limiter allowed it (clock skew, burst). |

---

### Edge Cases

#### 1. Failover triggers during an active fill management loop

Account A's Dhan fails while the OMS is in the middle of managing a fill (waiting for PART_TRADED to complete). The fill management loop is using the Dhan adapter.

**Handling:** The fill management loop detects the broker switch on its next API call (which will fail with the Dhan client's error). It checks `broker_router.get_state(account_id).health`. If FAILOVER_ACTIVE: the loop cancels the pending order on Dhan (best-effort, may fail), and the position is hedged by the HedgeManager. The fill loop does NOT re-place the order on Upstox — that would require re-entering the fill management lifecycle from the OMS, which is not within the fill loop's scope.

#### 2. Two accounts failover simultaneously

Account A and Account B both hit 3 consecutive Dhan failures at the same time (plausible if Dhan has a system-wide outage).

**Handling:** Each account's failover is independent. Both create their own UpstoxClient, authenticate separately, and hedge independently. Their Upstox rate limits are also independent (50 OPS each). No cross-account coordination is needed.

#### 3. Symbol mapping stale after market-hours contract rollover

A new weekly expiry contract is listed after the 08:30 symbol build. A strategy signals a trade on the new contract at 09:30.

**Handling:** The SymbolMapper build runs once at 08:30 and is immutable for the session. NSE lists new weekly options before 08:30, so this case is unlikely. If it occurs: the Instrument Resolver (Component 2) fetches the contract from the Dhan on-demand chain API (which uses Dhan security_id natively). The Upstox mapping will be missing, meaning failover for that specific instrument is degraded. Telegram WARNING issued.

#### 4. Failback attempt while hedge unwind order is in-flight

The primary broker recovers and failback begins. The hedge unwind order on Upstox is placed but not yet filled. Meanwhile, the system switches active broker back to Dhan.

**Handling:** The failback procedure waits for hedge unwind confirmation before switching the active broker. The `unwind_hedges` method awaits each close order's response. If a close order hangs (no response within 30s): force-close with IOC, log WARNING.

#### 5. Dhan returns 200 for order placement but the order is phantom

During degraded conditions, Dhan returns a success response with an `orderId`, but the order never appears in the order book. This is the same edge case documented in OMS Edge Case 10.

**Handling:** The fill management loop's TRANSIT timeout (10s) catches this. The broker router does not add additional detection — it delegates to the OMS's existing handling.

#### 6. Upstox instrument_key format changes

Upstox changes their CSV format or instrument_key structure between sessions.

**Handling:** The SymbolMapper's `_download_upstox_csv` parses column headers dynamically. If expected columns are missing: the build logs an error for Upstox and continues with Dhan-only mode. The system operates normally on the primary broker, with degraded failover. Operator alerted to update the Upstox CSV parser.

#### 7. Account authenticated on Dhan and Upstox simultaneously during failover

During failover, the account has active sessions on both brokers: primary (Dhan) for existing server-side SLs, backup (Upstox) for new hedge positions.

**Handling:** This is the expected state during failover. The DhanClient remains initialized (its SLs are still active on Dhan's servers). The UpstoxClient is created for hedge placement. Both sessions are valid. The `active_broker` field determines which broker receives new orders — Upstox during failover. Dhan SLs continue to protect existing positions independently of the active broker selection.

#### 8. Redis down during auth — tokens cannot be stored

Redis is unavailable at 08:25 when authentication runs. Tokens are obtained from Dhan but cannot be persisted.

**Handling:** Tokens are set on the in-memory DhanClient regardless of Redis. Redis is used for cross-process sharing and monitoring visibility, not as the primary token store. The DhanClient operates normally without Redis. A warning is logged and the OMS enters degraded mode (as described in OMS Component 5). When Redis recovers, the auth manager writes the tokens on the next health check cycle.

#### 9. Failover races with EOD flatten

At 15:20, the EOD flatten sequence begins for all accounts. At 15:21, Account A's Dhan connection fails and failover triggers.

**Handling:** EOD flatten has priority. If failover triggers during EOD flatten, the hedging step is skipped — the positions are being closed anyway. The `HedgeManager.hedge_account` checks with the OMS whether EOD flatten is in progress. If it is: log a warning and skip hedging. The flatten sequence on Dhan will either complete (using the already-placed cancel/exit orders) or fail (in which case the server-side SLs catch it at session close).

#### 10. Upstox HFT endpoint vs standard endpoint selection

Upstox offers `api-hft.upstox.com` for lower-latency order placement. During failover, should the system use the HFT endpoint?

**Handling:** v1 uses the standard `api.upstox.com` endpoint for failover. The HFT endpoint requires separate approval and has different rate limits. Since failover is an emergency path (not the primary execution path), the standard endpoint's latency is acceptable. The `UpstoxClient.BASE_URL` is configurable to switch to HFT if approved.
