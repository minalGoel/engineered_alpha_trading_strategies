## Component 8: Position & PnL Tracker

> **Multi-Account Note:** The Position Tracker is PER-ACCOUNT for position state and SHARED as a process. A single PositionTracker async process manages positions for ALL accounts, but each account's positions are tracked independently with separate DuckDB tables, separate Redis cache keys, and separate reconciliation cycles. Market data (prices for MTM) is shared. Entry prices, lot counts, and PnL differ across accounts because fill prices and allocation sizes diverge.

### Responsibility

- Real-time position state for every account (sole source of truth locally)
- Realized and unrealized PnL per account
- **Sole writer to DuckDB** -- all other components read via Redis cache
- Per-account reconciliation with Dhan positions API
- Greeks computation for option positions (shared market data, per-position results)
- State restoration on startup (per-account broker query + Redis merge)
- Aggregate views across accounts for monitoring and PMS reporting

---

### Position Model

Every position is uniquely identified by `(account_id, security_id, strategy_id)`. Two accounts holding the same instrument for the same strategy are two distinct Position objects with independent entry prices, quantities, and PnL.

```python
from datetime import date
from typing import Literal
import pydantic


class Position(pydantic.BaseModel):
    """
    Single position for one account, one instrument, one strategy.

    Position key: (account_id, security_id, strategy_id).
    Two accounts holding NIFTY 24500 CE for S1 are two Position objects.
    """
    # --- Identity ---
    account_id: str                          # Dhan client ID (e.g., "1000000001")
    strategy_id: str                         # S1..S7
    security_id: str                         # Dhan security ID (from instrument master)
    trading_symbol: str                      # e.g., "NIFTY-24500-CE-2026-04-02"
    underlying: str                          # e.g., "NIFTY50", "BANKNIFTY"
    instrument_type: Literal[
        "CE", "PE", "FUT", "EQ"
    ]

    # --- Position state ---
    direction: Literal["LONG", "SHORT"]
    quantity: int                            # signed: +ve for long, -ve for short
    avg_entry_price: float                   # VWAP of all entry fills
    current_price: float                     # last MTM price (bid/ask/LTP per table)
    unrealized_pnl: float                    # (current_price - avg_entry_price) * quantity
    realized_pnl: float                      # accumulated realized PnL for this position
    entry_ts: int                            # epoch ms of first entry fill
    last_update_ts: int                      # epoch ms of last price/fill update

    # --- Derivative fields (None for EQ) ---
    strike: float | None = None
    expiry: date | None = None
    option_type: Literal["CE", "PE"] | None = None
    lot_size: int | None = None              # exchange lot size (e.g., 65 for NIFTY)

    # --- Greeks (None for non-options) ---
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    iv: float | None = None                  # implied volatility used for Greeks

    # --- Order tracking ---
    order_ids: list[str] = []                # all entry order IDs that built this position
    sl_order_id: str | None = None           # current active SL order ID
    sl_type: Literal["DAY", "GTT"] | None = None

    # --- Flags ---
    is_overnight: bool = False               # True for S2, S6, S7 positions held overnight
    is_orphan: bool = False                  # True if found on broker but unknown locally

    @property
    def position_key(self) -> tuple[str, str, str]:
        return (self.account_id, self.security_id, self.strategy_id)

    @property
    def notional_value(self) -> float:
        return abs(self.quantity) * self.current_price

    @property
    def is_option(self) -> bool:
        return self.instrument_type in ("CE", "PE")

    def update_mtm(self, price: float, ts: int) -> None:
        """Update mark-to-market price and recalculate unrealized PnL."""
        self.current_price = price
        if self.direction == "LONG":
            self.unrealized_pnl = (price - self.avg_entry_price) * abs(self.quantity)
        else:
            self.unrealized_pnl = (self.avg_entry_price - price) * abs(self.quantity)
        self.last_update_ts = ts
```

---

### TradeRecord Model

Every fill generates a TradeRecord written to DuckDB. This is the immutable audit trail of executions per account.

```python
class TradeRecord(pydantic.BaseModel):
    """
    Immutable record of a single trade execution for one account.
    Written to trades_{account_id} table in DuckDB.
    """
    trade_id: str                            # UUID, globally unique
    account_id: str                          # Dhan client ID
    strategy_id: str                         # S1..S7
    signal_id: str                           # originating signal UUID
    order_id: str                            # Dhan order ID
    exchange_order_id: str | None            # exchange order ID (from Dhan)

    security_id: str                         # Dhan security ID
    trading_symbol: str                      # human-readable symbol
    underlying: str
    instrument_type: str                     # CE, PE, FUT, EQ
    strike: float | None = None
    expiry: date | None = None
    option_type: str | None = None

    transaction_type: Literal["BUY", "SELL"]
    trade_type: Literal[
        "ENTRY",                             # opening a new position
        "EXIT",                              # closing an existing position
        "SL_TRIGGERED",                      # stop-loss triggered fill
        "EOD_FLATTEN",                       # end-of-day flatten
        "EMERGENCY_EXIT",                    # global kill / risk halt
        "PARTIAL_ENTRY",                     # partial fill on entry
        "PARTIAL_EXIT",                      # partial fill on exit
    ]

    quantity: int                            # filled quantity (always positive)
    price: float                             # fill price
    notional_value: float                    # quantity * price

    # --- Cost breakdown (INR) ---
    brokerage: float                         # Dhan: 0 for delivery, 20/order for intraday
    stt: float                               # Securities Transaction Tax
    exchange_fees: float                     # NSE transaction charges
    sebi_fee: float                          # SEBI turnover fee
    gst: float                               # GST on brokerage + exchange fees
    stamp_duty: float                        # stamp duty (buy-side only)
    total_cost: float                        # sum of all costs

    # --- Timestamps ---
    exchange_ts: int                         # exchange fill timestamp (epoch ms)
    local_ts: int                            # our receive timestamp (epoch ms)
    trade_date: date                         # IST date of trade

    # --- Context ---
    entry_price: float | None = None         # avg entry for position (for exit trades)
    realized_pnl: float | None = None        # PnL realized by this trade (exits only)
    spot_at_fill: float | None = None        # underlying spot at time of fill
    iv_at_fill: float | None = None          # IV at time of fill (options only)
```

---

### Per-Account DuckDB Tables

DuckDB does not support concurrent writes from multiple processes. The PositionTracker is the sole writer. All tables are partitioned by account_id via separate table names. This avoids row-level filtering overhead and makes per-account queries trivial.

#### Schema Definitions

```sql
-- ============================================================
-- trades_{account_id}: every fill execution
-- ============================================================
CREATE TABLE IF NOT EXISTS trades_{account_id} (
    trade_id            VARCHAR PRIMARY KEY,
    strategy_id         VARCHAR NOT NULL,
    signal_id           VARCHAR NOT NULL,
    order_id            VARCHAR NOT NULL,
    exchange_order_id   VARCHAR,

    security_id         VARCHAR NOT NULL,
    trading_symbol      VARCHAR NOT NULL,
    underlying          VARCHAR NOT NULL,
    instrument_type     VARCHAR NOT NULL,       -- CE, PE, FUT, EQ
    strike              DOUBLE,
    expiry              DATE,
    option_type         VARCHAR,                -- CE, PE, NULL

    transaction_type    VARCHAR NOT NULL,       -- BUY, SELL
    trade_type          VARCHAR NOT NULL,       -- ENTRY, EXIT, SL_TRIGGERED, ...
    quantity            INTEGER NOT NULL,
    price               DOUBLE NOT NULL,
    notional_value      DOUBLE NOT NULL,

    brokerage           DOUBLE NOT NULL DEFAULT 0,
    stt                 DOUBLE NOT NULL DEFAULT 0,
    exchange_fees       DOUBLE NOT NULL DEFAULT 0,
    sebi_fee            DOUBLE NOT NULL DEFAULT 0,
    gst                 DOUBLE NOT NULL DEFAULT 0,
    stamp_duty          DOUBLE NOT NULL DEFAULT 0,
    total_cost          DOUBLE NOT NULL DEFAULT 0,

    exchange_ts         BIGINT NOT NULL,        -- epoch ms
    local_ts            BIGINT NOT NULL,
    trade_date          DATE NOT NULL,

    entry_price         DOUBLE,
    realized_pnl        DOUBLE,
    spot_at_fill        DOUBLE,
    iv_at_fill          DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_trades_{account_id}_strategy
    ON trades_{account_id} (strategy_id, trade_date);

CREATE INDEX IF NOT EXISTS idx_trades_{account_id}_date
    ON trades_{account_id} (trade_date);

CREATE INDEX IF NOT EXISTS idx_trades_{account_id}_security
    ON trades_{account_id} (security_id, trade_date);


-- ============================================================
-- daily_returns_{account_id}: one row per strategy per day
-- Used by Capital Allocator (ERC) via Redis cache
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_returns_{account_id} (
    trade_date          DATE NOT NULL,
    strategy_id         VARCHAR NOT NULL,
    gross_pnl           DOUBLE NOT NULL,        -- before costs
    total_costs         DOUBLE NOT NULL,
    net_pnl             DOUBLE NOT NULL,        -- after costs
    capital_deployed    DOUBLE NOT NULL,         -- capital allocated that day
    net_return          DOUBLE NOT NULL,         -- net_pnl / capital_deployed
    num_trades          INTEGER NOT NULL,
    num_winners         INTEGER NOT NULL,
    num_losers          INTEGER NOT NULL,
    max_intraday_dd     DOUBLE NOT NULL,         -- worst intraday drawdown
    end_of_day_position VARCHAR NOT NULL,        -- FLAT, LONG, SHORT
    PRIMARY KEY (trade_date, strategy_id)
);

CREATE INDEX IF NOT EXISTS idx_daily_returns_{account_id}_strategy
    ON daily_returns_{account_id} (strategy_id, trade_date);


-- ============================================================
-- realized_pnl_{account_id}: per-position realized PnL
-- One row per closed position (entry + exit matched)
-- ============================================================
CREATE TABLE IF NOT EXISTS realized_pnl_{account_id} (
    position_id         VARCHAR PRIMARY KEY,     -- UUID
    strategy_id         VARCHAR NOT NULL,
    security_id         VARCHAR NOT NULL,
    trading_symbol      VARCHAR NOT NULL,
    underlying          VARCHAR NOT NULL,
    instrument_type     VARCHAR NOT NULL,
    direction           VARCHAR NOT NULL,        -- LONG, SHORT

    entry_price         DOUBLE NOT NULL,
    exit_price          DOUBLE NOT NULL,
    quantity            INTEGER NOT NULL,
    gross_pnl           DOUBLE NOT NULL,
    total_costs         DOUBLE NOT NULL,
    net_pnl             DOUBLE NOT NULL,

    entry_ts            BIGINT NOT NULL,
    exit_ts             BIGINT NOT NULL,
    entry_date          DATE NOT NULL,
    exit_date           DATE NOT NULL,
    holding_period_ms   BIGINT NOT NULL,

    exit_reason         VARCHAR NOT NULL,        -- SIGNAL, SL_TRIGGERED, EOD_FLATTEN, EMERGENCY
    entry_order_ids     VARCHAR NOT NULL,         -- JSON array of order IDs
    exit_order_ids      VARCHAR NOT NULL,          -- JSON array of order IDs

    spot_at_entry       DOUBLE,
    spot_at_exit        DOUBLE,
    iv_at_entry         DOUBLE,
    iv_at_exit          DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_realized_pnl_{account_id}_strategy
    ON realized_pnl_{account_id} (strategy_id, exit_date);


-- ============================================================
-- reconciliation_{account_id}: audit trail of every recon event
-- ============================================================
CREATE TABLE IF NOT EXISTS reconciliation_{account_id} (
    recon_id            VARCHAR PRIMARY KEY,
    recon_type          VARCHAR NOT NULL,        -- QUICK, FULL, POST_ORDER, STARTUP
    recon_ts            BIGINT NOT NULL,
    status              VARCHAR NOT NULL,        -- MATCH, DISCREPANCY, FORCE_SYNCED

    -- discrepancy details (NULL if MATCH)
    security_id         VARCHAR,
    local_qty           INTEGER,
    broker_qty          INTEGER,
    local_avg_price     DOUBLE,
    broker_avg_price    DOUBLE,
    action_taken        VARCHAR,                 -- FORCE_SYNC, ORPHAN_CREATED, NONE
    details             VARCHAR                  -- JSON blob with full context
);


-- ============================================================
-- metadata: schema versioning
-- ============================================================
CREATE TABLE IF NOT EXISTS metadata (
    key                 VARCHAR PRIMARY KEY,
    value               VARCHAR NOT NULL
);

-- Initial version
INSERT OR IGNORE INTO metadata (key, value) VALUES ('schema_version', '1');
```

#### Single-Writer Pattern

```
PositionTracker (sole DuckDB writer, single async process)
  ├── Tables per account: trades_{aid}, daily_returns_{aid},
  │                       realized_pnl_{aid}, reconciliation_{aid}
  ├── Writes: on every fill, on recon, daily EOD summary
  ├── Reads: on startup (restore state), on request (daily returns for ERC)
  └── Caches to Redis:
        CACHE:daily_returns:{account_id}:{strategy_id}  -- JSON list, updated daily
        CACHE:strategy_pnl:{account_id}:{strategy_id}   -- current PnL, updated on fill
        CACHE:account_nav:{account_id}                   -- NAV, updated every MTM cycle
        POSITION:strategy:{strategy_id}                  -- aggregated position (all accounts)
        POSITION:account:{account_id}:{strategy_id}      -- per-account position

Other components (read from Redis cache, NEVER DuckDB directly):
  ├── Capital Allocator reads CACHE:daily_returns:{account_id}:{strategy_id}
  ├── Risk Manager reads CACHE:strategy_pnl:{account_id}:{strategy_id}
  ├── Strategy processes read POSITION:strategy:{strategy_id}
  └── Monitoring reads CACHE:account_nav:{account_id}
```

Why this pattern: DuckDB uses an OS-level file lock for writes. If two processes attempt concurrent writes, one blocks until the other finishes -- or worse, the file corrupts on unclean termination. By funneling all writes through the PositionTracker process, we guarantee sequential write access. Reads from other processes are safe (DuckDB supports concurrent reads) but we avoid even that by caching everything in Redis, keeping DuckDB as a durable store only.

---

### Per-Account Reconciliation

Each account is reconciled independently with its own Dhan positions API response. Account A may show a discrepancy while Account B matches perfectly.

#### Reconciliation Schedule

| When | Type | What | OPS Cost |
|------|------|------|----------|
| 09:16 IST | Full | Startup, after market open. Per account. | 1 per account |
| Every 5 min | Quick | security_id + quantity check. Per account. | 1 per account |
| Post-order timer | Targeted | Timer fires `max_patience_s + 5s` after placement. Per account per order. | 1 per account |
| 15:31 IST | Full | EOD reconciliation. Per account. | 1 per account |

With N accounts, the 5-minute quick recon costs N API calls (one per account), staggered 500ms apart to avoid burst. Total OPS impact at 3 accounts: 3 calls every 5 min = negligible.

#### Dhan Positions API

```python
# GET https://api.dhan.co/v2/positions
# Headers: {"access-token": "{account_access_token}"}
#
# Returns all positions for the authenticated account.

class DhanPosition(pydantic.BaseModel):
    """Single position from Dhan positions API response."""
    dhanClientId: str
    tradingSymbol: str
    securityId: str
    positionType: str                        # "LONG", "SHORT", "CLOSED"
    exchangeSegment: str                     # "NSE_FNO", "NSE_EQ"
    productType: str                         # "INTRADAY", "CNC", "MARGIN"
    buyAvg: float
    sellAvg: float
    netQty: int                              # positive = long, negative = short
    buyQty: int
    sellQty: int
    realizedProfit: float
    unrealizedProfit: float
    multiplier: int                          # lot multiplier (1 for equity)
    drvExpiryDate: str | None
    drvOptionType: str | None
    drvStrikePrice: float | None
```

#### Reconciliation Implementation

```python
class Discrepancy(pydantic.BaseModel):
    """Single discrepancy found during reconciliation."""
    account_id: str
    security_id: str
    trading_symbol: str
    local_qty: int
    broker_qty: int
    local_avg_price: float
    broker_avg_price: float
    discrepancy_type: Literal[
        "QTY_MISMATCH",           # quantities differ
        "MISSING_LOCAL",           # broker has it, we don't
        "MISSING_BROKER",          # we have it, broker doesn't
        "PRICE_MISMATCH",          # qty matches but avg price differs
    ]


async def reconcile_account(
    self,
    account_id: str,
    recon_type: Literal["QUICK", "FULL", "POST_ORDER", "STARTUP"],
) -> list[Discrepancy]:
    """
    Reconcile positions for a single account against Dhan API.

    QUICK: compare security_id + netQty only (fast, catches gross errors)
    FULL: compare security_id + netQty + avg_price (catches price drift)

    Returns list of discrepancies. Empty list = all matched.
    """
    account = self._accounts[account_id]
    rate_limiter = self._account_rate_limiters[account_id]

    # 1. Fetch broker positions for THIS account
    await rate_limiter.acquire("POLL")
    broker_positions: list[DhanPosition] = await self._dhan_get_positions(account)

    # 2. Build broker position map: security_id -> DhanPosition
    broker_map: dict[str, DhanPosition] = {}
    for bp in broker_positions:
        if bp.netQty != 0:  # skip closed positions
            broker_map[bp.securityId] = bp

    # 3. Build local position map for this account
    local_map: dict[str, Position] = {}
    for pos in self._positions.values():
        if pos.account_id == account_id:
            local_map[pos.security_id] = pos

    discrepancies: list[Discrepancy] = []

    # 4. Check all local positions exist on broker with correct qty
    for sec_id, local_pos in local_map.items():
        broker_pos = broker_map.pop(sec_id, None)
        if broker_pos is None:
            discrepancies.append(Discrepancy(
                account_id=account_id,
                security_id=sec_id,
                trading_symbol=local_pos.trading_symbol,
                local_qty=local_pos.quantity,
                broker_qty=0,
                local_avg_price=local_pos.avg_entry_price,
                broker_avg_price=0.0,
                discrepancy_type="MISSING_BROKER",
            ))
            continue

        if local_pos.quantity != broker_pos.netQty:
            discrepancies.append(Discrepancy(
                account_id=account_id,
                security_id=sec_id,
                trading_symbol=local_pos.trading_symbol,
                local_qty=local_pos.quantity,
                broker_qty=broker_pos.netQty,
                local_avg_price=local_pos.avg_entry_price,
                broker_avg_price=(broker_pos.buyAvg
                                  if broker_pos.netQty > 0
                                  else broker_pos.sellAvg),
                discrepancy_type="QTY_MISMATCH",
            ))
        elif recon_type in ("FULL", "STARTUP"):
            # FULL recon also checks avg price
            broker_avg = (broker_pos.buyAvg
                         if broker_pos.netQty > 0
                         else broker_pos.sellAvg)
            if abs(local_pos.avg_entry_price - broker_avg) > 0.05:
                discrepancies.append(Discrepancy(
                    account_id=account_id,
                    security_id=sec_id,
                    trading_symbol=local_pos.trading_symbol,
                    local_qty=local_pos.quantity,
                    broker_qty=broker_pos.netQty,
                    local_avg_price=local_pos.avg_entry_price,
                    broker_avg_price=broker_avg,
                    discrepancy_type="PRICE_MISMATCH",
                ))

    # 5. Check remaining broker positions not in local state
    for sec_id, broker_pos in broker_map.items():
        discrepancies.append(Discrepancy(
            account_id=account_id,
            security_id=sec_id,
            trading_symbol=broker_pos.tradingSymbol,
            local_qty=0,
            broker_qty=broker_pos.netQty,
            local_avg_price=0.0,
            broker_avg_price=(broker_pos.buyAvg
                             if broker_pos.netQty > 0
                             else broker_pos.sellAvg),
            discrepancy_type="MISSING_LOCAL",
        ))

    # 6. Log reconciliation result to DuckDB
    recon_id = str(uuid.uuid4())
    status = "MATCH" if not discrepancies else "DISCREPANCY"
    self._write_recon_record(account_id, recon_id, recon_type, status,
                             discrepancies)

    # 7. If discrepancies found, force sync and alert
    if discrepancies:
        logger.critical("position_discrepancy",
                       account_id=account_id,
                       recon_type=recon_type,
                       discrepancies=[d.model_dump() for d in discrepancies])

        await telegram.send(CRITICAL,
            f"POSITION DISCREPANCY: Account {account_id}\n"
            + "\n".join(
                f"  {d.trading_symbol}: local={d.local_qty} broker={d.broker_qty} "
                f"({d.discrepancy_type})"
                for d in discrepancies
            ))

        await self.force_sync_from_broker(account_id)

    return discrepancies


async def force_sync_from_broker(self, account_id: str) -> None:
    """
    Force-sync local position state to match broker for one account.

    Broker is ALWAYS source of truth. This overwrites local state.

    Steps:
    1. Fetch broker positions for this account
    2. For each broker position:
       a. If local position exists: update qty and avg price
       b. If no local position: create ORPHAN position
    3. For each local position not on broker:
       a. Mark as closed (qty went to zero via unknown fill)
    4. Update Redis cache
    """
    account = self._accounts[account_id]
    rate_limiter = self._account_rate_limiters[account_id]

    await rate_limiter.acquire("POLL")
    broker_positions = await self._dhan_get_positions(account)

    broker_map: dict[str, DhanPosition] = {
        bp.securityId: bp
        for bp in broker_positions
        if bp.netQty != 0
    }

    # --- Sync existing + create orphans ---
    synced_security_ids: set[str] = set()

    for sec_id, broker_pos in broker_map.items():
        synced_security_ids.add(sec_id)

        # Find local position for this account + security
        local_key = None
        local_pos = None
        for key, pos in self._positions.items():
            if pos.account_id == account_id and pos.security_id == sec_id:
                local_key = key
                local_pos = pos
                break

        broker_avg = (broker_pos.buyAvg
                     if broker_pos.netQty > 0
                     else broker_pos.sellAvg)

        if local_pos is not None:
            # Update local to match broker
            local_pos.quantity = broker_pos.netQty
            local_pos.direction = "LONG" if broker_pos.netQty > 0 else "SHORT"
            local_pos.avg_entry_price = broker_avg
            local_pos.last_update_ts = now_ms()
            logger.warning("force_sync_updated",
                          account_id=account_id,
                          security_id=sec_id,
                          new_qty=broker_pos.netQty)
        else:
            # Broker has position we don't know about -> ORPHAN
            orphan = Position(
                account_id=account_id,
                strategy_id="ORPHAN",
                security_id=sec_id,
                trading_symbol=broker_pos.tradingSymbol,
                underlying=self._infer_underlying(broker_pos),
                instrument_type=self._infer_instrument_type(broker_pos),
                direction="LONG" if broker_pos.netQty > 0 else "SHORT",
                quantity=broker_pos.netQty,
                avg_entry_price=broker_avg,
                current_price=broker_avg,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_ts=now_ms(),
                last_update_ts=now_ms(),
                is_orphan=True,
            )
            key = orphan.position_key
            self._positions[key] = orphan

            logger.critical("orphan_position_created",
                          account_id=account_id,
                          security_id=sec_id,
                          trading_symbol=broker_pos.tradingSymbol,
                          qty=broker_pos.netQty)

            await telegram.send(CRITICAL,
                f"ORPHAN POSITION: Account {account_id} "
                f"{broker_pos.tradingSymbol} qty={broker_pos.netQty}. "
                f"Unknown to local state. Manual review required.")

    # --- Remove local positions not on broker ---
    keys_to_remove = []
    for key, pos in self._positions.items():
        if (pos.account_id == account_id
                and pos.security_id not in synced_security_ids):
            logger.warning("force_sync_removed_local",
                          account_id=account_id,
                          security_id=pos.security_id,
                          prev_qty=pos.quantity)
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del self._positions[key]

    # --- Update Redis cache ---
    await self._publish_all_positions(account_id)
```

---

### Per-Account Mark-to-Market

Market data is shared: all accounts see the same bid, ask, and LTP for every instrument. However, MTM PnL differs across accounts because entry prices differ (different fill times, different partial fills).

#### MTM Price Selection

| Instrument Type | Long Position Mark | Short Position Mark | Source | Rationale |
|----------------|-------------------|--------------------|---------|----|
| Options (CE/PE) | Best bid | Best ask | Option chain snapshot | Liquidation mark: what you would receive if closing now |
| Futures | LTP | LTP | `LASTTICK:{symbol}` | Futures have tight spreads; LTP is representative |
| Equity | LTP | LTP | `LASTTICK:{symbol}` | Same reasoning as futures |

**Two PnL views:**

| View | Mark Used | Purpose |
|------|-----------|---------|
| Risk PnL | Liquidation mark (table above) | Conservative. Used by Risk Manager for drawdown, kill conditions. |
| Display PnL | Mid-price: `(bid + ask) / 2` | Less noisy. Used in Telegram summaries and Grafana dashboard. |

#### MTM Cycle

The MTM cycle runs every 5 seconds. It iterates over all accounts' positions using the same price snapshot:

```python
async def mark_to_market(self) -> None:
    """
    Update all positions across all accounts with current prices.
    Runs every 5 seconds during market hours.

    Market data is fetched ONCE (shared), then applied to each
    account's positions independently.
    """
    # 1. Get current prices for all instruments with open positions
    instruments = set()
    for pos in self._positions.values():
        instruments.add(pos.security_id)

    price_snapshot: dict[str, MTMPrice] = {}
    for sec_id in instruments:
        tick = await self._redis.get(f"LASTTICK:{self._sec_to_symbol[sec_id]}")
        if tick is None:
            continue
        tick_data = orjson.loads(tick)
        price_snapshot[sec_id] = MTMPrice(
            ltp=tick_data["ltp"],
            bid=tick_data["bid"],
            ask=tick_data["ask"],
            ts=tick_data["exchange_ts"],
        )

    # 2. Apply prices to each position (per-account, independent PnL)
    ts = now_ms()
    for pos in self._positions.values():
        snap = price_snapshot.get(pos.security_id)
        if snap is None:
            continue

        # Select mark based on instrument type and direction
        if pos.is_option:
            if pos.direction == "LONG":
                risk_mark = snap.bid      # liquidation: sell at bid
                display_mark = (snap.bid + snap.ask) / 2
            else:
                risk_mark = snap.ask      # liquidation: buy back at ask
                display_mark = (snap.bid + snap.ask) / 2
        else:
            risk_mark = snap.ltp
            display_mark = snap.ltp

        pos.update_mtm(risk_mark, ts)

    # 3. Update Redis cache per account
    for account_id in self._account_ids:
        account_positions = [
            p for p in self._positions.values()
            if p.account_id == account_id
        ]

        # Per-account NAV
        total_unrealized = sum(p.unrealized_pnl for p in account_positions)
        total_realized = sum(p.realized_pnl for p in account_positions)
        nav = self._account_capitals[account_id] + total_unrealized + total_realized

        await self._redis.set(
            f"CACHE:account_nav:{account_id}",
            orjson.dumps({"nav": nav, "unrealized": total_unrealized,
                         "realized": total_realized, "ts": ts}),
        )

        # Per-account, per-strategy PnL
        strategy_pnl: dict[str, float] = {}
        for p in account_positions:
            strategy_pnl[p.strategy_id] = (
                strategy_pnl.get(p.strategy_id, 0.0) + p.unrealized_pnl
            )
        for sid, pnl in strategy_pnl.items():
            await self._redis.set(
                f"CACHE:strategy_pnl:{account_id}:{sid}",
                orjson.dumps({"unrealized_pnl": pnl, "ts": ts}),
            )

    # 4. Update Prometheus gauges
    for pos in self._positions.values():
        unrealized_pnl_inr.labels(
            strategy=pos.strategy_id,
            account=pos.account_id,
        ).set(pos.unrealized_pnl)

    # 5. Aggregate position for strategy processes (cross-account)
    for sid in self._strategy_ids:
        agg_qty = sum(
            p.quantity for p in self._positions.values()
            if p.strategy_id == sid
        )
        direction = "FLAT" if agg_qty == 0 else ("LONG" if agg_qty > 0 else "SHORT")
        await self._redis.set(
            f"POSITION:strategy:{sid}",
            orjson.dumps({"direction": direction, "quantity": agg_qty,
                         "ts": ts}),
        )


class MTMPrice(pydantic.BaseModel):
    ltp: float
    bid: float
    ask: float
    ts: int
```

---

### Multi-Day Position Restoration

On startup (or after crash recovery), positions must be restored for each account independently. The broker is authoritative for what positions exist; Redis saved state provides strategy attribution.

```python
async def restore_positions(self) -> None:
    """
    Startup position recovery for all accounts.

    For each account:
    1. Query Dhan positions API -> broker truth
    2. Read Redis saved state -> strategy attribution
    3. Merge: broker authoritative for qty/price
    4. Unknown positions -> ORPHAN

    Called once during system startup, before strategies begin.
    """
    for account_id in self._account_ids:
        await self._restore_account_positions(account_id)

    logger.info("position_restore_complete",
               total_positions=len(self._positions),
               accounts=len(self._account_ids))


async def _restore_account_positions(self, account_id: str) -> None:
    """Restore positions for a single account."""
    account = self._accounts[account_id]
    rate_limiter = self._account_rate_limiters[account_id]

    # 1. Fetch broker positions
    await rate_limiter.acquire("POLL")
    broker_positions = await self._dhan_get_positions(account)
    broker_map: dict[str, DhanPosition] = {
        bp.securityId: bp
        for bp in broker_positions
        if bp.netQty != 0
    }

    # 2. Read Redis saved state for this account
    saved_state: dict[str, dict] = {}
    cursor = 0
    while True:
        cursor, keys = await self._redis.scan(
            cursor,
            match=f"POSITION:saved:{account_id}:*",
            count=100,
        )
        for key in keys:
            data = await self._redis.get(key)
            if data:
                parsed = orjson.loads(data)
                saved_state[parsed["security_id"]] = parsed
        if cursor == 0:
            break

    # 3. Merge: broker positions + Redis strategy attribution
    for sec_id, broker_pos in broker_map.items():
        saved = saved_state.get(sec_id)
        broker_avg = (broker_pos.buyAvg
                     if broker_pos.netQty > 0
                     else broker_pos.sellAvg)

        if saved is not None:
            # Known position: use broker qty/price, Redis strategy info
            pos = Position(
                account_id=account_id,
                strategy_id=saved["strategy_id"],
                security_id=sec_id,
                trading_symbol=broker_pos.tradingSymbol,
                underlying=saved.get("underlying", self._infer_underlying(broker_pos)),
                instrument_type=self._infer_instrument_type(broker_pos),
                direction="LONG" if broker_pos.netQty > 0 else "SHORT",
                quantity=broker_pos.netQty,
                avg_entry_price=broker_avg,
                current_price=broker_avg,
                unrealized_pnl=0.0,
                realized_pnl=saved.get("realized_pnl", 0.0),
                entry_ts=saved.get("entry_ts", now_ms()),
                last_update_ts=now_ms(),
                strike=saved.get("strike"),
                expiry=saved.get("expiry"),
                option_type=saved.get("option_type"),
                lot_size=saved.get("lot_size"),
                order_ids=saved.get("order_ids", []),
                sl_order_id=saved.get("sl_order_id"),
                sl_type=saved.get("sl_type"),
                is_overnight=saved.get("is_overnight", False),
            )
            logger.info("position_restored",
                       account_id=account_id,
                       strategy_id=pos.strategy_id,
                       security_id=sec_id,
                       qty=pos.quantity)

        else:
            # Unknown position on broker -> ORPHAN
            pos = Position(
                account_id=account_id,
                strategy_id="ORPHAN",
                security_id=sec_id,
                trading_symbol=broker_pos.tradingSymbol,
                underlying=self._infer_underlying(broker_pos),
                instrument_type=self._infer_instrument_type(broker_pos),
                direction="LONG" if broker_pos.netQty > 0 else "SHORT",
                quantity=broker_pos.netQty,
                avg_entry_price=broker_avg,
                current_price=broker_avg,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_ts=now_ms(),
                last_update_ts=now_ms(),
                is_orphan=True,
            )
            logger.critical("orphan_on_startup",
                          account_id=account_id,
                          security_id=sec_id,
                          trading_symbol=broker_pos.tradingSymbol,
                          qty=broker_pos.netQty)

            await telegram.send(CRITICAL,
                f"ORPHAN ON STARTUP: Account {account_id} "
                f"{broker_pos.tradingSymbol} qty={broker_pos.netQty}. "
                f"Not in saved state. Manual review required.")

        self._positions[pos.position_key] = pos

    # 4. Check for saved positions NOT on broker (closed overnight by broker)
    for sec_id, saved in saved_state.items():
        if sec_id not in broker_map:
            logger.warning("saved_position_not_on_broker",
                          account_id=account_id,
                          security_id=sec_id,
                          strategy_id=saved["strategy_id"],
                          saved_qty=saved.get("quantity", 0))
            # Position was closed by broker (auto-square, SL triggered overnight)
            # Do NOT recreate it. Log for audit.

    # 5. Publish restored positions to Redis
    await self._publish_all_positions(account_id)


async def save_positions(self) -> None:
    """
    Save all position state to Redis for crash recovery.
    Called during graceful shutdown (Phase 4) and periodically every 60s.
    """
    for pos in self._positions.values():
        key = f"POSITION:saved:{pos.account_id}:{pos.security_id}"
        await self._redis.set(
            key,
            orjson.dumps(pos.model_dump(mode="json")),
            ex=86400 * 3,  # 3-day TTL; stale state is worse than no state
        )
```

---

### Aggregate Views (Monitoring)

Aggregate views combine data across accounts for portfolio-level monitoring. These are computed in the MTM cycle and published to Redis for Grafana dashboards.

```python
async def compute_aggregate_views(self) -> AggregateView:
    """
    Compute portfolio-level aggregates across all accounts.
    Called after every MTM cycle.
    """
    account_navs: dict[str, float] = {}
    total_unrealized = 0.0
    total_realized = 0.0
    total_exposure = 0.0
    strategy_pnl: dict[str, float] = {}  # strategy -> sum across accounts

    for account_id in self._account_ids:
        acct_positions = [
            p for p in self._positions.values()
            if p.account_id == account_id
        ]
        acct_unrealized = sum(p.unrealized_pnl for p in acct_positions)
        acct_realized = sum(p.realized_pnl for p in acct_positions)
        nav = self._account_capitals[account_id] + acct_unrealized + acct_realized
        account_navs[account_id] = nav
        total_unrealized += acct_unrealized
        total_realized += acct_realized
        total_exposure += sum(p.notional_value for p in acct_positions)

        for p in acct_positions:
            strategy_pnl[p.strategy_id] = (
                strategy_pnl.get(p.strategy_id, 0.0)
                + p.unrealized_pnl + p.realized_pnl
            )

    total_aum = sum(account_navs.values())

    view = AggregateView(
        total_aum=total_aum,
        total_unrealized_pnl=total_unrealized,
        total_realized_pnl=total_realized,
        total_exposure=total_exposure,
        account_navs=account_navs,
        strategy_pnl=strategy_pnl,
        ts=now_ms(),
    )

    # Publish to Redis for Grafana
    await self._redis.set("CACHE:aggregate_view",
                          orjson.dumps(view.model_dump()))

    # Update Prometheus
    portfolio_drawdown_pct.set(self._compute_portfolio_drawdown(total_aum))
    for sid, pnl in strategy_pnl.items():
        realized_pnl_inr.labels(strategy=sid).set(pnl)

    return view


class AggregateView(pydantic.BaseModel):
    total_aum: float                         # sum of all account NAVs
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_exposure: float                    # sum of notional across all positions
    account_navs: dict[str, float]           # account_id -> NAV
    strategy_pnl: dict[str, float]           # strategy_id -> total PnL across accounts
    ts: int
```

**Per-account NAV for PMS reporting:** Each account's NAV is tracked independently in `CACHE:account_nav:{account_id}`. This enables per-client reporting where each account owner sees only their own performance, while the system operator sees the aggregate.

---

### Greeks Computation

Greeks are computed per position using Black-76 (for exchange-traded options on futures-settled underlyings). Market data inputs are shared; output Greeks are per-position because strike, expiry, and quantity differ.

```python
async def compute_greeks(self) -> None:
    """
    Compute Greeks for all option positions across all accounts.
    Runs every 30 seconds during market hours.

    Inputs (shared across accounts):
    - Spot price: LASTTICK for underlying
    - IV: from option chain snapshot or BSM inversion
    - Risk-free rate: from config (6.5%, updated quarterly)

    Outputs (per position):
    - delta, gamma, theta, vega stored on Position object
    """
    spot_cache: dict[str, float] = {}  # underlying -> spot
    iv_cache: dict[str, float] = {}    # security_id -> IV

    for pos in self._positions.values():
        if not pos.is_option:
            continue

        # Get spot (cached per underlying, shared across accounts)
        if pos.underlying not in spot_cache:
            tick = await self._redis.get(f"LASTTICK:{pos.underlying}")
            if tick is None:
                continue
            spot_cache[pos.underlying] = orjson.loads(tick)["ltp"]
        spot = spot_cache[pos.underlying]

        # Get IV (cached per security_id, shared across accounts)
        if pos.security_id not in iv_cache:
            iv = await self._get_iv(pos.security_id, spot,
                                     pos.strike, pos.expiry,
                                     pos.option_type)
            if iv is None:
                continue
            iv_cache[pos.security_id] = iv
        iv = iv_cache[pos.security_id]

        # Time to expiry in years
        tte = self._time_to_expiry_years(pos.expiry)
        if tte <= 0:
            # Expired option — set Greeks to terminal values
            pos.delta = 1.0 if pos.option_type == "CE" and spot > pos.strike else 0.0
            pos.gamma = 0.0
            pos.theta = 0.0
            pos.vega = 0.0
            pos.iv = 0.0
            continue

        r = self._config.risk_free_rate  # 0.065

        # Black-76 Greeks
        F = spot * math.exp(r * tte)  # forward price
        d1 = (math.log(F / pos.strike) + 0.5 * iv**2 * tte) / (iv * math.sqrt(tte))
        d2 = d1 - iv * math.sqrt(tte)

        discount = math.exp(-r * tte)
        n_d1 = norm.cdf(d1)
        n_d2 = norm.cdf(d2)
        n_prime_d1 = norm.pdf(d1)

        if pos.option_type == "CE":
            pos.delta = discount * n_d1
            pos.theta = (
                -(spot * n_prime_d1 * iv) / (2 * math.sqrt(tte))
                - r * pos.strike * discount * n_d2
            ) / 365  # per-day theta
        else:  # PE
            pos.delta = discount * (n_d1 - 1)
            pos.theta = (
                -(spot * n_prime_d1 * iv) / (2 * math.sqrt(tte))
                + r * pos.strike * discount * (1 - n_d2)
            ) / 365

        pos.gamma = (discount * n_prime_d1) / (spot * iv * math.sqrt(tte))
        pos.vega = spot * discount * n_prime_d1 * math.sqrt(tte) / 100  # per 1% IV
        pos.iv = iv

        # Scale by quantity for portfolio Greeks
        # (Individual Greeks are per-unit; portfolio aggregation uses qty)

    # Portfolio delta exposure check
    total_delta_lots = sum(
        (p.delta or 0) * p.quantity / (p.lot_size or 1)
        for p in self._positions.values()
        if p.is_option
    )
    delta_exposure.set(total_delta_lots)

    if abs(total_delta_lots) > 50:
        logger.warning("delta_exposure_high",
                      delta_lots=total_delta_lots)
        await telegram.send(WARNING,
            f"DELTA EXPOSURE: Portfolio net delta = {total_delta_lots:.1f} "
            f"NIFTY lot equivalents (threshold: +/-50)")
```

---

### PositionTracker Class

```python
class PositionTracker:
    """
    Central position management for all accounts.

    Single async process. Sole writer to DuckDB. Manages per-account
    positions, reconciliation, MTM, Greeks, and aggregate views.

    Position key: (account_id, security_id, strategy_id)
    """

    def __init__(
        self,
        accounts: list["Account"],
        redis: aioredis.Redis,
        duckdb_path: str,
        config: "TradingConfig",
    ):
        self._accounts: dict[str, "Account"] = {
            a.account_id: a for a in accounts
        }
        self._account_ids: list[str] = [a.account_id for a in accounts]
        self._redis = redis
        self._db = duckdb.connect(duckdb_path)
        self._config = config

        # Positions: (account_id, security_id, strategy_id) -> Position
        self._positions: dict[tuple[str, str, str], Position] = {}

        # Per-account rate limiters (shared with OMS via reference)
        self._account_rate_limiters: dict[str, "PriorityRateLimiter"] = {}

        # Per-account capital (loaded from config, updated daily)
        self._account_capitals: dict[str, float] = {
            a.account_id: a.initial_capital for a in accounts
        }

        # Symbol mapping caches
        self._sec_to_symbol: dict[str, str] = {}

        # Strategy IDs (derived from config)
        self._strategy_ids: list[str] = config.strategy_ids

        # Initialize DuckDB tables for all accounts
        self._init_tables()

    def _init_tables(self) -> None:
        """Create DuckDB tables for each account if they don't exist."""
        for account_id in self._account_ids:
            self._db.execute(f"""
                CREATE TABLE IF NOT EXISTS trades_{account_id} (
                    trade_id VARCHAR PRIMARY KEY,
                    strategy_id VARCHAR NOT NULL,
                    signal_id VARCHAR NOT NULL,
                    order_id VARCHAR NOT NULL,
                    exchange_order_id VARCHAR,
                    security_id VARCHAR NOT NULL,
                    trading_symbol VARCHAR NOT NULL,
                    underlying VARCHAR NOT NULL,
                    instrument_type VARCHAR NOT NULL,
                    strike DOUBLE,
                    expiry DATE,
                    option_type VARCHAR,
                    transaction_type VARCHAR NOT NULL,
                    trade_type VARCHAR NOT NULL,
                    quantity INTEGER NOT NULL,
                    price DOUBLE NOT NULL,
                    notional_value DOUBLE NOT NULL,
                    brokerage DOUBLE NOT NULL DEFAULT 0,
                    stt DOUBLE NOT NULL DEFAULT 0,
                    exchange_fees DOUBLE NOT NULL DEFAULT 0,
                    sebi_fee DOUBLE NOT NULL DEFAULT 0,
                    gst DOUBLE NOT NULL DEFAULT 0,
                    stamp_duty DOUBLE NOT NULL DEFAULT 0,
                    total_cost DOUBLE NOT NULL DEFAULT 0,
                    exchange_ts BIGINT NOT NULL,
                    local_ts BIGINT NOT NULL,
                    trade_date DATE NOT NULL,
                    entry_price DOUBLE,
                    realized_pnl DOUBLE,
                    spot_at_fill DOUBLE,
                    iv_at_fill DOUBLE
                )
            """)
            self._db.execute(f"""
                CREATE TABLE IF NOT EXISTS daily_returns_{account_id} (
                    trade_date DATE NOT NULL,
                    strategy_id VARCHAR NOT NULL,
                    gross_pnl DOUBLE NOT NULL,
                    total_costs DOUBLE NOT NULL,
                    net_pnl DOUBLE NOT NULL,
                    capital_deployed DOUBLE NOT NULL,
                    net_return DOUBLE NOT NULL,
                    num_trades INTEGER NOT NULL,
                    num_winners INTEGER NOT NULL,
                    num_losers INTEGER NOT NULL,
                    max_intraday_dd DOUBLE NOT NULL,
                    end_of_day_position VARCHAR NOT NULL,
                    PRIMARY KEY (trade_date, strategy_id)
                )
            """)
            self._db.execute(f"""
                CREATE TABLE IF NOT EXISTS realized_pnl_{account_id} (
                    position_id VARCHAR PRIMARY KEY,
                    strategy_id VARCHAR NOT NULL,
                    security_id VARCHAR NOT NULL,
                    trading_symbol VARCHAR NOT NULL,
                    underlying VARCHAR NOT NULL,
                    instrument_type VARCHAR NOT NULL,
                    direction VARCHAR NOT NULL,
                    entry_price DOUBLE NOT NULL,
                    exit_price DOUBLE NOT NULL,
                    quantity INTEGER NOT NULL,
                    gross_pnl DOUBLE NOT NULL,
                    total_costs DOUBLE NOT NULL,
                    net_pnl DOUBLE NOT NULL,
                    entry_ts BIGINT NOT NULL,
                    exit_ts BIGINT NOT NULL,
                    entry_date DATE NOT NULL,
                    exit_date DATE NOT NULL,
                    holding_period_ms BIGINT NOT NULL,
                    exit_reason VARCHAR NOT NULL,
                    entry_order_ids VARCHAR NOT NULL,
                    exit_order_ids VARCHAR NOT NULL,
                    spot_at_entry DOUBLE,
                    spot_at_exit DOUBLE,
                    iv_at_entry DOUBLE,
                    iv_at_exit DOUBLE
                )
            """)
            self._db.execute(f"""
                CREATE TABLE IF NOT EXISTS reconciliation_{account_id} (
                    recon_id VARCHAR PRIMARY KEY,
                    recon_type VARCHAR NOT NULL,
                    recon_ts BIGINT NOT NULL,
                    status VARCHAR NOT NULL,
                    security_id VARCHAR,
                    local_qty INTEGER,
                    broker_qty INTEGER,
                    local_avg_price DOUBLE,
                    broker_avg_price DOUBLE,
                    action_taken VARCHAR,
                    details VARCHAR
                )
            """)

    # --- Fill handling ---

    async def on_fill(self, fill: "OrderResult") -> None:
        """
        Process a completed fill from the OMS.

        Creates or updates position for the specific account.
        Writes TradeRecord to DuckDB.
        Updates Redis cache.
        """
        account_id = fill.account_id
        key = (account_id, fill.security_id, fill.strategy_id)

        if fill.trade_type in ("ENTRY", "PARTIAL_ENTRY"):
            if key in self._positions:
                # Add to existing position (scale-in)
                pos = self._positions[key]
                old_notional = pos.avg_entry_price * abs(pos.quantity)
                new_notional = fill.avg_fill_price * fill.filled_qty
                total_qty = abs(pos.quantity) + fill.filled_qty
                pos.avg_entry_price = (old_notional + new_notional) / total_qty
                pos.quantity = total_qty if pos.direction == "LONG" else -total_qty
                pos.order_ids.append(fill.order_id)
                pos.last_update_ts = now_ms()
            else:
                # New position
                pos = Position(
                    account_id=account_id,
                    strategy_id=fill.strategy_id,
                    security_id=fill.security_id,
                    trading_symbol=fill.trading_symbol,
                    underlying=fill.underlying,
                    instrument_type=fill.instrument_type,
                    direction=fill.direction,
                    quantity=(fill.filled_qty if fill.direction == "LONG"
                             else -fill.filled_qty),
                    avg_entry_price=fill.avg_fill_price,
                    current_price=fill.avg_fill_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    entry_ts=now_ms(),
                    last_update_ts=now_ms(),
                    strike=fill.strike,
                    expiry=fill.expiry,
                    option_type=fill.option_type,
                    lot_size=fill.lot_size,
                    order_ids=[fill.order_id],
                    sl_order_id=fill.sl_order_id,
                    sl_type=fill.sl_type,
                    is_overnight=fill.strategy_id in ("S2", "S6", "S7"),
                )
                self._positions[key] = pos

        elif fill.trade_type in ("EXIT", "SL_TRIGGERED", "EOD_FLATTEN",
                                  "EMERGENCY_EXIT", "PARTIAL_EXIT"):
            pos = self._positions.get(key)
            if pos is not None:
                # Calculate realized PnL
                if pos.direction == "LONG":
                    rpnl = (fill.avg_fill_price - pos.avg_entry_price) * fill.filled_qty
                else:
                    rpnl = (pos.avg_entry_price - fill.avg_fill_price) * fill.filled_qty

                remaining = abs(pos.quantity) - fill.filled_qty
                if remaining <= 0:
                    # Position fully closed
                    pos.realized_pnl += rpnl
                    self._write_realized_pnl(pos, fill, rpnl)
                    del self._positions[key]
                else:
                    # Partial exit
                    pos.quantity = remaining if pos.direction == "LONG" else -remaining
                    pos.realized_pnl += rpnl
                    pos.last_update_ts = now_ms()

        # Write trade record to DuckDB
        trade = self._build_trade_record(fill)
        self.write_trade(trade)

        # Update Redis caches
        await self._publish_position_update(account_id, fill.strategy_id)

        # Update Prometheus
        open_positions.labels(
            strategy=fill.strategy_id,
            account=account_id,
        ).set(self._count_positions(account_id, fill.strategy_id))

    async def on_partial_fill(self, state: "OrderState") -> None:
        """Handle partial fill notification. SL qty sync handled by OMS/SLManager."""
        # Position update deferred until manage_order loop completes.
        # This method updates interim state for MTM accuracy.
        key = (state.account_id, state.security_id, state.strategy_id)
        pos = self._positions.get(key)
        if pos is not None:
            pos.last_update_ts = now_ms()

    def get_strategy_position(
        self,
        strategy_id: str,
        account_id: str | None = None,
    ) -> str:
        """
        Get position direction for a strategy.

        If account_id is None, returns aggregate across all accounts.
        """
        positions = [
            p for p in self._positions.values()
            if p.strategy_id == strategy_id
            and (account_id is None or p.account_id == account_id)
        ]
        total_qty = sum(p.quantity for p in positions)
        if total_qty == 0:
            return "FLAT"
        return "LONG" if total_qty > 0 else "SHORT"

    def get_all_open(self, account_id: str | None = None) -> list[Position]:
        """Get all open positions, optionally filtered by account."""
        if account_id is None:
            return list(self._positions.values())
        return [p for p in self._positions.values()
                if p.account_id == account_id]

    def get_overnight_positions(self, account_id: str) -> list[Position]:
        """Get overnight positions for a specific account."""
        return [
            p for p in self._positions.values()
            if p.account_id == account_id and p.is_overnight
        ]

    # --- DuckDB write methods (sole writer) ---

    def write_trade(self, trade: TradeRecord) -> None:
        """Write a single trade record to the account's trades table."""
        table = f"trades_{trade.account_id}"
        self._db.execute(f"""
            INSERT INTO {table} VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            trade.trade_id, trade.strategy_id, trade.signal_id,
            trade.order_id, trade.exchange_order_id,
            trade.security_id, trade.trading_symbol, trade.underlying,
            trade.instrument_type, trade.strike, trade.expiry,
            trade.option_type, trade.transaction_type, trade.trade_type,
            trade.quantity, trade.price, trade.notional_value,
            trade.brokerage, trade.stt, trade.exchange_fees,
            trade.sebi_fee, trade.gst, trade.stamp_duty, trade.total_cost,
            trade.exchange_ts, trade.local_ts, trade.trade_date,
            trade.entry_price, trade.realized_pnl, trade.spot_at_fill,
            trade.iv_at_fill,
        ])

    def write_daily_pnl(
        self,
        account_id: str,
        strategy_id: str,
        d: date,
        summary: "DailySummary",
    ) -> None:
        """Write daily PnL summary. Called at EOD for each strategy per account."""
        table = f"daily_returns_{account_id}"
        self._db.execute(f"""
            INSERT OR REPLACE INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            d, strategy_id, summary.gross_pnl, summary.total_costs,
            summary.net_pnl, summary.capital_deployed, summary.net_return,
            summary.num_trades, summary.num_winners, summary.num_losers,
            summary.max_intraday_dd, summary.end_of_day_position,
        ])

        # Update Redis cache
        asyncio.create_task(self._update_daily_returns_cache(
            account_id, strategy_id))

    def get_daily_returns(
        self,
        account_id: str,
        strategy_id: str,
        lookback: int,
    ) -> list[float]:
        """
        Get recent daily net returns for a strategy in an account.
        Used by Capital Allocator for ERC computation.
        """
        table = f"daily_returns_{account_id}"
        result = self._db.execute(f"""
            SELECT net_return FROM {table}
            WHERE strategy_id = ?
            ORDER BY trade_date DESC
            LIMIT ?
        """, [strategy_id, lookback]).fetchall()
        return [row[0] for row in reversed(result)]

    async def _update_daily_returns_cache(
        self,
        account_id: str,
        strategy_id: str,
    ) -> None:
        """Push daily returns to Redis cache for Capital Allocator."""
        returns = self.get_daily_returns(account_id, strategy_id, lookback=252)
        await self._redis.set(
            f"CACHE:daily_returns:{account_id}:{strategy_id}",
            orjson.dumps(returns),
            ex=86400,  # 1-day TTL
        )

        # Also write an aggregated RETURNS:{sid}:daily Redis LIST (shared,
        # not per-account) by summing weighted returns across all accounts.
        # This is the key the Capital Allocator's ERC computation reads.
        await self._update_aggregated_returns(strategy_id)

    async def _publish_position_update(
        self,
        account_id: str,
        strategy_id: str,
    ) -> None:
        """Publish position state to Redis for other components."""
        # Per-account position
        positions = [
            p for p in self._positions.values()
            if p.account_id == account_id and p.strategy_id == strategy_id
        ]
        total_qty = sum(p.quantity for p in positions)
        direction = "FLAT" if total_qty == 0 else ("LONG" if total_qty > 0 else "SHORT")

        await self._redis.set(
            f"POSITION:account:{account_id}:{strategy_id}",
            orjson.dumps({
                "direction": direction,
                "quantity": total_qty,
                "unrealized_pnl": sum(p.unrealized_pnl for p in positions),
                "ts": now_ms(),
            }),
        )

        # Aggregate position (cross-account, for strategy processes)
        all_positions = [
            p for p in self._positions.values()
            if p.strategy_id == strategy_id
        ]
        agg_qty = sum(p.quantity for p in all_positions)
        agg_dir = "FLAT" if agg_qty == 0 else ("LONG" if agg_qty > 0 else "SHORT")

        await self._redis.set(
            f"POSITION:strategy:{strategy_id}",
            orjson.dumps({
                "direction": agg_dir,
                "quantity": agg_qty,
                "ts": now_ms(),
            }),
        )

    async def _publish_all_positions(self, account_id: str) -> None:
        """Publish all positions for an account to Redis."""
        strategy_ids = set(
            p.strategy_id for p in self._positions.values()
            if p.account_id == account_id
        )
        for sid in strategy_ids:
            await self._publish_position_update(account_id, sid)

    # --- Async event loop ---

    async def run(self) -> None:
        """
        Main event loop. Runs concurrently:
        1. MTM every 5s
        2. Greeks every 30s
        3. Quick recon every 5 min per account (staggered)
        4. Position save every 60s
        5. Fill processing (event-driven via Redis subscription)
        """
        await self.restore_positions()

        tasks = [
            asyncio.create_task(self._mtm_loop()),
            asyncio.create_task(self._greeks_loop()),
            asyncio.create_task(self._recon_loop()),
            asyncio.create_task(self._save_loop()),
            asyncio.create_task(self._fill_listener()),
        ]
        await asyncio.gather(*tasks)

    async def _mtm_loop(self) -> None:
        while True:
            try:
                await self.mark_to_market()
                await self.compute_aggregate_views()
            except Exception as e:
                logger.error("mtm_error", error=str(e))
            await asyncio.sleep(5)

    async def _greeks_loop(self) -> None:
        while True:
            try:
                await self.compute_greeks()
            except Exception as e:
                logger.error("greeks_error", error=str(e))
            await asyncio.sleep(30)

    async def _recon_loop(self) -> None:
        while True:
            for i, account_id in enumerate(self._account_ids):
                try:
                    await self.reconcile_account(account_id, "QUICK")
                except Exception as e:
                    logger.error("recon_error",
                               account_id=account_id, error=str(e))
                await asyncio.sleep(0.5)  # stagger 500ms between accounts
            await asyncio.sleep(300)  # 5 minutes

    async def _save_loop(self) -> None:
        while True:
            try:
                await self.save_positions()
            except Exception as e:
                logger.error("save_error", error=str(e))
            await asyncio.sleep(60)

    async def _fill_listener(self) -> None:
        """Listen for fill events from OMS via Redis pub/sub."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe("CHANNEL:FILLS")
        async for message in pubsub.listen():
            if message["type"] == "message":
                fill_data = orjson.loads(message["data"])
                fill = OrderResult(**fill_data)
                await self.on_fill(fill)
```

---

### State Table

Every piece of state is either per-account or shared. This table is the definitive reference.

| State | Scope | Redis Key Pattern | Updated By | Read By |
|-------|-------|-------------------|------------|---------|
| Position (live) | Per-account | `POSITION:account:{account_id}:{strategy_id}` | PositionTracker | Risk Manager, OMS |
| Position (aggregated) | Shared (cross-account) | `POSITION:strategy:{strategy_id}` | PositionTracker | Strategy processes |
| Position (saved for recovery) | Per-account | `POSITION:saved:{account_id}:{security_id}` | PositionTracker | PositionTracker (startup) |
| Daily returns cache | Per-account | `CACHE:daily_returns:{account_id}:{strategy_id}` | PositionTracker | Capital Allocator |
| Aggregated daily returns | Shared | `RETURNS:{sid}:daily` (Redis LIST) | PositionTracker (weighted sum across all accounts) | Capital Allocator (ERC computation) |
| Strategy PnL cache | Per-account | `CACHE:strategy_pnl:{account_id}:{strategy_id}` | PositionTracker | Risk Manager |
| Account NAV | Per-account | `CACHE:account_nav:{account_id}` | PositionTracker | Monitoring |
| Aggregate view | Shared | `CACHE:aggregate_view` | PositionTracker | Grafana |
| Market data (ticks) | Shared | `LASTTICK:{symbol}` | Data Ingester | PositionTracker (MTM) |
| Option chain | Shared | `CHAIN:{underlying}:{expiry}` | Instrument Resolution (background poller) | PositionTracker (Greeks) |
| DuckDB trades | Per-account | N/A (file, table per account) | PositionTracker | PositionTracker only |
| DuckDB daily returns | Per-account | N/A (file, table per account) | PositionTracker | PositionTracker only |
| DuckDB realized PnL | Per-account | N/A (file, table per account) | PositionTracker | PositionTracker only |
| DuckDB reconciliation | Per-account | N/A (file, table per account) | PositionTracker | PositionTracker only |
| Prometheus metrics | Labels include account_id | N/A (scrape endpoint) | PositionTracker | Grafana |

---

### Failure Modes

| # | Scenario | Detection | Impact | Recovery | Severity |
|---|----------|-----------|--------|----------|----------|
| 1 | DuckDB file locked by stale process | Write timeout (>5s) | Trades not persisted, cache stale | Kill stale process. Replay fills from audit log. | CRITICAL |
| 2 | DuckDB file corrupted | Read/write exception on startup | All historical data lost | Restore from S3 nightly backup. Replay today's fills from audit log. RPO: 1 day. | CRITICAL |
| 3 | Redis unavailable | ConnectionError on cache write | MTM cache stale. Other components read stale data. Positions still tracked in-memory. | Sentinel failover (<1s). If persistent: positions protected by server-side SLs. | HIGH |
| 4 | Dhan positions API returns error | HTTP 4xx/5xx on recon call | Reconciliation skipped for one cycle | Retry next cycle (5 min). If 3 consecutive failures: Telegram CRITICAL. | MEDIUM |
| 5 | Dhan positions API returns stale data | Qty mismatch persists across multiple recon cycles | False discrepancy alerts | After 3 consecutive mismatches, force sync from broker (broker is truth). | MEDIUM |
| 6 | Fill notification lost (WS drop + REST miss) | Post-order timer fires, broker shows TRADED | Local state missing a position. Ghost risk. | force_sync_from_broker catches it. Pre-order broker position check prevents second entry. | CRITICAL |
| 7 | MTM price stale (no tick for >30s) | Tick staleness check in MTM loop | PnL and risk calculations use stale prices | Use last known price. Flag stale instruments. Skip Greeks computation for stale prices. | MEDIUM |
| 8 | Greeks computation error (IV solve fails) | Exception in BSM inversion | Greeks set to None for that position. Delta exposure potentially underestimated. | Use previous Greeks. Log warning. Do not block MTM. | LOW |
| 9 | Account API key expired mid-session | 401 on any Dhan API call for that account | Cannot reconcile, cannot sync that account | Telegram CRITICAL. Other accounts unaffected. Existing SLs still active. | HIGH |
| 10 | Orphan position on startup | Broker shows position not in saved state | Unknown position with no strategy attribution | Create ORPHAN position. Telegram CRITICAL. Manual review. Do not auto-close. | HIGH |
| 11 | Position saved to Redis but DuckDB write failed | DuckDB exception after Redis publish | Redis shows position, DuckDB missing trade record | On next startup: audit log replay fills any DuckDB gaps. | MEDIUM |
| 12 | Concurrent startup (two PositionTracker processes) | DuckDB file lock contention | Second process blocks or corrupts | PID file check on startup. If PID file exists and process alive: abort with error. | CRITICAL |
| 13 | System clock drift >100ms | Chrony alert | Timestamps unreliable for ordering, IV computation skewed | Alert ops. Do not halt trading. Exchange timestamps remain authoritative. | LOW |
| 14 | Memory pressure (too many positions in-memory) | RSS monitoring via Prometheus | OOM kill risk | Position count should never exceed ~50 across all accounts. If >100: investigate. | LOW |

---

### Edge Cases

**1. Same instrument, same strategy, two accounts, different directions.**
Account A is LONG NIFTY 24500 CE for S1 (filled earlier), Account B is SHORT the same (filled after reversal signal, A's exit failed). This is a valid state. Each position tracked independently. Aggregate view for S1 may show FLAT even though individual accounts are not.

**2. Partial fill on one account, full fill on another.**
Signal for S1 goes to accounts A, B, C. A fills 10 lots, B fills 6 lots (partial, remainder abandoned), C fills 0 (rejected). Positions: A has 10 lots, B has 6 lots, C has nothing. Daily returns diverge. Divergence tracked by DivergenceTracker (OMS component) and reflected in per-account DuckDB tables.

**3. Orphan position at startup with negative PnL.**
Broker shows -200k position for Account B that is not in saved state. Position created as ORPHAN. No auto-close (could make losses worse). Manual review via Telegram alert. Operator can assign to a strategy or close manually.

**4. EOD flatten partially completes, then system crashes.**
Account A: 3 of 5 positions flattened. System crashes. On restart: broker shows 2 remaining positions for A. Restored as overnight (they survived session close via Dhan's auto-square timeout). Reconciliation at 09:16 next day picks them up. If they were intraday strategies, operator notified.

**5. DuckDB backup running during EOD write burst.**
S3 backup at 16:30 reads DuckDB while EOD summaries are being written. DuckDB supports concurrent reads, so this is safe. The backup may capture a partial EOD state, but the next backup captures everything. Backup is a snapshot of the file; DuckDB's ACID guarantees mean the snapshot is always consistent.

**6. Account removed from config between sessions.**
On startup, the PositionTracker does not create tables for the removed account. But if broker still shows positions for that account (not yet closed), the account is not queried because it is not in the config. Resolution: operator must close all positions for the removed account before removing it from config. Config validation checks this.

**7. Strategy reassignment (position moves from S1 to S3).**
Not supported automatically. If needed: close position under S1, open under S3. The close and open are separate trades with separate cost impact. Cross-strategy position transfer is not a thing.

**8. Expiry day: option expires worthless while position still open.**
On expiry day, NSE auto-exercises/expires options at settlement. The position disappears from broker at EOD. Next morning reconciliation at 09:16 detects MISSING_BROKER for the expired option. If it expired worthless: realize full loss of premium. If it expired ITM: exchange settlement applies; realized PnL computed from settlement price. The PositionTracker writes a trade record with trade_type "EXPIRY_SETTLEMENT".

**9. Split fill across multiple exchanges.**
NSE only in v1. Not applicable. If BSE added in v2, security_id is exchange-specific, so positions are naturally separate.

**10. Negative avg_entry_price after cost adjustment.**
Not possible. avg_entry_price is the raw fill price. Costs are tracked separately in TradeRecord. PnL computation: `(exit_price - entry_price) * qty - total_costs`. Entry price is never adjusted for costs.

---

### Concurrency Model

The PositionTracker runs as a single async process (`asyncio` event loop) within one OS process. All account operations execute within this single event loop, sequentially for DuckDB writes and concurrently for Redis reads.

```
┌─────────────────────────────────────────────────────────────────┐
│  PositionTracker Process (single asyncio event loop)           │
│                                                                 │
│  Concurrent tasks (asyncio.gather):                             │
│  ├── _mtm_loop          (every 5s, iterates ALL accounts)       │
│  ├── _greeks_loop       (every 30s, iterates ALL accounts)      │
│  ├── _recon_loop        (every 5min, per-account, staggered)    │
│  ├── _save_loop         (every 60s, iterates ALL accounts)      │
│  └── _fill_listener     (event-driven, processes fills FIFO)    │
│                                                                 │
│  DuckDB writes:  SEQUENTIAL (one write at a time)               │
│  Redis reads:    CONCURRENT (asyncio, non-blocking)             │
│  Redis writes:   SEQUENTIAL per key (no race, single writer)    │
│  Dhan API calls: SEQUENTIAL per account (rate limiter)          │
│                  CONCURRENT across accounts (independent OPS)   │
│                                                                 │
│  Fill processing order:                                         │
│  1. fill_listener receives fill event                           │
│  2. on_fill() updates in-memory Position                        │
│  3. write_trade() inserts to DuckDB (sync, blocks event loop    │
│     for ~1ms per insert)                                        │
│  4. _publish_position_update() writes to Redis (async)          │
│  5. Next fill processed only after step 4 completes             │
│                                                                 │
│  Why single process, not one per account:                       │
│  - DuckDB file lock prevents concurrent writers anyway          │
│  - Single process simplifies aggregate view computation         │
│  - N accounts x 5-10 positions = 15-50 positions total          │
│  - MTM loop for 50 positions: <1ms. No parallelism needed.      │
│  - Recon API calls are I/O bound; asyncio handles concurrency   │
└─────────────────────────────────────────────────────────────────┘
```

**DuckDB write latency:** A single INSERT into a DuckDB table takes ~0.5-1ms on NVMe SSD. With peak fill rate of 10 fills/second (across all accounts during EOD flatten), DuckDB writes consume ~10ms/s of event loop time. This leaves >99% of the event loop available for MTM, Greeks, and Redis operations.

**Scaling limit:** At 10+ accounts with 20+ positions each, the MTM loop iterating 200+ positions may take >5ms per cycle. If this becomes a bottleneck, the MTM loop can be moved to a thread pool executor (`loop.run_in_executor`), keeping DuckDB writes on the main thread. This is a v2 optimization; v1 targets 3-5 accounts.
