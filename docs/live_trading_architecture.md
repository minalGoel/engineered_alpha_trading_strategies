# Live Trading System — Architecture Document

**Date:** 2026-03-23
**Status:** Design (pre-implementation)
**Scope:** v1 — single-leg orders, single broker for data + execution

---

## 1. SYSTEM OVERVIEW

A production live trading system for 7 strategies on NSE, using a single broker (Dhan or Upstox) for both tick-by-tick market data and order execution. The system runs on an AWS `ap-south-1` EC2 instance with a static Elastic IP (SEBI compliance). Initial capital: ₹50L, scaling to ₹10Cr by Year 2.

The backtest pipeline (existing `pipeline/`) remains untouched. The live system is a new top-level package `live/` that reuses the cost model and strategy signal logic but adds real-time execution infrastructure.

### Broker Selection

The architecture is broker-agnostic via an adapter pattern. Either Dhan or Upstox serves as the single broker for both data and execution.

| Dimension | Dhan | Upstox |
|-----------|------|--------|
| OPS limit | 10 (reduced from 25, Mar 2025) | 50 |
| WS instruments | 25,000 | 100 |
| Depth levels | 20/200 | 5 |
| Tick delivery | Event-driven | Event-driven |
| Historical data | 5 years (including expired options) | Multi-year |
| HFT endpoint | No | `api-hft.upstox.com` |
| Static IP | Live | Unclear |
| Daily order limit | 5,000 | Higher |

**Recommendation:** Dhan for v1 (better data depth, expired options for research, static IP confirmed). Switch to Upstox if 10 OPS becomes a binding constraint at scale.

**Convention:** This document uses "broker" generically. Broker-specific details (endpoint URLs, auth flows, field names) live in the `BrokerAdapter` implementation, not in the architecture.

### Process Topology

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EC2 ap-south-1 (Ubuntu, static Elastic IP)                           │
│                                                                         │
│  ┌────────────────┐                                                     │
│  │ Redis Sentinel  │  (primary + 1 replica, same host)                 │
│  └────────┬───────┘                                                     │
│           │                                                              │
│  ┌────────▼───────┐  Redis Streams     ┌──────────────────────────┐    │
│  │ Data Ingester   │ ────────────────→ │ Strategy Processes (×7)  │    │
│  │ (async)         │  STREAM:TICK:*    │  S1..S7 (multiprocess)   │    │
│  │ + Tick WAL      │                   └──────────┬───────────────┘    │
│  └──────┬─────────┘                               │ signals            │
│         │ Parquet (5s flush)                       ▼                    │
│         │                              ┌──────────────────────────┐    │
│  ┌──────▼─────────┐                   │  Signal Router (async)   │    │
│  │ Tick Archive    │                   │  + Capital Allocator     │    │
│  │ (Parquet → S3)  │                   │  + Instrument Resolver   │    │
│  └────────────────┘                   │  + Risk Gate             │    │
│                                        └──────────┬───────────────┘    │
│                                                    │ approved orders    │
│                                        ┌───────────▼──────────────┐    │
│                                        │ OMS (async)              │    │
│                                        │  + Fill Manager          │    │
│                                        │  + SL Lifecycle Manager  │    │
│                                        │  + EOD Flatten           │    │
│                                        └───────────┬──────────────┘    │
│                                                    │ fills              │
│                                        ┌───────────▼──────────────┐    │
│                                        │ Position & PnL Tracker   │    │
│                                        │ (sole DuckDB writer)     │    │
│                                        └───────────┬──────────────┘    │
│                                                    │                    │
│  ┌────────────────┐                   ┌───────────▼──────────────┐    │
│  │ Risk Monitor    │◄─────────────────│ Post-Trade Monitor       │    │
│  │ (kill/halt)     │                   └──────────────────────────┘    │
│  └──────┬─────────┘                                                    │
│         │                                                               │
│  ┌──────▼──────────────────────┐    ┌────────────────────┐            │
│  │ Prometheus + Grafana        │    │ Audit Logger        │            │
│  └──────┬──────────────────────┘    │ (structlog → S3)    │            │
│         │                            └────────────────────┘            │
│  ┌──────▼─────────┐                                                    │
│  │ Telegram Notifs │  (status only — no execution control)             │
│  └────────────────┘                                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

**Infrastructure honesty:** Redis Sentinel on the same host protects against Redis process crashes only — not EC2 instance failure, EBS failure, or AZ outage. Host-level protection comes from server-side stop-losses and broker auto-square at session close. True cross-host HA (separate Redis instance) is a v2 concern at ₹1Cr+.

### Signal-to-Fill Data Flow

```
Broker WS (ticks) ──→ DataIngester ──→ Redis STREAM:TICK:* ──→ Strategy Process
                           │                                         │
                      Tick WAL + Parquet                    StrategySignal
                                                                     │
                                                        Redis STREAM:SIGNAL
                                                                     ▼
                                              SignalRouter
                                                │
                                    1. Signal dedup check
                                    2. Capital allocator (existing position aware)
                                    3. Instrument resolver (on-demand chain API)
                                    4. Risk gate (incl. broker position check)
                                                │
                                         ResolvedOrder
                                                │
                                                ▼
                                    OMS.place_entry_with_sl()
                                      ├── Entry LIMIT order
                                      └── Paired SL order (server-side)
                                                │
                                    Broker WS (order updates) ──→ FillManager
                                                │
                                                ▼
                                    PositionTracker.on_fill()
                                      + post-order reconciliation timer
                                      + SL qty sync
                                                │
                                    Audit log + Prometheus + Telegram
```

### Latency Budget (Signal-to-Exchange)

```
Signal fires (bar close)              t = 0ms
Signal dedup + allocation check       t = 1-5ms
On-demand option chain API call       t = 200-400ms
Risk gate checks                      t = 5-20ms
Rate limiter wait (avg)               t = 0-100ms
Entry order API call                  t = 50-200ms
SL order API call                     t = 50-200ms
Broker OMS → Exchange                 t = 100-3,000ms (async, not in our control)
─────────────────────────────────────────────────────
Total (our side):                     ~300-900ms
Total (incl. broker-to-exchange):     ~500-4,000ms

Signal half-life (minimum):           >30 seconds
Our latency as % of signal life:      1-3%
```

---

## 2. COMPONENT DESIGNS

---

### Component 1: Data Ingestion Layer

#### Responsibility
- Subscribe to broker's tick-by-tick websocket for real-time market data
- Normalize ticks into a canonical `Tick` model
- Distribute ticks to strategy processes via Redis Streams (persistent, backpressure-aware)
- Record raw ticks via write-ahead log (WAL) + Parquet with 5-second flush
- **Does NOT** interpret ticks, compute indicators, or make trading decisions

#### Interface Contracts

**Input:** Broker tick websocket (single connection, up to instrument limit).

**Output — Redis Streams:**

```python
class Tick(pydantic.BaseModel):
    symbol: str              # canonical: e.g. "NIFTY50-INDEX"
    ltp: float               # last traded price
    bid: float               # best bid
    ask: float               # best ask
    bid_qty: int
    ask_qty: int
    volume: int              # cumulative daily volume
    oi: int                  # open interest (0 for indices)
    exchange_ts: int         # exchange timestamp (epoch ms)
    receive_ts: int          # our receive timestamp (epoch ms)
```

Streams (one per instrument class):
- `STREAM:TICK:SPOT` — NIFTY50, BANKNIFTY, VIX indices
- `STREAM:TICK:FUT:{symbol}` — individual futures
- `STREAM:TICK:OPT:{underlying}` — option ticks by underlying
- `STREAM:TICK:EQ:{symbol}` — equities (S4)

Each strategy process creates a **consumer group** on the streams it subscribes to. Ticks persist in the stream, trimmed to N entries via `XTRIM MAXLEN`. The trim threshold is tuned after load testing — initial estimate: 60,000 entries per stream (~5 min at 200 ticks/s), validated by measuring actual peak publish rates during market open.

If a strategy process is slow or reconnects, it reads from its last acknowledged offset. No lost ticks.

Message format: orjson-serialized `Tick`. ~150 bytes per tick. Published via `XADD`.

**Output — Tick WAL + Parquet:**

```
WAL:      /var/log/trading/tick_wal/{date}.wal  (append-only mmap'd binary)
Parquet:  /var/log/trading/parquet/{date}/{symbol}/{HH}.parquet
S3:       s3://trading-data-archive/{date}/ticks/{symbol}/{HH}.parquet
```

Every tick is appended to the WAL **before** publishing to Redis. The WAL survives process crashes including SIGKILL and segfaults. On restart, the Parquet flush task replays the WAL from the last checkpoint, deduplicates by `(symbol, exchange_ts)` pair, and writes to Parquet.

WAL record format: fixed-width 60 bytes. At 1000 ticks/s peak = 60KB/s = ~180MB/day.

Parquet flush interval: **5 seconds**. At ~75KB per flush — trivial I/O. `atexit.register()` AND `signal.signal(SIGTERM/SIGINT)` handlers flush on graceful shutdown. For SIGKILL/segfault, only the WAL survives (max 5s of un-flushed ticks).

**Idempotent replay:** The WAL contains a monotonic sequence number per tick. The Parquet writer tracks the last written sequence. On replay, ticks with sequence ≤ last_written are skipped. Partial Parquet writes (crash mid-flush) are detected by checking the footer — incomplete files are discarded and rewritten from WAL.

#### State
| What | Where | Lifecycle |
|------|-------|-----------|
| WS connection handle | In-memory | Process lifetime |
| Subscription list | Config → in-memory | Session |
| Tick WAL file | Disk (mmap'd) | Survives crashes. New file per day. |
| Tick buffer (pre-Parquet) | In-memory deque, max 5s | Flushed every 5s |
| Last tick per symbol | Redis `LASTTICK:{symbol}` (hash) | Overwritten per tick |
| Connection health | Redis `HEALTH:data_ingester` | Heartbeat every 5s with TTL 15s |

#### Concurrency Model
Single async process: `asyncio` + `websockets`.

- `asyncio.Task`: WS reader (receives ticks, writes WAL, publishes to Redis Stream, updates LASTTICK)
- `asyncio.Task`: Parquet flush loop (5s)
- `asyncio.Task`: Heartbeat (5s)

I/O-bound — a single async loop handles thousands of ticks/sec.

#### Failure Modes

| Failure | Detection | Recovery |
|---------|-----------|----------|
| WS disconnect | `ConnectionClosed` exception | Exponential backoff: 1s, 2s, 4s, 8s, max 30s. Re-subscribe. |
| WS stale (connected, no ticks) | Watchdog: no tick on ANY instrument for 10s | Force close + reconnect |
| **All instruments silent simultaneously** | Zero ticks on all streams for 15s | Exchange halt/closure. Telegram CRITICAL. Suppress new entries. Existing positions protected by server-side SL. |
| Redis down | `aioredis` ConnectionError | Retry via Sentinel. WAL continues locally. Strategies paused. OMS enters degraded mode. |
| Process crash | systemd detects exit | Restart. WAL survives. Replay WAL → Parquet. |

---

### Component 2: Instrument Resolution Layer

#### Responsibility
- Translate abstract strategy signals ("go long NIFTY", "sell PE") into a concrete broker instrument ID with a limit price
- Maintain broker instrument master (daily CSV) as canonical mapping
- **Always make an on-demand option chain API call when a signal fires** for order pricing
- Maintain background 5s polling for monitoring/Greeks/display only
- Validate liquidity and spread
- **Does NOT** place orders, manage positions, or size trades

#### Instrument Master

Downloaded at 08:30 IST daily. Parsed into O(1) lookup indices:

```python
class InstrumentRecord(pydantic.BaseModel):
    instrument_id: str            # broker's unique ID
    trading_symbol: str           # human-readable
    underlying: str               # "NIFTY" | "BANKNIFTY" | equity ticker
    exchange_segment: str         # "NSE_FNO" | "NSE_EQ"
    instrument_type: str          # "OPTIDX" | "FUTIDX" | "EQUITY"
    option_type: str | None       # "CE" | "PE" | None
    strike_price: float | None
    expiry_date: date | None
    lot_size: int
    tick_size: float              # e.g. 0.05 for options
    freeze_qty: int               # max qty before slicing needed

class InstrumentMaster:
    _option_index: dict[tuple[str, date, float, str], InstrumentRecord]
    _futures_index: dict[tuple[str, date], InstrumentRecord]
    _equity_index: dict[str, InstrumentRecord]

    def lookup_option(self, underlying, expiry, strike, option_type) -> InstrumentRecord | None
    def lookup_future(self, underlying, expiry) -> InstrumentRecord | None
    def lookup_equity(self, symbol) -> InstrumentRecord | None
```

**Fallback on download failure:** Retry 3x (08:30, 08:35, 08:40). If all fail, use yesterday's CSV with WARNING. **On Tuesdays (weekly expiry day): if fallback triggered, disable S5 for the day** — 0-DTE contracts may not be in yesterday's file. Telegram WARNING.

#### Option Chain — Dual Mode

| Mode | Purpose | Frequency | Used For |
|------|---------|-----------|----------|
| Background poll | Monitoring, Greeks, dashboards | Every 5s | Grafana, risk monitor, Position Tracker Greeks |
| **On-demand at signal time** | **Order pricing** | Per signal (~10-20/day) | Limit price, spread check, liquidity validation |

The on-demand call adds 200-400ms to the signal-to-order path. This is the correct tradeoff: 200ms latency is far cheaper than placing orders with 5s-stale pricing data that would require multiple reprice cycles.

```python
class OptionChainEntry(pydantic.BaseModel):
    strike: float
    option_type: str              # "CE" | "PE"
    ltp: float
    bid: float
    ask: float
    bid_qty: int
    ask_qty: int
    oi: int
    volume: int
    iv: float | None              # may not be provided by all brokers
    delta: float | None           # may not be provided — compute locally if missing
    gamma: float | None
    theta: float | None
    vega: float | None
    last_trade_ts: int            # epoch ms

class OptionChainSnapshot:
    underlying: str
    expiry: date
    spot_price: float
    snapshot_ts: int
    entries: dict[tuple[float, str], OptionChainEntry]
```

**Greeks availability:** If the broker's option chain API does not return Greeks, compute locally using Black-76 with IV derived from bid/ask midpoint and the BSM inversion. This must be validated during paper trading.

#### Resolution Pipeline

```python
class StrategySignal(pydantic.BaseModel):
    strategy_id: str
    signal_id: str                # UUID, unique per signal
    signal_ts: int                # epoch ms
    direction: Literal["LONG", "SHORT"]
    underlying: str
    instrument_hint: Literal["CE", "PE", "FUT", "EQ"]
    expiry_preference: Literal["WEEKLY", "MONTHLY", "NEAREST"] | None
    urgency: Literal["NORMAL", "URGENT"]
    metadata: dict                # indicator values, bar data for audit

class ResolvedOrder(pydantic.BaseModel):
    signal: StrategySignal
    instrument_id: str            # broker instrument ID
    trading_symbol: str
    exchange_segment: str
    transaction_type: Literal["BUY", "SELL"]
    product_type: Literal["INTRADAY", "CNC", "MARGIN"]
    quantity: int                  # in units (lot_size × lots)
    limit_price: float
    tick_size: float
    lot_size: int
    freeze_qty: int
    spread_bps: float
    cost_estimate: float          # estimated RT cost in ₹
    sl_trigger_price: float       # for mandatory server-side SL
    sl_limit_price: float         # SL limit = trigger ∓ 2 ticks
    sl_validity: str              # "DAY" for intraday, "GTT" for overnight (S2)
```

**Algorithm:**

```
1. RECEIVE StrategySignal

2. DETERMINE expiry:
   S1: monthly (last Thu)    S2: nearest futures    S3: monthly
   S4: equity (no expiry)    S5: weekly Tuesday     S6: monthly
   S7: nearest stock futures

3. FOR options (CE/PE):
   a. GET spot price from Redis LASTTICK (real-time)
   b. COMPUTE ATM strike = floor(spot / 50 + 0.5) * 50
      Tie-break on exact midpoint: lower strike (better liquidity)
   c. MAKE on-demand option chain API call (200-400ms)
      - On failure: retry ONCE. If still fails, fall back to cached
        snapshot if <2s old. If cache also stale: REJECT signal.
   d. GET bid/ask for (ATM_strike, CE/PE) from fresh snapshot
   e. LIQUIDITY check: if bid==0 or ask==0, try ±1 strike ITM.
      Still zero → REJECT.
   f. SPREAD check:
      spread_bps = (ask - bid) / mid * 10000
      Normal: reject if > 500 bps
      0-DTE (S5 Tuesday): reject if > 1000 bps
   g. LIMIT PRICE:
      BUY + NORMAL:  best_ask
      BUY + URGENT:  best_ask + 1 tick
      SELL + NORMAL: best_bid
      SELL + URGENT: best_bid - 1 tick
   h. ROUND to tick_size
   i. COMPUTE SL prices:
      BUY: sl_trigger = entry - strategy_stop_points
      SELL: sl_trigger = entry + strategy_stop_points
      sl_limit = trigger ∓ 2 ticks
   j. DETERMINE SL validity:
      Intraday strategies (S1,S3,S5): "DAY"
      Overnight/multi-day (S2,S6,S7): "GTT" (Good Till Triggered)
   k. LOOKUP instrument_id from InstrumentMaster
   l. ESTIMATE cost via cost_model
   m. RETURN ResolvedOrder

4. FOR futures: same flow, no strike resolution
5. FOR equities: direct lookup, no chain call needed
```

**0-DTE guards (S5 on Tuesday):** Reject bid_qty < 100 lots. Wider spread threshold. Force urgency=URGENT.

**v2 extensibility:** Signal can carry `legs: list[LegSpec]` for multi-leg. v1: always single leg.

#### State
| What | Where | Lifecycle |
|------|-------|-----------|
| InstrumentMaster | In-memory | Loaded 08:30, immutable for session |
| Background chain snapshots | In-memory + Redis `CHAIN:*` | Updated 5s (monitoring only) |
| Spot prices | Redis `LASTTICK:*` | Updated per tick |

#### Concurrency
Async tasks within the Signal Router process. `resolve()` is `async def` — the on-demand API call is awaited, not blocking.

---

### Component 3: Strategy Orchestrator

#### Responsibility
- Launch each strategy as an isolated OS process
- Route market data via Redis Streams consumer groups
- Collect signals and forward to Signal Router
- Lifecycle management: start, stop, restart, kill
- **Does NOT** place orders, manage positions, or allocate capital

#### Strategy Interface

```python
class LiveStrategy(ABC):
    strategy_id: str
    subscriptions: list[str]        # Redis stream names
    bar_interval_s: int             # bar duration in seconds (e.g. 1800 for S1's 30-min)

    @abstractmethod
    def on_tick(self, tick: Tick) -> StrategySignal | None:
        """Process tick. Must be <10ms. Return signal or None."""
        ...

    @abstractmethod
    def on_bar_close(self, bar: Bar) -> StrategySignal | None:
        """Called on bar close. May take up to 100ms (runs in thread executor)."""
        ...

    @abstractmethod
    def on_fill(self, fill: FillNotification) -> None:
        """Fill/reject notification. Update internal state."""
        ...

    def get_state(self) -> dict:
        """Serializable state for crash recovery."""
        ...

    def restore_state(self, state: dict) -> None:
        """Restore after restart."""
        ...
```

#### Bar Builder

Each strategy has its own `BarBuilder` instance, initialized with the strategy's `bar_interval_s`.

```python
class BarBuilder:
    """Aggregates ticks into OHLCV bars with gap detection."""

    def __init__(self, interval_s: int):
        self.interval_s = interval_s
        self._ticks: list[Tick] = []
        self._bar_start_ts: int = 0
        self._last_tick_ts: int = 0

    def add_tick(self, tick: Tick) -> Bar | None:
        """Add tick. Returns completed Bar when interval closes, else None."""
        ...

    def _build_bar(self) -> Bar:
        gap_ms = max((t2.exchange_ts - t1.exchange_ts)
                     for t1, t2 in pairwise(self._ticks))
        return Bar(
            ...,
            has_gap=(gap_ms > 2 * self._expected_tick_interval_ms),
            max_tick_gap_ms=gap_ms,
        )

class Bar(pydantic.BaseModel):
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    bar_start_ts: int
    bar_end_ts: int
    vwap: float
    tick_count: int
    has_gap: bool               # True if suspicious tick gap detected
    max_tick_gap_ms: int        # longest gap between consecutive ticks
```

**Bar intervals per strategy:**

| Strategy | Bar Interval | Notes |
|----------|-------------|-------|
| S1 (ORB) | 1800s (30 min) | ORB range = first 30-min bar |
| S2 (Overnight) | N/A | Event-driven (15:15 entry, 09:20 exit) |
| S3 (VWAP MR) | 300s (5 min) | VWAP computed from tick stream |
| S4 (Momentum) | 86400s (daily) | Monthly rebalance, only active on rebalance days |
| S5 (Expiry Day) | 60s (1 min) | Fast bars for 0-DTE |
| S6 (Vol Premium) | 300s (5 min) | VIX regime checks |
| S7 (Pairs) | 300s (5 min) | Spread monitoring |

**Tick gap handling:** When `has_gap=True`, the strategy can choose to suppress signals for that bar. The architecture does not force suppression — some strategies (momentum) are robust to gaps, others (mean reversion) are not. Each strategy's `on_bar_close` receives the `has_gap` flag and decides.

#### Strategy Process Event Loop

```python
async def strategy_main(strategy: LiveStrategy):
    group = f"strategy_{strategy.strategy_id}"
    bar_builder = BarBuilder(strategy.bar_interval_s)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def tick_reader():
        """Read ticks from Redis Streams. NEVER blocked by bar computation."""
        while True:
            try:
                entries = await redis.xreadgroup(
                    group, consumer=strategy.strategy_id,
                    streams=strategy.subscriptions,
                    count=100, block=50
                )
                for stream_name, messages in entries:
                    for msg_id, data in messages:
                        tick = Tick.model_validate_json(data[b"tick"])

                        # on_tick: fast path (<10ms)
                        signal = await strategy.on_tick(tick)
                        if signal:
                            await publish_signal(signal)

                        # Bar aggregation
                        bar = bar_builder.add_tick(tick)
                        if bar is not None:
                            # on_bar_close: runs in THREAD EXECUTOR
                            # Does NOT block the event loop
                            bar_signal = await asyncio.get_event_loop().run_in_executor(
                                executor, strategy.on_bar_close, bar
                            )
                            if bar_signal:
                                await publish_signal(bar_signal)

                        await redis.xack(stream_name, group, msg_id)
            except aioredis.ConnectionError:
                logger.warning("redis_connection_lost")
                await asyncio.sleep(2)  # retry; consumer group preserves offset

    async def heartbeat():
        while True:
            await redis.set(f"HEALTH:strategy:{strategy.strategy_id}",
                          str(now_ms()), ex=15)
            await asyncio.sleep(5)

    async def parent_watchdog():
        parent_pid = os.getppid()
        while True:
            await asyncio.sleep(5)
            try:
                os.kill(parent_pid, 0)
            except OSError:
                logger.critical("parent_died")
                await redis.set(f"STATE:strategy:{strategy.strategy_id}",
                              orjson.dumps(strategy.get_state()))
                sys.exit(1)

    # Run all tasks. If tick_reader throws, log and restart it — don't kill the process.
    tasks = [
        asyncio.create_task(_restart_on_error(tick_reader, "tick_reader")),
        asyncio.create_task(heartbeat()),
        asyncio.create_task(parent_watchdog()),
    ]
    await asyncio.gather(*tasks)

async def _restart_on_error(coro_fn, name: str):
    """Wrap a coroutine so unhandled exceptions restart it instead of killing the process."""
    while True:
        try:
            await coro_fn()
        except Exception:
            logger.exception(f"task_{name}_crashed_restarting")
            await asyncio.sleep(1)
```

**Key design:** `on_bar_close` runs in a thread executor. The event loop continues reading ticks while the strategy computes. Redis Streams consumer groups ensure no ticks are lost during computation or reconnection.

#### Data Routing

| Strategy | Subscriptions |
|----------|--------------|
| S1 (ORB) | `STREAM:TICK:SPOT` |
| S2 (Overnight) | `STREAM:TICK:FUT:NIFTY` |
| S3 (VWAP MR) | `STREAM:TICK:SPOT`, `STREAM:TICK:FUT:NIFTY` |
| S4 (Momentum) | `STREAM:TICK:EQ:*` |
| S5 (Expiry Day) | `STREAM:TICK:SPOT` |
| S6 (Vol Premium) | `STREAM:TICK:SPOT` |
| S7 (Pairs) | `STREAM:TICK:FUT:{sym1}`, `STREAM:TICK:FUT:{sym2}` |

#### Cross-Strategy Awareness

S3 reads S1's position from Redis `POSITION:strategy:S1` (read-only check). The Position Tracker is the sole writer of these keys.

#### Consumer Group Creation

The orchestrator creates consumer groups in the startup sequence (Phase 8) **before** starting strategy processes:

```python
for stream in all_subscribed_streams:
    await redis.xgroup_create(stream, group_name, id="$", mkstream=True)
    # "$" = only new messages from this session
```

Using `$` ensures strategies see only live ticks, not stale data from a previous session.

#### Failure Modes

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Strategy crashes | Heartbeat missing 15s | Orchestrator restarts, calls `restore_state()`. Positions protected by server-side SL. |
| Strategy hangs | Heartbeat missing 30s, PID alive | SIGTERM → 5s → SIGKILL → restart |
| Redis lost in child | ConnectionError in stream reader | Retry every 2s. Consumer group preserves offset — no lost ticks on reconnect. |

---

### Component 4: Signal Router & Capital Allocator

#### Responsibility
- Receive signals from strategy processes
- **Signal deduplication** — prevent duplicate orders
- **Position-aware allocation** — check existing positions before sizing
- ERC allocation, Kelly sizing, margin management
- Route to Instrument Resolver → Risk Gate → OMS
- **Does NOT** place orders

#### Signal Deduplication

```python
class SignalDeduplicator:
    """Suppress duplicate signals within a cooldown window."""

    _last_signal: dict[tuple[str, str, str], int]  # (sid, direction, underlying) → timestamp_ms

    COOLDOWN_MS = {
        "S1": 300_000,   # 5 min — ORB fires once
        "S2": 86400_000, # 1 day — overnight fires once
        "S3": 60_000,    # 1 min
        "S4": 86400_000, # 1 day — monthly rebalance
        "S5": 30_000,    # 30s — fast expiry play
        "S6": 3600_000,  # 1 hour — patient vol selling
        "S7": 300_000,   # 5 min
    }

    def is_duplicate(self, signal: StrategySignal) -> bool:
        key = (signal.strategy_id, signal.direction, signal.underlying)
        last = self._last_signal.get(key, 0)
        cooldown = self.COOLDOWN_MS[signal.strategy_id]
        if signal.signal_ts - last < cooldown:
            return True
        self._last_signal[key] = signal.signal_ts
        return False
```

Also: each signal carries a UUID `signal_id`. The OMS tracks placed `signal_id`s and rejects any signal_id it has already processed (idempotent placement).

#### Position-Aware Allocation

```python
class AllocationRequest(pydantic.BaseModel):
    strategy_id: str
    direction: Literal["LONG", "SHORT"]
    underlying: str
    instrument_type: Literal["OPT_BUY", "OPT_SELL", "FUT", "EQ"]
    estimated_premium: float
    lot_size: int
    existing_position_qty: int    # current position in this strategy (signed)

class AllocationResponse(pydantic.BaseModel):
    approved: bool
    allocated_capital: float
    max_lots: int
    kelly_fraction: float
    rejection_reason: str | None

class CapitalAllocator:
    _lock: asyncio.Lock           # NOT threading.Lock — would block event loop

    async def request_allocation(self, req: AllocationRequest) -> AllocationResponse:
        async with self._lock:
            # Check: does this strategy already hold a position?
            if req.existing_position_qty != 0:
                if same_direction(req.direction, req.existing_position_qty):
                    return AllocationResponse(approved=False,
                        rejection_reason="strategy_already_positioned_same_direction")
                # Opposite direction = exit signal, always approve
                return AllocationResponse(approved=True, ...)

            # Normal allocation logic
            ...
```

**Lock scope:** The `asyncio.Lock` serializes allocation decisions only. The allocation computation itself is fast (<1ms — dict lookups and arithmetic). Pre-computing per-strategy budgets at session start (and on every fill) ensures the lock is held for microseconds, not milliseconds. No head-of-line blocking.

#### ERC Allocation

Recomputed daily at 08:45 IST:

```
σ_i = 20-day rolling vol of strategy daily returns
w_i = (1/σ_i) / Σ(1/σ_j)
```

Days 1-20 (no history): static weights — S1:20%, S2:15%, S3:15%, S4:20%, S5:10%, S6:15%, S7:5%.
Days 20-40: blend 50% static + 50% ERC. Days 40+: pure ERC.

#### Kelly Ramp

```
Days 1-20:   0.25 (quarter Kelly)
Days 21-60:  linear ramp 0.25 → 0.50
Days 61+:    0.50 (half Kelly)

Override: strategy 20d Sharpe < 0.5 → clamp at 0.25
Override: portfolio drawdown > 5% → clamp all at 0.25
```

#### Margin Management

Broker fund/margin API polled every 30s + on every fill. Conservative: full margin per strategy, no netting.

#### Priority

```
1. Exit orders (always highest)
2. S5 (expiry day, time-sensitive)
3. S1 (ORB, time-sensitive)
4. S3 (VWAP MR)
5. S2 (overnight, only at 15:15)
6. S6 (vol premium)
7. S7 (pairs)
8. S4 (momentum rebalance)
```

#### State
| What | Where | Lifecycle |
|------|-------|-----------|
| ERC weights | Redis `ALLOC:erc_weights` | Daily 08:45 |
| Kelly fractions | Redis `KELLY:{sid}` | Daily |
| Current allocations | Redis `ALLOC:{sid}` | Updated on fill/release |
| Dedup state | In-memory dict | Session |
| Margin state | Redis `MARGIN:state` | Every 30s from broker API |
| Daily returns (for vol) | Redis cache, written by Position Tracker | Daily |

---

### Component 5: Order Management System (OMS)

#### Responsibility
- Place limit orders + **mandatory server-side stop-loss orders**
- Track order status via broker WS (primary) — no REST polling when WS is healthy
- Execute fill management loop with per-order locking
- **SL lifecycle management** as a first-class concern
- EOD position flatten (batched, parallel, IOC escalation)
- Rate limiting via priority token bucket
- Degraded mode if Redis is down
- **Global kill switch**

#### Mandatory Server-Side Stop-Loss

Every entry order is paired with a server-side SL. This is the only protection during system outage (process crash, EC2 reboot, Redis failure). It is not optional.

```python
async def place_entry_with_sl(self, order: ResolvedOrder) -> tuple[str, str]:
    """Place entry + SL. Not truly atomic (two API calls), but entry is
    worthless without SL — if SL placement fails, cancel entry immediately."""

    # 1. Place entry
    await self.rate_limiter.acquire("NEW")
    entry_resp = await self.broker.place_order(
        instrument_id=order.instrument_id,
        transaction_type=order.transaction_type,
        order_type="LIMIT",
        validity="DAY",
        quantity=order.quantity,
        price=order.limit_price,
        correlation_id=f"{order.signal.strategy_id}_{order.signal.signal_id[:20]}",
    )

    if entry_resp.status == "REJECTED":
        return (None, None)

    # 2. Place paired SL
    sl_txn = "SELL" if order.transaction_type == "BUY" else "BUY"
    sl_validity = order.sl_validity  # "DAY" for intraday, "GTT" for overnight

    await self.rate_limiter.acquire("SL")
    sl_resp = await self.broker.place_sl_order(
        instrument_id=order.instrument_id,
        transaction_type=sl_txn,
        validity=sl_validity,
        quantity=order.quantity,
        trigger_price=order.sl_trigger_price,
        limit_price=order.sl_limit_price,
        correlation_id=f"SL_{order.signal.strategy_id}_{order.signal.signal_id[:16]}",
    )

    if sl_resp.status == "REJECTED":
        # SL failed — cancel entry immediately. Position without SL is unacceptable.
        logger.critical("sl_placement_failed_cancelling_entry",
                       entry_id=entry_resp.order_id, sl_error=sl_resp.error)
        await self.rate_limiter.acquire("EXIT")
        await self.broker.cancel_order(entry_resp.order_id)
        return (None, None)

    # 3. Register pairing
    self.sl_manager.register(entry_resp.order_id, sl_resp.order_id, order)

    return (entry_resp.order_id, sl_resp.order_id)
```

**Unprotected window:** Between entry fill and SL placement, there is a 100-400ms window. At ₹50L scale, worst-case exposure during a flash crash (65 lots × 200pt move × 200ms): ~₹130. Acceptable at this scale; flag for review at ₹5Cr+.

#### SL Lifecycle Manager

SL is tracked as a first-class object, not just a dict entry. Every SL state transition is verified.

```python
class SLState(pydantic.BaseModel):
    sl_order_id: str
    entry_order_id: str
    instrument_id: str
    strategy_id: str
    sl_qty: int                    # must match filled qty of entry
    sl_trigger: float
    sl_limit: float
    validity: str                  # "DAY" | "GTT"
    status: Literal["PENDING", "TRIGGERED", "CANCELLED", "REJECTED", "UNKNOWN"]
    last_verified_ts: int          # epoch ms of last REST verification

class SLLifecycleManager:
    _sl_states: dict[str, SLState]   # entry_order_id → SLState

    def register(self, entry_id: str, sl_id: str, order: ResolvedOrder) -> None:
        """Register a new SL pairing."""
        ...

    async def on_entry_partial_fill(self, entry_id: str, filled_qty: int) -> None:
        """IMMEDIATELY modify SL qty to match filled qty."""
        sl = self._sl_states[entry_id]
        if sl.sl_qty != filled_qty:
            await self.rate_limiter.acquire("SL")
            await self.broker.modify_order(sl.sl_order_id, quantity=filled_qty)
            sl.sl_qty = filled_qty
            logger.info("sl_qty_synced", entry_id=entry_id, new_qty=filled_qty)

    async def on_entry_fully_filled(self, entry_id: str) -> None:
        """Entry fully filled. SL qty should already match. Verify."""
        sl = self._sl_states[entry_id]
        # Verify via REST — don't trust WS alone
        await self._verify_sl_status(entry_id)

    async def on_normal_exit(self, entry_id: str) -> None:
        """Strategy exits normally (target/signal/time). Cancel SL."""
        sl = self._sl_states[entry_id]
        try:
            await self.rate_limiter.acquire("SL")
            await self.broker.cancel_order(sl.sl_order_id)
            sl.status = "CANCELLED"
        except OrderNotCancellable:
            # SL may have already triggered — check
            await self._verify_sl_status(entry_id)
            if sl.status == "TRIGGERED":
                logger.warning("sl_triggered_during_exit", entry_id=entry_id)
                # Position already closed by SL. Skip the exit order.

    async def on_eod_for_overnight(self, entry_id: str) -> None:
        """EOD for overnight position (S2). DAY SL expires at session close.
        Place fresh GTT or AMO SL for overnight protection."""
        sl = self._sl_states[entry_id]
        if sl.validity == "DAY":
            # Place GTT/Forever order SL for overnight
            await self.rate_limiter.acquire("SL")
            new_sl = await self.broker.place_sl_order(
                instrument_id=sl.instrument_id,
                validity="GTT",  # Good Till Triggered — survives overnight
                quantity=sl.sl_qty,
                trigger_price=sl.sl_trigger,
                limit_price=sl.sl_limit,
                ...
            )
            if new_sl.status == "REJECTED":
                # GTT not supported or other issue — try AMO
                await self.rate_limiter.acquire("SL")
                new_sl = await self.broker.place_sl_order(
                    validity="AMO_OPEN",  # After Market Order, active at next open
                    ...)
            sl.sl_order_id = new_sl.order_id
            sl.validity = "GTT"
            logger.info("sl_converted_to_gtt", entry_id=entry_id)

    async def _verify_sl_status(self, entry_id: str) -> None:
        """REST verification — don't trust WS alone for SL state."""
        sl = self._sl_states[entry_id]
        await self.rate_limiter.acquire("POLL")
        resp = await self.broker.get_order_status(sl.sl_order_id)
        sl.status = resp.status
        sl.last_verified_ts = now_ms()

    async def periodic_verification(self) -> None:
        """Every 60s, verify all active SL orders via REST.
        Catches silent rejections, status lag, and broker-side cancellations."""
        for entry_id, sl in self._sl_states.items():
            if sl.status == "PENDING":
                await self._verify_sl_status(entry_id)
                if sl.status not in ("PENDING", "TRIGGERED"):
                    logger.critical("sl_unexpectedly_inactive",
                                  entry_id=entry_id, sl_status=sl.status)
                    # Immediately re-place SL
                    await self._re_place_sl(entry_id)
```

**SL problem cases handled:**
1. Entry partial fill → SL qty synced immediately via `on_entry_partial_fill`
2. SL triggers while modifying entry → detected in `on_normal_exit`, position already closed
3. Cancel SL fails silently → periodic verification catches it, re-places SL
4. SL placed but rejected → caught at placement time, entry cancelled
5. DAY SL expires for overnight → converted to GTT/AMO via `on_eod_for_overnight`

#### Priority Rate Limiter

```python
class PriorityRateLimiter:
    """Token bucket with priority dispatch. Higher priority dequeues first."""

    PRIORITIES = {
        "EXIT": 0,       # highest — closing positions
        "SL": 1,         # SL placement/modification
        "MODIFY": 2,     # fill management repricing
        "NEW": 3,        # new entry orders
        "POLL": 4,       # REST status checks (lowest)
    }

    def __init__(self, rate: int = 10):
        self.rate = rate
        self.tokens = rate
        self.max_tokens = rate
        self._queue: list[tuple[int, int, asyncio.Event]] = []  # min-heap
        self._lock = asyncio.Lock()

    async def acquire(self, priority: str) -> None:
        event = asyncio.Event()
        heapq.heappush(self._queue,
                       (self.PRIORITIES[priority], id(event), event))
        await self._try_dispatch()
        await asyncio.wait_for(event.wait(), timeout=10.0)
```

**OPS math with single broker:**

Normal operation (3 strategies with pending orders, WS healthy):
- S1/S3/S5 fill modifications: 0.2 + 0.2 + 0.33 = **0.73 OPS/s**
- No REST polling when WS healthy: **0 OPS/s**
- New entry (rare, burst): 2 OPS (entry + SL)
- **Total typical: ~1-3 OPS/s.** Well within 10.

EOD flatten (5 positions): see below.

#### Fill Management Loop

```python
class OrderState:
    """Mutable process-local state. NOT a Pydantic DTO — not serialized."""
    order_id: str
    lock: asyncio.Lock             # per-order lock for atomic read-decide-act
    sequence: int = 0              # monotonic, incremented on every state change
    update_event: asyncio.Event    # set by WS demuxer when update arrives
    status: str
    original_qty: int
    filled_qty: int
    remaining_qty: int
    avg_fill_price: float
    current_limit_price: float
    modification_count: int        # max 25 per order
    placed_at_ms: int

    def apply_update(self, update: dict) -> None:
        """Called by WS demuxer UNDER LOCK."""
        self.sequence += 1
        # ... apply fields
        self.update_event.set()

async def manage_order(self, request: OrderRequest) -> OrderResult:
    order = request.resolved_order
    params = request.fill_params
    entry_id, sl_id = await self.place_entry_with_sl(order)

    if entry_id is None:
        return OrderResult(outcome="REJECTED", ...)

    state = self.order_states[entry_id]
    start = now_ms()

    while True:
        # Wait for WS update — no polling when WS healthy
        try:
            await asyncio.wait_for(
                state.update_event.wait(),
                timeout=params.reprice_interval_s
            )
            state.update_event.clear()
        except asyncio.TimeoutError:
            if not self.ws_healthy:
                await self.rate_limiter.acquire("POLL")
                await self.poll_order_status(entry_id)

        elapsed_s = (now_ms() - start) / 1000

        # Per-order lock: atomic read + decide + act
        async with state.lock:
            match state.status:
                case "TRADED":
                    await self.sl_manager.on_entry_fully_filled(entry_id)
                    return OrderResult(outcome="FILLED", ...)

                case "PART_TRADED":
                    await self.position_tracker.on_partial_fill(state)
                    # IMMEDIATELY sync SL qty
                    await self.sl_manager.on_entry_partial_fill(
                        entry_id, state.filled_qty)

                    if self.price_has_moved(state, order):
                        if elapsed_s < params.max_patience_s:
                            await self.reprice_order(state, order, params)
                        else:
                            await self.cancel_order(entry_id)
                            # SL already synced to filled_qty
                            return OrderResult(outcome="PARTIAL_FILL", ...)

                case "PENDING":
                    if elapsed_s >= params.max_patience_s:
                        await self.cancel_order(entry_id)
                        await self.sl_manager.on_normal_exit(entry_id)
                        return OrderResult(outcome="TIMEOUT", ...)
                    if elapsed_s >= params.reprice_interval_s:
                        if self.price_has_moved(state, order):
                            await self.reprice_order(state, order, params)

                case "REJECTED":
                    await self.sl_manager.on_normal_exit(entry_id)
                    return OrderResult(outcome="REJECTED", ...)

async def reprice_order(self, state, order, params):
    new_price = self.compute_new_limit_price(order, params)

    if state.modification_count >= 24:
        # Cancel-replace
        await self.rate_limiter.acquire("MODIFY")
        try:
            await self.broker.cancel_order(state.order_id)
        except OrderAlreadyFilled:
            return  # race condition: filled between check and cancel — fine
        await self.rate_limiter.acquire("MODIFY")
        new_resp = await self.broker.place_order(
            quantity=state.remaining_qty,
            price=new_price, ...)
        state.order_id = new_resp.order_id
        state.modification_count = 0
    else:
        await self.rate_limiter.acquire("MODIFY")
        try:
            await self.broker.modify_order(
                state.order_id,
                quantity=state.original_qty,  # BROKER QUIRK: total, not remaining
                price=new_price)
            state.modification_count += 1
        except OrderNotModifiable:
            # Race: order filled between check and modify
            await self.rate_limiter.acquire("POLL")
            fresh = await self.broker.get_order_status(state.order_id)
            state.apply_update(fresh)
            # Next loop iteration handles the new status

    # Post-modify verification: confirm the modify was applied
    await self.rate_limiter.acquire("POLL")
    verified = await self.broker.get_order_status(state.order_id)
    state.apply_update(verified)
```

**Broker modify qty semantics:** The architecture assumes `quantity` in modify = original total order quantity. **This MUST be validated during paper trading.** If the broker interprets it differently, the modify logic must be adjusted.

#### Fill Parameters

| Strategy | reprice_interval_s | max_patience_s | pricing_mode |
|----------|-------------------|----------------|-------------|
| S1 (ORB) | 5 | 15 | AGGRESSIVE |
| S2 (Overnight) | 10 | 30 | MIDPOINT |
| S3 (VWAP MR) | 5 | 20 | AGGRESSIVE |
| S4 (Momentum) | 30 | 120 | PASSIVE |
| S5 (Expiry Day) | 3 | 10 | AGGRESSIVE |
| S6 (Vol Premium) | 15 | 60 | PASSIVE |
| S7 (Pairs) | 10 | 45 | MIDPOINT |

Pricing modes:
- **PASSIVE:** best_ask (buy) or best_bid (sell)
- **MIDPOINT:** ceil(mid/tick)*tick for buy, floor for sell
- **AGGRESSIVE:** ask+1tick (buy) or bid-1tick (sell)

#### EOD Flatten (15:20 IST)

Batched and parallel, with IOC escalation:

```
15:20:00 — Phase 1: Cancel all pending entries (burst)
  - Cancel all pending entry orders + their SL pairs
  - For N pending: 2N cancels. At 10 OPS: N/5 seconds.

15:20:01 — Phase 2: Place all exit orders in burst
  - For each open intraday position: LIMIT exit at AGGRESSIVE price
  - Cancel the entry's server-side SL (we're now managing exit)
  - All placed in parallel.

15:20:02 — Phase 3: Parallel fill management
  - All exit fill loops run concurrently
  - Share 10 OPS via priority rate limiter (EXIT priority)

15:22:00 — Phase 4: IOC escalation
  - Any exit not filled: cancel + re-place as IOC (Immediate or Cancel)
    at best_ask + 2 ticks (buy-to-close) or best_bid - 2 ticks (sell-to-close)
  - +2 ticks (not +5): SEBI limit-order rule means our limit must
    still be a reasonable limit, not a de facto market order

15:25:00 — Phase 5: Hard deadline
  - Any remaining: Telegram CRITICAL
  - Broker auto-squares at session close (~15:30) at market price
```

**Overnight positions (S2, S6, S7):** NOT flattened. Instead, `sl_manager.on_eod_for_overnight()` converts DAY SL orders to GTT/AMO for overnight protection.

#### OMS Degraded Mode (Redis Down)

The OMS holds all `OrderState` objects in-memory (primary) and mirrors to Redis (secondary). When Redis is down:

- **Still works:** Broker WS (order updates), broker REST (place/modify/cancel), fill management loops
- **Doesn't work:** Strategy signal delivery (blocked by Redis), LASTTICK for repricing
- **Mitigations:** Server-side SLs protect all positions. For repricing, fall back to broker option chain API directly (skip Redis cache). New entries blocked (no signals arriving).
- **Recovery:** On Redis reconnect, OMS publishes accumulated state changes.

#### Global Kill Switch

```python
async def global_kill(self) -> None:
    """
    One command: cancel everything, flatten everything, NOW.
    Triggered by: CLI command, Telegram command, or risk manager halt.
    """
    logger.critical("GLOBAL_KILL_ACTIVATED")

    # 1. Block all new signals immediately
    await redis.set("HALT:global", "1")

    # 2. Cancel ALL pending orders (entries and their SLs)
    for order_id, state in self.order_states.items():
        if state.status in ("PENDING", "PART_TRADED"):
            try:
                await self.broker.cancel_order(order_id)
            except Exception:
                pass  # best effort

    # 3. Flatten ALL positions with AGGRESSIVE pricing
    positions = self.position_tracker.get_all_open()
    for pos in positions:
        exit_order = self.build_exit_order(pos, urgency="URGENT")
        await self.rate_limiter.acquire("EXIT")
        await self.broker.place_order(exit_order)

    # 4. Alert
    await self.telegram.send(CRITICAL, "GLOBAL KILL: all orders cancelled, flattening all")
```

Invocable via:
- CLI: `python -m live.cli kill`
- Redis: `SET HALT:global 1` (any process can trigger it)
- Telegram: `/kill` command to bot (authenticated)

Resuming after a global kill requires manual intervention: `DEL HALT:global` in Redis + system restart.

#### State
| What | Where | Lifecycle |
|------|-------|-----------|
| OrderState objects | In-memory (primary) + Redis `ORDER:*` (mirror) | Created on placement, removed on terminal |
| SL pairings (SLState) | In-memory via SLLifecycleManager | Entry → SL mapping |
| Rate limiter | In-memory | Session |
| Daily order count | Redis `OMS:daily_count` | Reset 08:30 |
| WS connection health | In-memory flag | Updated on WS events |
| Placed signal_ids (dedup) | In-memory set | Session |

---

### Component 6: Risk Management Layer

#### Responsibility
- Pre-trade validation gate (all checks must pass before order placement)
- Post-trade continuous monitoring (drawdown, margin, VIX regime)
- Kill conditions per strategy + portfolio halt
- Broker position check to prevent ghost positions

#### Pre-Trade Checks

| Check | Rule |
|-------|------|
| Position size | qty × price ≤ allocation × 1.2 |
| Daily trade count | <50/strategy, <200/portfolio |
| Margin | required ≤ available |
| Cost hurdle | expected_edge_bps > 2 × cost_bps |
| Correlation | S1 active → block S3 |
| Strategy not killed | `KILLED:{sid}` not set |
| Portfolio not halted | `HALT:global` not set |
| Session hours | 09:15-15:25 IST |
| Daily order limit | count < 4,500 (buffer below 5,000) |
| **Broker position check** | No ghost position for this instrument |

**Broker position check:** Before placing ANY order, query broker positions API for this instrument_id. If broker shows a position that local state doesn't know about, trigger reconciliation and block the order. Costs 1 OPS per trade but prevents the double-position catastrophe where a lost fill notification leads to a second entry.

#### Post-Trade Monitoring

Separate async process. Every 10 seconds:
- Rolling drawdown per strategy
- Portfolio drawdown (>10% → halt all)
- Margin utilization (>80% → warning)
- VIX regime (>25 → suppress S1, S5)

**Post-order reconciliation:** After every order placement, a timer fires after `max_patience_s + 5s`. If `on_fill()` was never called but broker REST shows TRADED, force-sync positions from broker. Additionally, reconciliation runs **every 5 minutes** during market hours (1 API call, negligible OPS cost).

#### Kill Conditions

| Strategy | Metric | Threshold | Action |
|----------|--------|-----------|--------|
| S1 | After-cost Sharpe | < 0.5 over 60 days | stop_new_entries |
| S2 | Consecutive negative PnL days | 40 | stop_new_entries |
| S3 | After-cost Sharpe | < 0.3 over 60 days | stop_new_entries |
| S4 | Underperform NIFTY50 | > 15% over 12 months | stop_new_entries |
| S5 | Consecutive expiry-day losses | 3 | stop_new_entries |
| S6 | Monthly drawdown | -15% of allocation | **flatten_immediately** |
| S7 | After-cost Sharpe | < 0.3 over 6 months | stop_new_entries |
| **Portfolio** | Total drawdown | -10% | **halt_all** |

- `stop_new_entries`: Set `KILLED:{sid}`. Open positions exit via stops or normal logic. Capital redistributed at next daily recomputation.
- `flatten_immediately`: Kill + urgent exit all positions (S6 short options = unlimited risk).
- `halt_all`: Global kill switch activated. Manual resume required.

---

### Component 7: Position & PnL Tracker

#### Responsibility
- Real-time position state (sole source of truth locally)
- Realized and unrealized PnL
- **Sole writer to DuckDB** — all other components read via Redis cache
- Reconciliation with broker
- Greeks for option positions
- State restoration on startup

#### Interface

```python
class Position(pydantic.BaseModel):
    strategy_id: str
    instrument_id: str
    trading_symbol: str
    underlying: str
    instrument_type: str
    direction: Literal["LONG", "SHORT"]
    quantity: int
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    entry_ts: int
    last_update_ts: int
    strike: float | None
    expiry: date | None
    option_type: str | None
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    order_ids: list[str]
    sl_order_id: str | None

class PositionTracker:
    positions: dict[str, Position]    # instrument_id → Position

    async def on_fill(self, fill: OrderResult) -> None: ...
    async def on_partial_fill(self, state: OrderState) -> None: ...
    def mark_to_market(self) -> None: ...  # every 5s
    def get_strategy_position(self, sid: str) -> str: ...  # "FLAT"/"LONG"/"SHORT"
    async def reconcile_with_broker(self) -> list[Discrepancy]: ...
    async def force_sync_from_broker(self) -> None: ...

    # DuckDB access (sole writer)
    def write_trade(self, trade: TradeRecord) -> None: ...
    def write_daily_pnl(self, sid: str, d: date, pnl: float) -> None: ...
    def get_daily_returns(self, sid: str, lookback: int) -> list[float]: ...
```

#### DuckDB Single-Writer Pattern

DuckDB does not support concurrent writes from multiple processes.

```
Position Tracker (sole writer)
  ├── Tables: daily_returns, trades, realized_pnl, reconciliation
  ├── Writes: on every fill, daily EOD summary
  ├── Reads: on startup (restore), on request (daily returns for ERC)
  └── Caches to Redis:
        CACHE:daily_returns:{sid}  — JSON list, updated daily
        CACHE:strategy_pnl:{sid}   — current PnL, updated on fill

Other components (read from Redis cache, never DuckDB directly):
  ├── Capital Allocator reads CACHE:daily_returns:{sid}
  └── Risk Manager reads CACHE:strategy_pnl:{sid}
```

#### Mark-to-Market

| Instrument | Long Mark | Short Mark | Source |
|------------|----------|------------|--------|
| Options | Best bid (liquidation) | Best ask (liquidation) | Chain snapshot |
| Futures | LTP | LTP | LASTTICK |
| Equity | LTP | LTP | LASTTICK |

Display PnL uses mid-price. Risk PnL uses liquidation mark (conservative).

#### Greeks

Computed locally every 30s using Black-76 + IV from option chain. Inputs: spot (LASTTICK), strike/expiry/type (from Position), IV (from chain snapshot or BSM inversion), risk-free rate (6.5%, updated quarterly in config).

**Delta exposure alert:** Portfolio net delta > ±50 NIFTY lot equivalents → WARNING.

#### Reconciliation

| When | Type | What |
|------|------|------|
| 09:16 | Full | Startup, after market open |
| Every 5 min | Quick | instrument_id + quantity check |
| Post-order | Targeted | Timer fires if fill not received within patience+5s |
| 15:31 | Full | EOD |

On discrepancy: broker is source of truth. Force sync. Telegram CRITICAL.

#### Startup Recovery

1. Query broker positions API → all current positions
2. Read Redis `POSITION:saved:{sid}` → previous session state
3. Merge: broker authoritative for qty/price, Redis adds strategy attribution
4. Unknown positions → "ORPHAN" strategy + Telegram CRITICAL

---

### Component 8: Audit & Compliance Logger

#### Responsibility
- Structured JSON logging (structlog) of every tradable event
- 5-year S3 retention (SEBI)
- Signal context for decision reproduction

#### Event Schema

```python
class AuditEvent(pydantic.BaseModel):
    event_id: str                 # UUID
    event_type: str               # see list below
    timestamp_utc: str            # ISO 8601
    timestamp_ist: str            # ISO 8601 +05:30
    strategy_id: str | None
    signal_id: str | None
    order_id: str | None
    instrument_id: str | None
    trading_symbol: str | None
    payload: dict                 # event-specific data
    system_version: str           # git SHA
    hostname: str
    process_id: int
```

Event types: SIGNAL_GENERATED, SIGNAL_REJECTED, SIGNAL_DEDUPED, ORDER_PLACED, ORDER_MODIFIED, ORDER_CANCELLED, ORDER_FILLED, ORDER_PARTIAL_FILL, ORDER_REJECTED, SL_PLACED, SL_MODIFIED, SL_TRIGGERED, SL_CANCELLED, SL_VERIFICATION_FAILED, POSITION_OPENED, POSITION_CLOSED, POSITION_RECONCILED, RISK_CHECK_FAILED, KILL_TRIGGERED, GLOBAL_KILL, SYSTEM_STARTUP, SYSTEM_SHUTDOWN, WS_DISCONNECT, WS_RECONNECT.

**Signal context:** Every SIGNAL_GENERATED includes: spot_price, VIX, all indicator values, bar OHLCV, all active strategy parameters. ~2KB per signal.

#### Log Destination

```
Local:  /var/log/trading/audit/{date}/events.jsonl  (JSON Lines)
S3:     s3://trading-audit-logs/{year}/{month}/{date}/events.jsonl.gz
  Uploaded daily 16:00 IST
  Lifecycle: Standard → IA (90d) → Glacier (1yr)
  Retention: 5 years
```

DuckDB schema versioned via `schema_version` integer in a `metadata` table. Migrations are forward-only Python scripts in `live/migrations/`.

---

### Component 9: Monitoring & Alerting

#### Responsibility
- Prometheus metrics for dashboards
- Telegram notifications for actionable events
- **Telegram is notification-only** — order execution is fully automated, Telegram sends status updates

#### Prometheus Metrics

```python
# Orders
orders_placed_total       = Counter("orders_placed_total", ["strategy"])
orders_filled_total       = Counter("orders_filled_total", ["strategy"])
orders_rejected_total     = Counter("orders_rejected_total", ["strategy", "reason"])
order_to_fill_latency_ms  = Histogram("order_to_fill_latency_ms", ["strategy"],
                                       buckets=[100, 250, 500, 1000, 2000, 5000])
# PnL
realized_pnl_inr          = Gauge("realized_pnl_inr", ["strategy"])
unrealized_pnl_inr        = Gauge("unrealized_pnl_inr", ["strategy"])
drawdown_pct              = Gauge("drawdown_pct", ["strategy"])
portfolio_drawdown_pct    = Gauge("portfolio_drawdown_pct")

# Positions + Risk
open_positions            = Gauge("open_positions", ["strategy"])
delta_exposure            = Gauge("delta_exposure")
margin_utilized_pct       = Gauge("margin_utilized_pct")

# System
tick_latency_ms           = Histogram("tick_latency_ms", buckets=[10, 50, 100, 500])
ws_connected              = Gauge("ws_connected", ["channel"])  # data, orders
strategy_status           = Gauge("strategy_status", ["strategy"])
rate_limit_tokens         = Gauge("rate_limit_tokens")
daily_order_count         = Gauge("daily_order_count")
sl_active_count           = Gauge("sl_active_count")
```

#### Telegram Notifications

| Level | Events |
|-------|--------|
| CRITICAL | WS disconnect, global kill, position discrepancy, SL verification failed, EOD flatten timeout, exchange silence |
| WARNING | Margin >80%, drawdown approaching, OPS pressure, order rejection, daily orders >4500 |
| INFO | Daily PnL summary, strategy killed, system start/stop, order placed/filled |

Rate-limited: 1 msg/5s. CRITICAL bypasses limit.

#### Grafana Dashboards

1. **Live Trading:** PnL curve, drawdown gauge, margin, per-strategy PnL, position table, order flow
2. **Execution Quality:** Fill latency, slippage, modifications, spread at execution vs signal
3. **System Health:** WS status, tick latency, process uptime, Redis memory
4. **Risk:** Rolling Sharpe, kill condition proximity, delta exposure, SL status

#### Monitoring Self-Check

Separate cron job (60s). Checks Prometheus + Grafana health. If both down: Telegram CRITICAL via direct API. **Trading does NOT halt on monitoring failure** — risk manager runs independently.

---

## 3. CROSS-CUTTING CONCERNS

### A. Time Management

**Internal:** UTC epoch milliseconds (`int64`) everywhere. Conversion to IST only at display boundaries.

**Time synchronization:** EC2 instance runs `chrony` synced to AWS NTP (`169.254.169.123`). Alert if clock drift exceeds 100ms. Exchange timestamps in ticks are authoritative for ordering — local timestamps are for latency measurement only.

**Session phases:**
- PRE_OPEN (09:00-09:08): Ingester active, strategies warming up, no orders
- CONTINUOUS (09:15-15:30): Trading active from 09:20 (skip first 5 min)
- CLOSING (15:30-15:40): Monitor only, EOD flatten should be complete
- CLOSED: Overnight positions managed (S2, S6, S7)

**Holiday calendar:** Static JSON `config/nse_holidays_2026.json` + runtime verification:

```python
class HolidayCalendar:
    holidays: set[date]

    async def verify_exchange_open(self) -> bool:
        """Called at 09:10 before strategies start."""
        # 1. Check static calendar
        if today in self.holidays:
            return False
        # 2. Check broker market status API (if available)
        try:
            status = await broker.get_market_status()
            if status != "OPEN":
                return False
        except Exception:
            pass  # API unreliable, continue
        # 3. Check if ticks are flowing (after 09:15)
        # All-instrument-silence detector handles this
        return True

    def next_weekly_expiry(self, d: date) -> date:
        """Next Tuesday. If Tue is holiday, previous trading day."""
        ...
    def next_monthly_expiry(self, d: date) -> date:
        """Last Thursday of month. If holiday, previous trading day."""
        ...
```

### B. Configuration

```
config/
├── system.yaml             # non-secret system config
├── strategies.yaml         # per-strategy params + fill params
├── risk.yaml               # risk limits, kill conditions
├── broker.yaml             # broker endpoints (not credentials)
├── nse_holidays_2026.json
└── .env                    # secrets (NOT in git)
```

**Secrets:** `.env` loaded via `python-dotenv`. Contains broker API keys, Telegram bot token, AWS credentials. On EC2, prefer IAM roles for S3 access (no static AWS keys).

**Hot-reload:** Strategy parameters stored in Redis `CONFIG:strategy:{sid}`. Change flow: edit YAML → `python -m live.cli reload-config` → Redis updated → strategies pick up on next bar close.

Risk limits NOT hot-reloadable — require restart (deliberate friction for safety-critical params).

### C. Startup & Shutdown

#### Startup (08:25 → 09:20 IST)

```
08:25  Phase 1: Infrastructure
  ├─ Verify Redis Sentinel (primary + replica running)
  ├─ Verify DuckDB accessible
  ├─ Verify S3 credentials (or IAM role)
  ├─ Verify static IP attached (AWS metadata)
  ├─ Verify chrony sync (clock drift < 100ms)
  └─ Load config files

08:30  Phase 2: Authentication
  ├─ Authenticate broker → token in Redis
  └─ Verify: call profile/fund API

08:35  Phase 3: Data Load
  ├─ Download broker instrument CSV → parse → build InstrumentMaster
  │   (Tuesday + fallback → disable S5, see §2 Instrument Master)
  ├─ Load holiday calendar
  └─ Verify today is trading day

08:40  Phase 4: Position Recovery
  ├─ Query broker positions API → overnight positions
  ├─ Reconcile with Redis saved state
  ├─ For overnight positions: verify SL orders still active (GTT)
  └─ Initialize PositionTracker

08:50  Phase 5: Data Feed
  ├─ Connect broker data WS
  ├─ Subscribe instruments
  ├─ Verify ticks flowing (wait up to 60s)
  └─ Connect broker order update WS

08:55  Phase 6: Components
  ├─ Start option chain poller (background, 5s)
  ├─ Compute ERC weights for today
  ├─ Start risk monitor
  ├─ Start Prometheus metrics server
  ├─ Start audit logger
  ├─ Start SL lifecycle periodic verification (60s)
  └─ Start monitoring watchdog

09:10  Phase 7: Exchange Verification
  ├─ Holiday calendar check
  ├─ Broker market status check
  └─ If closed: abort, Telegram INFO

09:10  Phase 8: Strategy Processes
  ├─ Create Redis consumer groups (XGROUP CREATE ... $)
  ├─ Start S1..S7 (enabled)
  ├─ Each: connect streams, restore state
  └─ Strategies SUPPRESSED until 09:20

09:20  Phase 9: Go Live
  ├─ Un-suppress strategies
  ├─ OMS begins accepting orders
  └─ Telegram INFO: "System live. N strategies active. M positions carried."
```

Each phase has a health check and 5-minute timeout. Total startup budget: 55 minutes. Not ready by 09:20 → DEGRADED (manage positions only, no new trades).

**Dependency graph:**
```
Redis ──→ Data Ingester ──→ Consumer Groups ──→ Strategies
  │                                                 │
  └──→ Position Recovery ──→ OMS ──→ Signal Router ←┘
                              │
                     SL Verification
```

#### Shutdown

```
Phase 1: Block new entries (immediate) — SET HALT:no_new_entries
Phase 2: Flatten intraday (EOD sequence, up to 5 min)
Phase 3: Convert DAY SLs to GTT for overnight positions
Phase 4: Save state to Redis (30s)
Phase 5: Disconnect WS, logout broker (SEBI: daily logout)
Phase 6: Flush WAL + Parquet + audit to S3 (30s)
Phase 7: SIGTERM strategy processes → 5s → SIGKILL
Phase 8: Telegram INFO + exit
```

### D. State Recovery

**Redis Sentinel** protects against Redis process crashes (<1s failover). Same-host limitation acknowledged — EC2 failure kills both primary and replica. Protection in that case: server-side SLs + broker auto-square.

| Lost State (Redis crash) | Recovery |
|--------------------------|----------|
| LASTTICK | Auto-rebuilt from ticks within seconds |
| POSITION:live | Rebuilt from broker positions API |
| ORDER state | Query broker orders API |
| CONFIG | Reload from YAML |
| ALLOC/KELLY | Recompute from DuckDB (via Position Tracker) |

**OMS degraded mode** covers the gap: fill management continues via broker WS + REST. New entries blocked. Server-side SLs active.

**DuckDB:** RDB-style backup to S3 nightly at 16:30 IST. RPO: 1 day (trading data is reconstructible from broker records). RTO: <5 minutes (restore from S3 + replay today's fills from audit log).

---

## 4. TESTING & SECURITY

### Testing Strategy

| Level | What | How |
|-------|------|-----|
| Unit | Pydantic models, bar builder, cost model, strike resolution | pytest, property-based tests (hypothesis) for edge cases |
| Integration | Signal → resolve → risk gate → OMS mock | pytest with mock broker adapter |
| Replay | WAL/Parquet replay through full pipeline | Replay framework: read historical ticks, feed through strategies, compare signals |
| Paper trading | Full system, real market data, mock execution | `PAPER_MODE=true` in config → OMS logs orders but doesn't call broker API |
| Chaos | Redis kill mid-session, WS drop mid-order, duplicate WS messages | Scripted chaos tests run weekly during paper trading phase |
| OMS state machine | All order status transitions, partial fills, race conditions | Property tests: fuzz OrderState transitions, verify invariants |

**Paper vs prod separation:** Environment flag `TRADING_ENV=paper|live` controls:
- `paper`: OMS simulates fills (random delay 50-500ms, 90% fill rate). No real API calls. Uses paper broker credentials.
- `live`: Real broker API. Real money. Real consequences.

Both share the same code path — only the `BrokerAdapter` implementation differs.

### Security

| Concern | Approach |
|---------|----------|
| Secrets | `.env` file, 600 permissions. IAM roles for S3 (no static AWS keys on disk). |
| Network | EC2 security group: inbound SSH (IP-restricted) + Grafana (IP-restricted) only. Redis on localhost only (not exposed). |
| Broker API | TLS only. No TLS pinning in v1 (broker certs rotate). Validate cert chain. |
| Authentication | Broker tokens: 24hr validity, rotated daily. Stored in Redis (localhost only). |
| Access control | `reload-config` and `kill` CLI commands require SSH access to the EC2 instance. Telegram `/kill` requires bot authentication. |
| Logging | No secrets in logs. Audit log contains order details but not API tokens. |

### Runbooks

| Scenario | Runbook |
|----------|---------|
| System won't start | Check Redis, broker auth, instrument CSV. Logs in `/var/log/trading/`. |
| WS disconnected mid-session | Auto-reconnect. If persistent: check broker status page. Manual: restart data ingester. |
| Position discrepancy | Telegram CRITICAL. System auto-syncs from broker. Review audit log for missed fills. |
| Kill condition fires | Telegram INFO. Strategy stops new entries. Review: `python -m live.cli strategy-status`. Resume: `DEL KILLED:{sid}` in Redis after review. |
| Global kill fired | Everything cancelled + flattened. Review cause. Resume: `DEL HALT:global` + full system restart. |
| Post-incident | Reconcile audit log with broker contract notes. File incident report. |

---

## 5. OPEN QUESTIONS & MUST-PROTOTYPE

### Must Validate During Paper Trading

1. **Broker modify qty semantics.** Place 130, partial 65, modify with qty=130. Does remaining = 65 or 130? System-breaking if wrong.
2. **Broker option chain response schema.** Verify Greeks are present. If not, validate local BS computation.
3. **GTT/Forever order support.** Verify overnight SL via GTT works as expected. Verify AMO as fallback.
4. **SL trigger behavior when system is offline.** Confirm SL-Limit triggers and fills on broker infra without our WS connected.
5. **Redis Streams throughput.** Measure XADD/XREADGROUP with 7 consumer groups at 500+ ticks/s burst.
6. **Option chain API P99 latency.** If >500ms, the on-demand call needs a tighter timeout or hybrid approach (use 1s-old cache + LTP delta sanity check as alternative).
7. **0-DTE spread behavior.** Measure actual NIFTY weekly option spreads on 4 consecutive Tuesdays.
8. **End-to-end latency.** Signal fire → order on exchange. Target: P99 < 1s.

### Architecture Decisions and Rationale

| Decision | Why |
|----------|-----|
| Single broker | Eliminates cross-broker symbol mapping, split auth, position tracking complexity. Failover to second broker is a v2 concern. |
| Redis Streams (not pub/sub) | Persistent, backpressure-aware, consumer groups recover after disconnect |
| On-demand option chain at signal | 200ms << 5-10s repricing from stale data |
| Mandatory server-side SL | Only protection during system outage |
| SL as first-class lifecycle object | SL assumed correct = SL wrong. Verify constantly. |
| Priority rate limiter | Exit orders must never be starved |
| asyncio.Lock (not threading.Lock) | threading.Lock blocks entire event loop |
| DuckDB single-writer | DuckDB doesn't support concurrent writes |
| Same-host Redis Sentinel | Process-level HA. Host-level HA via SL + broker auto-square. |
| Signal dedup in router | Defense-in-depth against strategy bugs |
| Position-aware allocation | Prevents double positions from duplicate signals |
| Post-modify REST verification | Brokers silently reject or partially apply modifications |
| Tick WAL | Parquet buffer survives process crash |

### Scaling Triggers

| Trigger | Action |
|---------|--------|
| 10 OPS binding constraint | Switch to Upstox (50 OPS) or register algos for higher limits |
| WS instrument limit reached | Add second WS connection or switch to broker with higher limit |
| Single EC2 CPU saturated (7 strategies + Redis) | Split: Redis to managed service, strategies to second instance |
| ₹1Cr+ capital | Redis on separate instance. Consider hot-standby EC2. |
| ₹5Cr+ capital | Multi-AZ deployment. True cross-host HA. |

### Implementation Phases

```
Phase 1 (Week 1-2): Skeleton
  - Pydantic models for all interfaces
  - Redis Sentinel setup
  - BrokerAdapter (auth + place + modify + cancel + status + SL)
  - Broker WS connection (ticks + order updates)
  - InstrumentMaster loader

Phase 2 (Week 3-4): Core Loop
  - On-demand option chain resolution
  - OMS with fill management + per-order locks
  - SL lifecycle manager
  - Priority rate limiter
  - Position tracker (DuckDB single-writer)

Phase 3 (Week 5-6): Strategy Integration
  - LiveStrategy base + BarBuilder
  - Redis Streams consumer groups
  - Port S1 (ORB) as first live strategy
  - Signal dedup + position-aware allocator
  - Pre-trade risk gate (incl. broker position check)
  - Audit logger

Phase 4 (Week 7-8): Safety
  - Post-order reconciliation
  - SL periodic verification
  - EOD flatten (batch + parallel + IOC)
  - Overnight SL conversion (GTT/AMO)
  - OMS degraded mode
  - Global kill switch
  - Kill conditions

Phase 5 (Week 9-10): Monitoring
  - Prometheus + Grafana dashboards
  - Telegram notifications
  - Exchange open verification
  - Monitoring self-check

Phase 6 (Week 11-12): All Strategies + Hardening
  - Port S2-S7
  - ERC allocation + Kelly ramp
  - Paper trading (full system, 2+ weeks)
  - Chaos testing
  - Runbook validation
```

---

## 6. REGULATORY NOTES

This architecture addresses known SEBI algo trading requirements (static IP, limit orders only, daily logout, 5-year audit trail, server in India). Actual regulatory obligations depend on entity classification and broker arrangements. **Confirm with compliance counsel before deploying real capital.** The architecture provides the technical infrastructure for compliance but does not constitute legal advice.

---

*This document specifies every interface, state model, failure mode, and recovery path for the v1 live trading system. Every order is single-leg. A single broker handles both data and execution. Server-side stop-losses ensure no position is ever unprotected. The system is designed for ₹50L and scales to ₹10Cr with the triggers documented above.*
