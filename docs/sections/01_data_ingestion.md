## Component 1: Data Ingestion Layer

> **Multi-Account Note:** Data ingestion is SHARED across all accounts. Market data is not account-specific. A single Data Ingester instance serves all accounts. No downstream component (Signal Router, OMS, Position Tracker) affects what data is ingested or how it is stored. Adding or removing trading accounts requires zero changes to this component.

### Responsibility

- Subscribe to broker's tick-by-tick websocket for real-time market data
- Normalize ticks into a canonical `Tick` model
- Distribute ticks to strategy processes via Redis Streams (persistent, backpressure-aware)
- Record raw ticks via write-ahead log (WAL) + Parquet with 5-second flush
- **Does NOT** interpret ticks, compute indicators, or make trading decisions

### Interface Contracts

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

**Idempotent replay:** The WAL contains a monotonic sequence number per tick. The Parquet writer tracks the last written sequence. On replay, ticks with sequence <= last_written are skipped. Partial Parquet writes (crash mid-flush) are detected by checking the footer — incomplete files are discarded and rewritten from WAL.

### State

| What | Where | Lifecycle |
|------|-------|-----------|
| WS connection handle | In-memory | Process lifetime |
| Subscription list | Config -> in-memory | Session |
| Tick WAL file | Disk (mmap'd) | Survives crashes. New file per day. |
| Tick buffer (pre-Parquet) | In-memory deque, max 5s | Flushed every 5s |
| Last tick per symbol | Redis `LASTTICK:{symbol}` (hash) | Overwritten per tick |
| Connection health | Redis `HEALTH:data_ingester` | Heartbeat every 5s with TTL 15s |

**No per-account state in this component — all state is shared.** The subscription list, WAL, Parquet archive, Redis Streams, and `LASTTICK` hashes are account-agnostic. Whether the system trades one account or ten, the Data Ingester maintains the exact same state footprint.

### Concurrency Model

Single async process: `asyncio` + `websockets`.

- `asyncio.Task`: WS reader (receives ticks, writes WAL, publishes to Redis Stream, updates LASTTICK)
- `asyncio.Task`: Parquet flush loop (5s)
- `asyncio.Task`: Heartbeat (5s)

I/O-bound — a single async loop handles thousands of ticks/sec.

### Failure Modes

| Failure | Detection | Recovery |
|---------|-----------|----------|
| WS disconnect | `ConnectionClosed` exception | Exponential backoff: 1s, 2s, 4s, 8s, max 30s. Re-subscribe. |
| WS stale (connected, no ticks) | Watchdog: no tick on ANY instrument for 10s | Force close + reconnect |
| **All instruments silent simultaneously** | Zero ticks on all streams for 15s | Exchange halt/closure. Telegram CRITICAL. Suppress new entries. Existing positions protected by server-side SL. |
| Redis down | `aioredis` ConnectionError | Retry via Sentinel. WAL continues locally. Strategies paused. OMS enters degraded mode. |
| Process crash | systemd detects exit | Restart. WAL survives. Replay WAL -> Parquet. |

### Multi-Account Impact

Data ingestion is completely account-agnostic. This is a deliberate architectural property, not an accident.

**Why data ingestion has no account awareness:**

1. **Market data is universal.** The NIFTY50 index tick at 10:00:00.123 is the same number regardless of which account is trading on it. There is no concept of "Account A's NIFTY tick" versus "Account B's NIFTY tick." The exchange publishes one price stream per instrument, and we receive one copy.

2. **Subscriptions are instrument-driven, not account-driven.** The websocket subscription list is determined by which instruments the strategies need — NIFTY50 spot, BANKNIFTY spot, relevant futures and option chains. Adding a second or third trading account does not change which instruments we need to observe.

3. **Redis Streams are broadcast by design.** Each strategy process has its own consumer group on the relevant streams. Consumer groups are identified by strategy, not by account. Strategy S1 reads from `STREAM:TICK:SPOT` whether it is generating signals for one account or five. The tick fan-out to multiple accounts happens downstream in the Signal Router, not here.

4. **Storage is shared.** The WAL and Parquet archive record raw market data for audit and replay. These files serve all accounts equally. Replaying a day's ticks for backtesting or reconciliation does not require knowing which accounts were active.

**Scaling implications:** Adding accounts increases load on downstream components (Signal Router must route to more OMS instances, Position Tracker must maintain more account ledgers). It does NOT increase load on Data Ingestion. The tick rate, WAL write rate, Parquet flush rate, and Redis publish rate are all functions of instrument count and market activity, never of account count.

**Configuration boundary:** The Data Ingester's only configuration inputs are the broker API credentials (for the websocket connection) and the instrument subscription list. Neither of these is per-account. The broker credentials are for the data feed, which is a single connection regardless of how many trading accounts exist downstream.
