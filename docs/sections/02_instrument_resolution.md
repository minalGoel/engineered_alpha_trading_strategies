## Component 2: Instrument Resolution Layer

> **v3 shared resolution:** Instrument resolution is SHARED. The same `security_id` is used for all accounts. A signal produces one resolved instrument, which is then replicated to N accounts. The resolution pipeline runs ONCE per signal, not once per account.

---

### Responsibility

- Translate abstract strategy signals ("go long NIFTY", "sell PE") into a concrete broker instrument ID with a limit price
- Maintain broker instrument master (daily CSV) as canonical mapping
- **Always make an on-demand option chain API call when a signal fires** for order pricing
- Maintain background 5s polling for monitoring/Greeks/display only
- Validate liquidity and spread
- **Does NOT** place orders, manage positions, or size trades

---

### Instrument Master

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

The InstrumentMaster is loaded once and shared across all accounts. Since instrument IDs are exchange-level identifiers (not broker-account-level), a single master serves the entire system regardless of how many accounts are active.

---

### Option Chain — Dual Mode

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

---

### Resolution Pipeline

#### Signal and Resolved Order Models

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
    legs: list[LegSpec] | None    # v2 extensibility: multi-leg orders (see below)

class ResolvedOrder(pydantic.BaseModel):
    signal: StrategySignal
    instrument_id: str            # broker instrument ID
    trading_symbol: str
    exchange_segment: str
    transaction_type: Literal["BUY", "SELL"]
    product_type: Literal["INTRADAY", "CNC", "MARGIN"]  # v1 restriction: only these 3 product types. Dhan also supports MTF, CO, BO.
    quantity: int                  # in units (lot_size x lots)
    limit_price: float
    tick_size: float
    lot_size: int
    freeze_qty: int
    spread_bps: float
    cost_estimate: float          # estimated RT cost in INR
    sl_trigger_price: float       # for mandatory server-side SL
    sl_limit_price: float         # SL limit = trigger +/- 2 ticks
    sl_validity: Literal["DAY", "GTT"]  # "DAY" for intraday, "GTT" for overnight (Forever Order)
```

#### Algorithm

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
   e. LIQUIDITY check: if bid==0 or ask==0, try +/-1 strike ITM.
      Still zero -> REJECT.
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
      sl_limit = trigger +/- 2 ticks
   j. DETERMINE SL validity:
      Intraday strategies (S1,S3,S5): "DAY"
      Overnight/multi-day (S2,S6,S7): "GTT" (Good Till Triggered)
   k. LOOKUP instrument_id from InstrumentMaster
   l. ESTIMATE cost via cost_model
   m. RETURN ResolvedOrder

4. FOR futures: same flow, no strike resolution
5. FOR equities: direct lookup, no chain call needed
```

#### 0-DTE Guards (S5 on Tuesday)

Reject `bid_qty < 100` lots. Wider spread threshold (1000 bps vs 500 bps). Force `urgency=URGENT`.

**Tuesday CSV fallback interaction:** If the instrument master fell back to yesterday's CSV (download failure), S5 is disabled for the entire day. 0-DTE weekly contracts that expired or were listed overnight will not be in yesterday's file, making resolution unreliable. The system logs a Telegram WARNING at 08:40 when this fallback activates and S5 disablement takes effect.

---

### Multi-Leg Extensibility

**v2 extensibility note:** Signal can carry `legs: list[LegSpec]` for multi-leg orders. v1: always single leg.

When `legs` is populated, each `LegSpec` defines an independent strike/option_type/direction, and the resolution pipeline runs steps 3a-3m for each leg, producing a `list[ResolvedOrder]` instead of a single `ResolvedOrder`. Multi-leg resolution still counts as a single signal event — the on-demand option chain call in step 3c is made once and reused across all legs.

```python
class LegSpec(pydantic.BaseModel):
    leg_id: int                    # sequential: 0, 1, 2, ...
    instrument_hint: Literal["CE", "PE", "FUT", "EQ"]
    direction: Literal["LONG", "SHORT"]
    strike_offset: int             # ATM offset in strike steps (0 = ATM, +1 = 1 strike OTM, etc.)
    ratio: int                     # lot multiplier (1 for standard, 2 for ratio spreads)
```

---

### Multi-Account Impact

Instrument resolution is a **shared, account-agnostic** operation. The key insight is that instrument IDs, option chains, spot prices, and spread calculations are exchange-level properties — they do not vary by trading account.

**Execution flow for N accounts:**

```
StrategySignal
    |
    v
[Instrument Resolution]  <-- runs ONCE
    |
    v
ResolvedOrder (single instance)
    |
    +---> Account 1: size, place order (OMS)
    +---> Account 2: size, place order (OMS)
    +---> Account N: size, place order (OMS)
```

What varies per account is downstream of resolution:

| Concern | Shared (resolution) | Per-account (OMS/sizing) |
|---------|---------------------|--------------------------|
| Instrument ID | Yes | -- |
| Limit price | Yes | -- |
| Spread validation | Yes | -- |
| SL trigger/limit prices | Yes | -- |
| Position quantity | -- | Yes (capital-based sizing) |
| Order placement | -- | Yes (per-account API call) |
| Margin check | -- | Yes (per-account balance) |
| Freeze qty slicing | -- | Yes (per-account quantity may differ) |

This design avoids redundant API calls. A single on-demand option chain call (200-400ms) serves all accounts. If resolution ran per-account, N accounts would require N identical API calls adding N * 200-400ms of latency and N times the rate limit consumption — wasteful since the exchange state is identical for all of them.

---

### State

| What | Where | Lifecycle |
|------|-------|-----------|
| InstrumentMaster | In-memory | Loaded 08:30, immutable for session |
| Background chain snapshots | In-memory + Redis `CHAIN:*` | Updated 5s (monitoring only) |
| Spot prices | Redis `LASTTICK:*` | Updated per tick |

All state entries are shared across accounts. There is no per-account state within the resolution layer.

---

### Concurrency

Async tasks within the Signal Router process. `resolve()` is `async def` — the on-demand API call is awaited, not blocking.

If two signals arrive simultaneously (e.g., S1 and S3 both fire on the same bar), each gets its own `resolve()` coroutine. The option chain API calls are independent and can execute concurrently via `asyncio.gather`. The InstrumentMaster is read-only after 08:30, so no locking is required for instrument lookups.

In multi-account mode, the resolved result is passed by reference to the per-account OMS dispatch. Resolution does not need to be aware of how many accounts exist or how they are configured.

---

### Failure Modes

| Failure | Detection | Impact |
|---------|-----------|--------|
| Instrument CSV download fails 3x | Retry exhaustion at 08:40 | Use yesterday's CSV. Disable S5 on Tuesdays. Telegram WARNING. |
| On-demand chain API fails + retry fails | Two consecutive HTTP errors | Fall back to cached snapshot if <2s old. If stale: REJECT signal. Signal not sent to any account. |
| Instrument not found in master | `lookup_option` returns None | REJECT signal. Log with full signal details. |
| Spread exceeds threshold | `spread_bps > 500` (or 1000 for 0-DTE) | REJECT signal. Log spread value. |
| Zero liquidity (bid=0 or ask=0) | Liquidity check step 3e | Try adjacent strike. If still zero: REJECT. |
| Spot price missing from Redis | `LASTTICK` key absent | Cannot compute ATM strike. REJECT signal. |

Signal rejection at the resolution layer is absolute — if resolution rejects, no account receives the order. This is correct behavior: if the instrument cannot be resolved reliably, it should not be traded on any account.
