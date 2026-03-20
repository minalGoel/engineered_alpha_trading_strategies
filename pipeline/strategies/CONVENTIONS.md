# Strategy Implementation Conventions (5-Second Index Options)

This document is the **sole reference** for implementing strategy Python files.
Every strategy lives in `pipeline/strategies/` as a single `.py` file exporting a
`Strategy` class that inherits from `BaseStrategy`.

**This pipeline trades NIFTY and BANKNIFTY index options at 5-second frequency.**

---

## 1. OptionSignals Interface

`compute()` must return an `OptionSignals` dataclass (from `pipeline.strategies.base`).

### Array fields (all `np.ndarray`, length == `len(spot_df)`)

| Field | dtype | Meaning | Neutral value |
|---|---|---|---|
| `buy_ce` | `bool` | True on bars where BUY CALL should fire (bullish) | `False` |
| `buy_pe` | `bool` | True on bars where BUY PUT should fire (bearish) | `False` |
| `sell_ce` | `bool` | True on bars where signal-based SELL CALL fires | `False` |
| `sell_pe` | `bool` | True on bars where signal-based SELL PUT fires | `False` |
| `stop_points` | `float64` | Stop loss in option premium points per bar | `0.0` |
| `target_points` | `float64` | Target in option premium points per bar | `0.0` |
| `strike_offset` | `int32` | Offset from ATM strike (0=ATM, 1=OTM1, -1=ITM1) | `0` |

### Scalar parameters

| Field | Type | Default | Meaning |
|---|---|---|---|
| `time_stop_bars` | `int` | `24` | Force exit after N bars (1 bar = 5 seconds, 24 = 120s) |
| `max_trades_per_day` | `int` | `10` | Max trades per day |

**Key differences from equity pipeline:**
- No `long_entry` / `short_entry` — replaced by `buy_ce` / `buy_pe`
- No percentage-based stops — stops/targets are in option premium POINTS
- No ATR-based stops (ATR doesn't apply to option premiums the same way)
- No trailing stops or breakeven (too complex for 5-60 second holds)
- Strike offset controls which option contract to trade

---

## 2. Input DataFrames

The `compute()` method receives THREE DataFrames, not one:

### `spot_df` — Index 5-second candles (IST timestamps)

| Column | dtype | Description |
|---|---|---|
| `datetime` | `Datetime` | IST naive timestamp |
| `session_date` | `Date` | Trading date |
| `open` | `Float64` | Index open |
| `high` | `Float64` | Index high |
| `low` | `Float64` | Index low |
| `close` | `Float64` | Index close |
| `volume` | `Int64` | Index volume |
| `day_id` | `Int32` | Integer day identifier (0, 1, 2, ...) |
| `time_minutes` | `Int32` | Minutes from midnight IST (e.g., 555 = 09:15) |
| `atm_strike` | `Float64` | ATM strike (nearest to close, step=50 NIFTY / 100 BANKNIFTY) |

### `option_df` — Option chain 5-second candles

| Column | dtype | Description |
|---|---|---|
| `symbol` | `String` | Option symbol (e.g., "NSE:NIFTY2631024700CE") |
| `underlying` | `String` | "NSE:NIFTY50-INDEX" or "NSE:NIFTYBANK-INDEX" |
| `session_date` | `Date` | Trading date |
| `expiry` | `Date` | Expiry date |
| `strike` | `Float64` | Strike price |
| `option_type` | `String` | "CE" or "PE" |
| `datetime` | `Datetime` | IST naive timestamp |
| `open` | `Float64` | Option premium open |
| `high` | `Float64` | Option premium high |
| `low` | `Float64` | Option premium low |
| `close` | `Float64` | Option premium close |
| `volume` | `Int64` | Option volume |
| `open_interest` | `Int64` | Open interest |
| `day_id` | `Int32` | Integer day identifier |
| `time_minutes` | `Int32` | Minutes from midnight IST |

### `vix_df` — India VIX 5-second candles

Same columns as `spot_df` but for "NSE:INDIAVIX-INDEX".

---

## 3. Time Constants (IST, minutes from midnight)

| Time IST | Minutes | Use |
|----------|---------|-----|
| 09:15 | 555 | Market open |
| 09:20 | 560 | Default session start (skip pre-open noise) |
| 15:20 | 920 | Expiry day flatten |
| 15:25 | 925 | EOD flatten |
| 15:30 | 930 | Market close |

---

## 4. Cost Model (always ON)

Every trade incurs:
- **Bid-ask spread:** ~1.5 points per side (3 points round trip)
- **STT:** 0.0625% on sell-side (exit premium × 0.000625 × qty)
- **Brokerage:** ₹20 per order (₹40 round trip)
- **Exchange charges:** ~₹5 per lot per side

**Minimum premium move to breakeven:** ~4 points for NIFTY (lot=75), ~6 points for BANKNIFTY (lot=15)

A strategy must clear ~4 option points per trade after costs to be viable.

---

## 5. Lot Sizes

| Underlying | Lot Size | Strike Step |
|-----------|----------|-------------|
| NIFTY | 75 | 50 |
| BANKNIFTY | 15 | 100 |

---

## 6. Strategy Class Requirements

```python
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl

class Strategy(BaseStrategy):
    name = "my_strategy_v1"
    underlying = "NIFTY"           # or "BANKNIFTY" or "BOTH"
    session_start_minutes = 560    # 09:20 IST
    session_end_minutes = 925      # 15:25 IST
    max_trades_per_day = 10
    max_lookback = 120             # warmup bars (120 × 5s = 10 min)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("threshold", 0.05, 0.01, 0.15),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)
        # ... compute indicators on spot_df ...
        # ... generate signals ...
        return OptionSignals(
            buy_ce=buy_ce_mask,
            buy_pe=buy_pe_mask,
            sell_ce=sell_ce_mask,
            sell_pe=sell_pe_mask,
            stop_points=np.full(n, 5.0),
            target_points=np.full(n, 10.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,         # 120 seconds
            max_trades_per_day=5,
        )
```

---

## 7. NaN Handling

- Forward-fill NaN in Polars BEFORE converting to numpy
- NaN stop_points → 0.0 (no stop)
- NaN target_points → 0.0 (no target)
- NaN booleans → False
- VIX close NaN → 15.0 (neutral)
- Option premium NaN → skip (don't trade)

---

## 8. Reference Implementations

### Example 1: Simple VIX Regime Strategy

```python
class Strategy(BaseStrategy):
    name = "vix_regime_simple_v1"
    underlying = "NIFTY"
    session_start_minutes = 560
    session_end_minutes = 925
    max_lookback = 60  # 5 minutes warmup

    def tunable_params(self):
        return [
            TunableParam("vix_low", 12.0, 8.0, 16.0),
            TunableParam("vix_high", 20.0, 16.0, 28.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # Get VIX close aligned to spot bars
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime", strategy="backward"
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # 5-second returns
        ret_1 = np.zeros(n)
        ret_1[1:] = (close[1:] - close[:-1]) / close[:-1]

        # Rolling 60-bar (5-min) momentum
        mom = np.zeros(n)
        for i in range(60, n):
            mom[i] = (close[i] - close[i-60]) / close[i-60]

        vix_low = params.get("vix_low", 12.0)
        vix_high = params.get("vix_high", 20.0)

        # Session filter
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # Low VIX + positive momentum → buy CE
        buy_ce = in_session & (vix_close < vix_low) & (mom > 0.001) & (ret_1 > 0)

        # High VIX + negative momentum → buy PE
        buy_pe = in_session & (vix_close > vix_high) & (mom < -0.001) & (ret_1 < 0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 5.0),
            target_points=np.full(n, 10.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=5,
        )
```

### Example 2: IV Dislocation Strategy

```python
class Strategy(BaseStrategy):
    name = "iv_dislocation_v1"
    underlying = "BOTH"
    session_start_minutes = 570  # 09:30 IST
    session_end_minutes = 920    # 15:20 IST
    max_lookback = 240           # 20 min warmup

    def tunable_params(self):
        return [
            TunableParam("iv_zscore_threshold", 2.0, 1.5, 3.0),
            TunableParam("stop_pts", 8.0, 4.0, 15.0),
            TunableParam("target_pts", 12.0, 6.0, 25.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].to_numpy()
        atm = spot_df["atm_strike"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        threshold = params.get("iv_zscore_threshold", 2.0)
        stop_pts = params.get("stop_pts", 8.0)
        target_pts = params.get("target_pts", 12.0)

        # Compute realized vol (5-second returns, annualized)
        ret = np.zeros(n)
        ret[1:] = np.log(close[1:] / close[:-1])
        rv_60 = np.zeros(n)
        for i in range(60, n):
            rv_60[i] = np.std(ret[i-60:i]) * np.sqrt(252 * 6.25 * 3600 / 5)

        # Use VIX as IV proxy
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime", strategy="backward"
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        iv = vix_close / 100.0  # convert from % to decimal
        iv_rv_spread = iv - rv_60
        spread_mean = np.zeros(n)
        spread_std = np.ones(n)
        for i in range(240, n):
            window = iv_rv_spread[i-240:i]
            spread_mean[i] = np.mean(window)
            s = np.std(window)
            spread_std[i] = s if s > 0.001 else 0.001
        iv_zscore = (iv_rv_spread - spread_mean) / spread_std

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # IV much higher than RV → IV will revert down → buy PE (premium will drop, but we
        # are betting on the underlying dropping as high IV often precedes sell-offs)
        buy_pe = in_session & (iv_zscore > threshold) & (rv_60 > 0)

        # IV much lower than RV → volatility will mean-revert up → buy CE
        buy_ce = in_session & (iv_zscore < -threshold) & (rv_60 > 0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=3,
        )
```

---

## 9. Decisions & Assumptions

1. **Class must be named `Strategy`** (not the strategy name)
2. **Self-contained files** — no imports between strategy files
3. **Helper functions** prefixed with `_` (e.g., `_compute_momentum()`)
4. **Indicators computed on SPOT data** (index), not on option premiums
5. **Signals fire on spot data**, execution happens on option premiums
6. **No option writing** — buy CE or buy PE only, sell to close
7. **No simultaneous positions** — one position at a time
8. **ATM strike by default** — use `strike_offset=0` unless strategy requires OTM/ITM
9. **5-second bars** — every indicator must make sense at this frequency
10. **12 trading days only** — results are preliminary, flag this limitation
11. **UTC→IST conversion** — timestamps in data are UTC, convert with `convert_time_zone("Asia/Kolkata")`
12. **Forward-fill** NaN in Polars before `.to_numpy()`
13. **Loops acceptable** — hot path is state_machine.py via Numba
14. **Cost model always ON** — strategies must clear ~4 pts/trade

---

## 10. CONVERSION PRINCIPLES — Adapting Strategies from Original to 5s Index Options

This section governs how strategies from `trading_strategies/unique_strategies_all/` are converted
to 5-second index option strategies in `unique_strategies/`. These rules are **mandatory**.

### 10.1 BAD vs GOOD Conversion

**Original:** `rsi_oversold_bounce_v1` — "Buy when RSI(14) drops below 30 on 1-min bars of NIFTY200 stocks, sell when RSI crosses above 50. Hold 5-30 min."

**BAD conversion (REJECTED):**
```json
{
  "mechanism": "Short-term selling pressure exhausts available demand; remaining limit buy orders create a floor and price reverts as new buyers step in",
  "indicators": [{"name": "rsi_168", "formula": "RSI(close, 168)", "lookback": 168}],
  "stop_points": 3,
  "target_points": 5,
  "hold_time_seconds": [15, 60]
}
```
Problems: mechanism is a generic template (could describe ANY mean reversion), lookback blindly scaled
(14 × 12 = 168), stops are category defaults ("3 because scalp").

**GOOD conversion (ACCEPTED):**
```json
{
  "mechanism": "On NIFTY, a sharp 2-3 minute selloff (RSI dropping below 30 over ~36 bars) triggers stop-loss cascades and algo selling that overshoots. The overshoot creates a 10-20 point spot reversion as selling exhausts and resting buy orders absorb flow. At 5-second resolution, we can enter as the reversion begins (RSI turning up from below 30) rather than waiting for a full 1-minute bar to confirm, capturing the first 5-10 points of bounce.",
  "indicators": [
    {"name": "rsi_36", "formula": "RSI(close, 36)", "lookback": 36, "computed_on": "spot",
     "purpose": "3-minute RSI to detect short-term selloff exhaustion. Scaled from 14-min (original) to 3-min because at 5s bars we're trading the FAST reversion, not the slow one. The thesis is about the first bounce, not the full recovery."}
  ],
  "stop_points": 4,
  "stop_rationale": "4 points because NIFTY mean-reversion bounces average 6-8 spot points (3-4 option points at delta 0.5) and we want 1:1.5 risk-reward",
  "target_points": 6,
  "target_rationale": "6 points = ~12 spot point move, which is the median first-leg reversion from RSI<30 on NIFTY based on 3-minute selloffs"
}
```
The difference: the GOOD version **thinks about what changes** when moving from 1-min equity to 5s index options.

### 10.2 Mandatory Rules

1. **No template mechanisms.** Every mechanism must reference NIFTY or BANKNIFTY by name and describe
   what happens in the INDEX order book when this strategy's specific conditions are met. If the same
   mechanism paragraph could describe a different strategy by changing the indicator name, it's a template.

2. **No category-default stops.** Every strategy's `stop_points` and `target_points` must have a
   `stop_rationale` and `target_rationale` field. "3 points because scalp" is REJECTED. "3 points
   because momentum continuation on NIFTY should not retrace more than 6 spot points (3 option pts
   at delta 0.5) within 30 seconds if the thesis is correct" is ACCEPTED.

3. **No blind 12x scaling.** Every indicator's `purpose` field must explain WHY this lookback was
   chosen. "168 bars because 14 × 12" is REJECTED. "36 bars (3 min) because we're catching the fast
   overshoot, not the slow recovery" is ACCEPTED. Sometimes 12x IS correct — but you must explain why.

4. **Mechanism must reference NIFTY/BANKNIFTY specifically.** Not "stocks," "equities," or "the market."
   What happens on the INDEX? Indices are smoother, more liquid, less gappy, driven by
   institutional/FII flow rather than single-stock events.

5. **Kill strategies that don't work at 5s.** If during conversion you realize the strategy only makes
   sense at 15-minute resolution or longer, don't force it. Skip it. Write a note to
   `outputs/conversion_kills.json` explaining why.

6. **Differentiate variants.** If two originals are near-duplicates (e.g., same thesis, different
   parameters), the converted versions must have DIFFERENT mechanisms, indicators, or entry logic.
   Two VWAP reversion strategies should trade different deviation depths, different timeframes, or
   different microstructure signals.

### 10.3 Lookback Decision Framework

When converting a lookback period from the original timeframe to 5-second bars:

| Question | If YES | If NO |
|----------|--------|-------|
| Is this a FIXED TIME WINDOW (e.g., opening range 09:15-09:35)? | Keep the same time, just more bars (20 min = 240 bars) | Continue to next question |
| Is this a CUMULATIVE measure (e.g., VWAP from session open)? | No scaling needed — cumulative measures don't have lookbacks | Continue to next question |
| Does the original's time horizon MATCH our hold time? | Scale proportionally (if original holds 10 min and looks back 30 min, and we hold 60s, look back ~3 min) | Continue to next question |
| Is the indicator a TRADE TRIGGER (direct entry signal)? | Compress to match hold time (if holding 30-90s, trigger should be 1-5 min lookback) | It's a CONTEXT FILTER — can be longer (15-60 min) |

### 10.4 Stop/Target Calibration Reference

Typical NIFTY/BANKNIFTY moves at different timescales (ATM option points, delta ~0.5):

| Timeframe | NIFTY typical move | BANKNIFTY typical move |
|-----------|-------------------|----------------------|
| 15 seconds (3 bars) | 1-3 pts | 2-4 pts |
| 30 seconds (6 bars) | 2-5 pts | 3-7 pts |
| 1 minute (12 bars) | 3-7 pts | 5-10 pts |
| 2 minutes (24 bars) | 5-10 pts | 7-15 pts |
| 5 minutes (60 bars) | 7-15 pts | 10-25 pts |

Use this to sanity-check stops and targets against hold time.

### 10.5 Gold Standard Conversion Examples

These 5 conversions are the quality reference. Every subsequent conversion must match this level of
specificity and reasoning.

---

#### Example A: Trend-Following Pullback (sma_crossover_banknifty_v1)

**Original:** Triple SMA(10/29/100) alignment on BankNifty 1-min bars. Hold 5-20 min.

**Converted:**
- **Mechanism:** "On BANKNIFTY, large institutional TWAP/VWAP algorithms working directional orders create
  persistent micro-trends visible as SMA alignment across 3-min, 15-min, and 60-min windows. When all
  three align bullishly, the order flow is sustained at multiple timescales — not just a spike but
  continuous demand absorption. We enter on micro-pullbacks (price touching the 3-min SMA) and ride the
  next 20-30 spot point push."
- **Lookback:** SMA(36/180/720) = 3/15/60 min. NOT 12x blind (which would give 10/29/100 min — same as
  original). Compressed because we hold 30-60s, not 5-20 min. The 3-min SMA is the trade trigger; 15/60-min
  are context filters.
- **Stop:** 4 pts — micro-pullback retraces 10-15 spot pts; if >8 spot pts, micro-trend broken.
- **Target:** 7 pts — micro-bounce covers 20-30 spot pts; 7 is ~50% of expected move (1:1.75 R:R).

---

#### Example B: Momentum Continuation (index_momentum_v1)

**Original:** Enter when Nifty50 5-minute return > 0.2%. 1-min bars. Hold 1-10 min.

**Converted:**
- **Mechanism:** "NIFTY experiences sharp 1-minute directional bursts when macro news hits or algorithmic
  momentum strategies trigger cascading entries. These bursts have autocorrelation at the 30-60 second
  scale because the underlying institutional orders are typically only 40-60% filled."
- **Lookback:** Two-layer signal: ret_12 (1-min fast burst, NEW) + ret_60 (5-min trend, SAME as original
  time horizon). Fast layer detects acceleration WITHIN the trend, not the trend itself. Original had
  only one layer because 1-min bars couldn't distinguish fast bursts from slow drifts.
- **Stop:** 3 pts — momentum is high-conviction; if stalling within 30s, thesis is wrong. Tightest stop
  of all 5 examples because this is the fastest trade.
- **Target:** 5 pts — capture 40% of expected continuation (1:1.67 R:R).

---

#### Example C: Deep VWAP Reversion (vwap_mean_reversion_v18)

**Original:** Buy when VWAP z-score < -1.7 with volume surge on Nifty200 stocks. 15-min bars. Hold all day.

**Converted:**
- **Mechanism:** "On NIFTY, sharp intraday selloffs push the index 40-80 spot points below session VWAP.
  At this deviation level, every VWAP-benchmarked algorithm starts accumulating aggressively — they're
  getting fills significantly better than their benchmark."
- **Lookback:** VWAP (no scaling — cumulative), dev_momentum_24 (2-min rate of deviation change, REPLACES
  relative volume because index volume is noisy), vwap_stddev_360 (30-min rolling stddev, captures current
  session regime rather than original's full-day window).
- **Stop:** 6 pts — WIDEST of all 5 examples. Mean reversion needs room for the final flush before bounce.
- **Target:** 10 pts — deep deviations revert 50-70%; capture the lower end (1:1.67 R:R).

---

#### Example D: Quick VWAP Bounce (vwap_mean_reversion_v19)

**Original:** Nearly identical to v18 but 1-min bars, z-score 2.1. Hold 10-30 min.

**Converted (DIFFERENTIATED from v18):**
- **Mechanism:** "On NIFTY, moderate 20-35 spot point deviations from VWAP occur 5-10 times per session.
  Market makers short gamma from option selling are the first to fade these moves — they delta-hedge by
  buying the dip within 30-60 seconds."
- **Lookback:** bar_range_6 (30s range contraction to detect basing) + close_vs_low_6 (position in range
  to detect bounce start). COMPLETELY DIFFERENT indicators from v18 — detects microstructure basing, not
  macro deviation momentum.
- **Stop:** 4 pts — shallower deviation; if it deepens to v18 territory, this thesis is wrong.
- **Target:** 6 pts — smaller bounce, but 4-8 trades/day compensates (1:1.5 R:R).

---

#### Example E: Opening Range Breakout (orb_momentum_v22)

**Original:** Break above 20-min opening range high with VWAP + volume. 1-min bars. Hold 10-30 min.

**Converted:**
- **Mechanism:** "NIFTY's first 20 minutes is the overnight information absorption window. Large
  institutional participants wait for this range to form before initiating directional orders. Breakout
  signals institutional buy-side flow has overwhelmed the opening range's supply ceiling."
- **Lookback:** Opening range is a FIXED TIME WINDOW (09:15-09:35 = 240 bars). NOT scaled — it's the same
  20 minutes regardless of bar size. Added 3-bar persistence filter (15 seconds) to filter 5s spikes.
- **Stop:** 5 pts — breakouts "throwback" to the breakout level; 10 spot pt throwback is normal.
- **Target:** 8 pts — first leg of ORB breakout extends 20-40 spot pts beyond range (1:1.6 R:R).

---

### 10.6 JSON Structure for Converted Strategies

Every converted strategy in `unique_strategies/` must include these fields:

```json
{
  "name": "strategy_name",
  "version": "0.2.0",
  "author": "claude_strategist",
  "tags": ["category_tags"],
  "original_strategy": "path/to/original.json",
  "original_name": "original_strategy_name",
  "mechanism": "UNIQUE paragraph referencing NIFTY/BANKNIFTY specifically",
  "universe": "NIFTY | BANKNIFTY | BOTH",
  "timeframe": "5s",
  "session": "HH:MM-HH:MM IST",
  "hold_time_seconds": [min, max],
  "max_hold_bars": N,
  "trades_per_day": "range",
  "max_trades_per_day": N,
  "data": { "ohlcv_5s": true, "option_chain": true, "vix": bool, "index": "...", "order_book": false, "other": [] },
  "indicators": [
    { "name": "...", "formula": "...", "lookback": N, "computed_on": "spot", "purpose": "EXPLAINS lookback choice" }
  ],
  "entry": {
    "buy_ce": { "conditions": ["..."], "confirmation": "..." },
    "buy_pe": { "conditions": ["..."], "confirmation": "..." },
    "entry_price": "market"
  },
  "exit": {
    "stop_points": N,
    "stop_rationale": "WHY this number — references typical NIFTY/BANKNIFTY move sizes",
    "target_points": N,
    "target_rationale": "WHY this number — references expected move magnitude",
    "time_stop_bars": N,
    "eod_rule": "flatten at HH:MM"
  },
  "filters": { "vix_filter": "...", "time_filter": "...", "volume_filter": "...", "trend_filter": "...", "spread_filter": "..." },
  "edge": { "expected_bps": N, "win_rate_estimate": 0.XX, "avg_winner_to_loser": X.X, "profit_factor_estimate": X.X, "edge_decay_risk": "..." },
  "adaptation_notes": "DETAILED explanation of what changed from original and WHY",
  "weaknesses": ["specific to this strategy at 5s on this index"],
  "_source": {
    "original_file": "...",
    "original_timeframe": "...",
    "original_hold_time": "...",
    "original_indicators": ["..."],
    "original_stop": "...",
    "original_target": "..."
  }
}
```
