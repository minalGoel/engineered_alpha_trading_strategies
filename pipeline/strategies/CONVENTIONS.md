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
