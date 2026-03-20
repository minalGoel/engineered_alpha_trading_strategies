# Strategy Implementation Conventions

This document is the **sole reference** for implementing new strategy Python files.
Every strategy lives in `pipeline/strategies/` as a single `.py` file exporting a
`Strategy` class that inherits from `BaseStrategy`.

---

## 1. StrategySignals Interface

`compute()` must return a `StrategySignals` dataclass (from `pipeline.strategies.base`).

### Array fields (all `np.ndarray`, length == `len(df)`)

| Field | dtype | Meaning | Neutral value |
|---|---|---|---|
| `long_entry` | `bool` | `True` on bars where a BUY should fire | `False` |
| `short_entry` | `bool` | `True` on bars where a SHORT should fire | `False` |
| `signal_exit_long` | `bool` | `True` on bars where a signal-based SELL should fire | `False` |
| `signal_exit_short` | `bool` | `True` on bars where a signal-based COVER should fire | `False` |
| `atr_arr` | `float64` | ATR values per bar (used for ATR-based stops/targets) | `0.0` (zeros array if unused) |
| `target_indicator` | `float64` | Reference price level for indicator-touch targets (e.g. VWAP, EMA) | `0.0` (zeros array if unused) |

### Scalar exit parameters

| Field | Type | Default | Meaning |
|---|---|---|---|
| `stop_loss_pct` | `float` | `0.0` | Stop loss as decimal (0.003 = 0.3%). 0 = no pct stop. |
| `stop_loss_atr_mult` | `float` | `0.0` | ATR multiplier for stops. 0 = not used. Uses `atr_arr[entry_bar]`. |
| `target_pct` | `float` | `0.0` | Target as decimal (0.005 = 0.5%). 0 = no pct target. |
| `target_atr_mult` | `float` | `0.0` | ATR multiplier for targets. 0 = not used. |
| `use_target_indicator` | `bool` | `False` | If `True`, target is the `target_indicator` array value (e.g. VWAP touch). |
| `trailing_stop_pct` | `float` | `0.0` | Trailing stop as decimal. 0 = no trailing stop. |
| `trailing_activate_pct` | `float` | `0.0` | Profit threshold to activate trailing. 0 = always active when trailing enabled. |
| `breakeven_pct` | `float` | `0.0` | Move stop to breakeven after this profit. 0 = disabled. |
| `time_stop_bars` | `int` | `0` | Force exit after N bars in position. 0 = use DEFAULT_MAX_HOLD_BARS (120). |

**Critical**: All exit parameters are **scalars**, not arrays. The state machine uses the
same stop/target for the entire trade, computed at entry time.

**Priority**: Only ONE of `stop_loss_pct` or `stop_loss_atr_mult` is used (pct checked first).
Same for `target_pct` vs `target_atr_mult` vs `use_target_indicator`.

---

## 2. Input DataFrame Columns

The `df: pl.DataFrame` passed to `compute()` has these columns:

| Column | Polars dtype | Description |
|---|---|---|
| `datetime` | `Datetime` | Bar timestamp (IST timezone) |
| `open` | `Float64` | Bar open price |
| `high` | `Float64` | Bar high price |
| `low` | `Float64` | Bar low price |
| `close` | `Float64` | Bar close price |
| `volume` | `Float64` | Bar volume (may contain NaN — always clean) |
| `vix` | `Float64` | India VIX value at this bar (may contain NaN — clean to 99.0) |
| `index_close` | `Float64` | Nifty 50 close at this bar (may contain NaN) |
| `day_id` | `Int32` | Integer day identifier. Same value for all bars in one trading day. Increments across days. Used for VWAP daily reset via `.over("day_id")`. |
| `time_minutes` | `Int32` | Minutes since midnight IST. 555=09:15, 570=09:30, 920=15:20, 930=15:30. |

**Note**: OHLC are forward-filled before `compute()` is called by the backtester. You do
NOT need to forward-fill them yourself. But `volume`, `vix`, and `index_close` may still
have NaN — always clean them.

---

## 3. Stop/Target Price Convention (State Machine)

The Numba state machine in `pipeline/state_machine.py` computes stop and target prices
**at entry time** from the scalar parameters you provide.

### For LONG positions

```
entry_price = close[entry_bar]

# Stop loss (only ONE is used, pct checked first):
if stop_loss_pct > 0:
    stop_price = entry_price * (1.0 - stop_loss_pct)
elif stop_loss_atr_mult > 0:
    stop_price = entry_price - stop_loss_atr_mult * atr_arr[entry_bar]

# Target (only ONE is used, pct checked first):
if target_pct > 0:
    target_price = entry_price * (1.0 + target_pct)
elif target_atr_mult > 0:
    target_price = entry_price + target_atr_mult * atr_arr[entry_bar]

# Indicator target (checked every bar):
if use_target_indicator:
    actual_target = target_indicator[current_bar]  # e.g. VWAP
    # Long exits when high >= actual_target

# Trailing stop:
trailing_stop_price = highest_since_entry * (1.0 - trailing_stop_pct)
# Long exits when low <= trailing_stop_price

# Breakeven:
if profit_pct >= breakeven_pct:
    stop_price = max(stop_price, entry_price)
```

### For SHORT positions

```
entry_price = close[entry_bar]

# Stop loss:
if stop_loss_pct > 0:
    stop_price = entry_price * (1.0 + stop_loss_pct)
elif stop_loss_atr_mult > 0:
    stop_price = entry_price + stop_loss_atr_mult * atr_arr[entry_bar]

# Target:
if target_pct > 0:
    target_price = entry_price * (1.0 - target_pct)
elif target_atr_mult > 0:
    target_price = entry_price - target_atr_mult * atr_arr[entry_bar]

# Indicator target:
if use_target_indicator:
    actual_target = target_indicator[current_bar]
    # Short exits when low <= actual_target

# Trailing stop:
trailing_stop_price = lowest_since_entry * (1.0 + trailing_stop_pct)
# Short exits when high >= trailing_stop_price
```

### Exit priority (checked in order on each bar)
1. Stop loss (fills at stop_price)
2. Target (fills at target_price or target_indicator value)
3. Trailing stop (fills at trailing_stop_price)
4. Signal exit (fills at close)
5. Time stop (fills at close)
6. EOD flatten at 15:20 IST / time_minutes=920 (fills at close)

### Session window
The state machine only allows **entries** when `session_start <= time_minutes < session_end`.
Exits can happen at any time. EOD flatten happens at time_minutes >= 920 regardless.

---

## 4. VWAP Computation (Daily Reset)

Always use this exact Polars pattern for VWAP with daily reset:

```python
df_v = df.with_columns([
    ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
    .cum_sum().over("day_id").alias("_ctv"),
    pl.col("volume").cum_sum().over("day_id").alias("_cv"),
]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
vwap = df_v["_vwap"].to_numpy().astype(np.float64)
vwap = np.nan_to_num(vwap, nan=0.0)
```

**Why `.over("day_id")`**: The `day_id` column groups bars by trading day. `cum_sum().over("day_id")`
resets the cumulative sum at each new day boundary, giving correct intraday VWAP.

---

## 5. NaN Handling

Every numpy array extracted from the DataFrame must be cleaned before use. Different
indicator types get different neutral values:

| Data type | NaN replacement | Reasoning |
|---|---|---|
| `volume` | `1.0` | Avoid division by zero in volume ratios |
| `vix` | `99.0` | High VIX = conservative, blocks most VIX filters |
| `index_close` | `0.0` | Neutral; index-based signals won't fire |
| `close`, `open`, `high`, `low` | Already forward-filled by backtester | Don't need cleaning |
| `atr_arr` (output) | `0.0` | State machine treats 0 ATR as "not available" |
| `target_indicator` (output) | `0.0` | State machine skips indicator target if 0 |
| EMA / SMA | `0.0` | Polars fills initial NaN with 0 |
| RSI | `50.0` | Neutral RSI = no signal |
| Bollinger std | `1e10` (huge) or `0.0` | Huge std = wide bands = no signal during warmup |
| Rolling mean for volume | `1.0` then `np.clip(..., 1.0, None)` | Avoid div-by-zero |

**Pattern**: Always clean immediately after extraction:
```python
volume = df["volume"].to_numpy().astype(np.float64)
volume = np.nan_to_num(volume, nan=1.0)

vix = df["vix"].to_numpy().astype(np.float64)
vix = np.nan_to_num(vix, nan=99.0)
```

**Division safety**: When dividing by a rolling mean or any denominator, always clip:
```python
avg_vol = np.nan_to_num(avg_vol, nan=1.0)
avg_vol = np.clip(avg_vol, 1.0, None)
ratio = volume / avg_vol
```

---

## 6. TunableParam Convention

`tunable_params()` returns a list of `TunableParam(name, default, low, high)`.
Optuna will suggest values in `[low, high]` and pass them via the `params` dict.

### What IS tunable
- Numeric thresholds: RSI levels, VWAP deviation percentages, VIX limits
- Stop/target percentages: `stop_loss_pct`, `target_pct`, `breakeven_pct`
- Multipliers: `vol_mult`, `stop_loss_atr_mult`
- Count thresholds: minimum consecutive bars, minimum agreeing sub-signals

### What is NOT tunable
- **Indicator periods** (EMA span, RSI period, ATR period, BB window). These are hardcoded.
- **Session times** (session_start, session_end). These are class attributes.
- **Boolean flags** (is_long_only, use_target_indicator).

### Common parameter bounds

| Parameter type | Typical default | Low | High |
|---|---|---|---|
| `stop_loss_pct` | 0.003-0.005 | 0.001 | 0.01 |
| `target_pct` | 0.005-0.008 | 0.002 | 0.015 |
| `trailing_stop_pct` | 0.002-0.003 | 0.001 | 0.005 |
| `trailing_activate_pct` | 0.003 | 0.001 | 0.005 |
| `breakeven_pct` | 0.0025-0.003 | 0.001 | 0.006 |
| `rsi_long_thresh` | 30.0-35.0 | 15.0 | 45.0 |
| `rsi_short_thresh` | 65.0-70.0 | 55.0 | 85.0 |
| `vix_max` | 22.0-25.0 | 15.0 | 30.0 |
| `vol_ratio_thresh` | 1.5 | 1.0 | 3.0 |
| `vwap_dev (decimal)` | 0.005 | 0.002 | 0.015 |
| `adx_thresh` | 25.0 | 18.0 | 35.0 |

### Usage in compute()
Always use `params.get(name, default)` — never assume the key exists:
```python
stop_pct = params.get("stop_loss_pct", 0.003)
```

---

## 7. Session Time Handling

`session_start` and `session_end` are class attributes in **minutes from midnight IST**.

| IST Time | time_minutes |
|---|---|
| 09:15 | 555 |
| 09:20 | 560 |
| 09:25 | 565 |
| 09:30 | 570 |
| 09:45 | 585 |
| 10:00 | 600 |
| 10:15 | 615 |
| 10:30 | 630 |
| 11:00 | 660 |
| 11:15 | 675 |
| 12:00 | 720 |
| 13:00 | 780 |
| 14:00 | 840 |
| 14:15 | 855 |
| 14:30 | 870 |
| 14:45 | 885 |
| 15:00 | 900 |
| 15:10 | 910 |
| 15:15 | 915 |
| 15:20 | 920 (EOD flatten) |
| 15:30 | 930 (market close) |

**The backtester applies session_start/session_end to the entry masks automatically.**
You do NOT need to filter by session time in your entry logic — but you can add a
redundant time_ok filter for clarity, it won't hurt.

**Session end must be <= 920** (EOD flatten time). The backtester caps it:
`capped_session_end = min(strategy.session_end, 920)`.

---

## 8. Common Indicator Snippets

Copy-paste these directly. All use Wilder's smoothing where applicable.

### ATR (Wilder's method)

```python
def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr
```

### RSI (Wilder's method)

```python
def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    for i in range(period+1, n):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi
```

### EMA (loop-based)

```python
def _compute_ema(data, period):
    n = len(data)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = data[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = data[i] * k + ema[i-1] * (1 - k)
    return ema
```

Or use Polars (simpler but allocates a Series):
```python
ema9 = df["close"].ewm_mean(span=9).to_numpy().astype(np.float64)
ema9 = np.nan_to_num(ema9, nan=0.0)
```

### SMA (Polars)

```python
sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
sma20 = np.nan_to_num(sma20, nan=0.0)
```

### Bollinger Bands

```python
sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
sma20 = np.nan_to_num(sma20, nan=0.0)
std20 = np.nan_to_num(std20, nan=0.0)
bb_upper = sma20 + 2.0 * std20
bb_lower = sma20 - 2.0 * std20
```

### MACD

```python
ema12 = _compute_ema(close, 12)
ema26 = _compute_ema(close, 26)
macd_line = ema12 - ema26
macd_signal = _compute_ema(macd_line, 9)
macd_hist = macd_line - macd_signal
```

### VWAP (daily reset)

```python
df_v = df.with_columns([
    ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
    .cum_sum().over("day_id").alias("_ctv"),
    pl.col("volume").cum_sum().over("day_id").alias("_cv"),
]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
vwap = df_v["_vwap"].to_numpy().astype(np.float64)
vwap = np.nan_to_num(vwap, nan=0.0)
```

### Z-Score

```python
rolling_mean = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
rolling_std = df["close"].rolling_std(20).to_numpy().astype(np.float64)
rolling_mean = np.nan_to_num(rolling_mean, nan=0.0)
rolling_std = np.nan_to_num(rolling_std, nan=1.0)
rolling_std = np.clip(rolling_std, 1e-10, None)
zscore = (close - rolling_mean) / rolling_std
```

### Relative Volume

```python
avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
avg_vol = np.nan_to_num(avg_vol, nan=1.0)
avg_vol = np.clip(avg_vol, 1.0, None)
rel_vol = volume / avg_vol
```

### ADX with +DI / -DI (Wilder's method)

```python
def _compute_adx(high, low, close, period):
    n = len(close)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    adx = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return adx, plus_di, minus_di

    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    # Wilder smoothing
    sm_tr = np.zeros(n, dtype=np.float64)
    sm_pdm = np.zeros(n, dtype=np.float64)
    sm_mdm = np.zeros(n, dtype=np.float64)
    sm_tr[period] = np.sum(tr[1:period+1])
    sm_pdm[period] = np.sum(plus_dm[1:period+1])
    sm_mdm[period] = np.sum(minus_dm[1:period+1])
    for i in range(period+1, n):
        sm_tr[i] = sm_tr[i-1] - sm_tr[i-1]/period + tr[i]
        sm_pdm[i] = sm_pdm[i-1] - sm_pdm[i-1]/period + plus_dm[i]
        sm_mdm[i] = sm_mdm[i-1] - sm_mdm[i-1]/period + minus_dm[i]

    for i in range(period, n):
        if sm_tr[i] > 0:
            plus_di[i] = 100.0 * sm_pdm[i] / sm_tr[i]
            minus_di[i] = 100.0 * sm_mdm[i] / sm_tr[i]

    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        s = plus_di[i] + minus_di[i]
        if s > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / s

    if n >= 2 * period:
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period

    return adx, plus_di, minus_di
```

---

## 9. Reference Implementations

### 9a. Simple — VWAP Mean Reversion (cursor_opus46max_001)

VWAP deviation + RSI + volume + 2-bar confirmation. Target = VWAP touch.

```python
"""VWAP Mean Reversion Basic v1 — cursor_opus46max_001

Thesis: Stocks deviating significantly from VWAP during the first 90 minutes
revert because VWAP represents institutional fair price. Enter on deviation
with volume + RSI confirmation after two consecutive confirming bars.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_001"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_thresh", default=0.6, low=0.3, high=1.2),
            TunableParam("vol_ratio_thresh", default=1.5, low=1.0, high=3.0),
            TunableParam("rsi_long_thresh", default=35.0, low=20.0, high=45.0),
            TunableParam("rsi_short_thresh", default=65.0, low=55.0, high=80.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
            TunableParam("breakeven_pct", default=0.003, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        dev_thresh = params.get("vwap_dev_thresh", 0.6)
        vol_thresh = params.get("vol_ratio_thresh", 1.5)
        rsi_long = params.get("rsi_long_thresh", 35.0)
        rsi_short = params.get("rsi_short_thresh", 65.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        be_pct = params.get("breakeven_pct", 0.003)

        n = len(df)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        close = df["close"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP deviation % ──
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        vwap_dev_pct = (close - vwap) / safe_vwap * 100.0

        # ── Volume ratio ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── Two consecutive green/red bar confirmation ──
        green2 = np.zeros(n, dtype=np.bool_)
        red2 = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            green2[i] = (close[i] > close[i - 1]) and (close[i - 1] > close[i - 2])
            red2[i] = (close[i] < close[i - 1]) and (close[i - 1] < close[i - 2])

        # ── Time / VIX filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)
        vix_ok = vix < vix_max

        # ── Entries ──
        long_cond = (
            (close < vwap * (1.0 - dev_thresh / 100.0))
            & (vwap_dev_pct < -dev_thresh)
            & (vol_ratio > vol_thresh)
            & (rsi < rsi_long)
            & vix_ok & time_ok & green2
        )
        short_cond = (
            (close > vwap * (1.0 + dev_thresh / 100.0))
            & (vwap_dev_pct > dev_thresh)
            & (vol_ratio > vol_thresh)
            & (rsi > rsi_short)
            & vix_ok & time_ok & red2
        )

        # ── Signal exit: VWAP deviation crosses zero ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if vwap_dev_pct[i - 1] < 0 and vwap_dev_pct[i] >= 0:
                sig_exit_long[i] = True
            if vwap_dev_pct[i - 1] > 0 and vwap_dev_pct[i] <= 0:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_cond,
            short_entry=short_cond,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            stop_loss_atr_mult=1.5,
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=60,
        )
```

### 9b. Medium — ADX Trend Strength (cursor_opus46max_023)

ADX crossover + DI spread + VWAP + false breakout filter + signal exits.

```python
"""ADX Trend Strength v1 — cursor_opus46max_023

Thesis: ADX(14) crossing above 25 from below signals a nascent trend.
Enter when +DI > -DI (longs) or -DI > +DI (shorts) with DI spread > 10,
VWAP alignment, and ADX still rising as confirmation.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_adx_full(high, low, close, period):
    """Returns (adx, plus_di, minus_di) arrays."""
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return adx, plus_di, minus_di
    atr = _compute_atr(high, low, close, period)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    smooth_plus = np.zeros(n, dtype=np.float64)
    smooth_minus = np.zeros(n, dtype=np.float64)
    smooth_plus[period] = np.sum(plus_dm[1:period + 1])
    smooth_minus[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        smooth_plus[i] = smooth_plus[i - 1] - smooth_plus[i - 1] / period + plus_dm[i]
        smooth_minus[i] = smooth_minus[i - 1] - smooth_minus[i - 1] / period + minus_dm[i]
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * smooth_plus[i] / atr[i] / period
            minus_di[i] = 100.0 * smooth_minus[i] / atr[i] / period
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        denom = plus_di[i] + minus_di[i]
        if denom > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / denom
    if n >= 2 * period:
        adx[2 * period - 1] = np.mean(dx[period:2 * period])
        for i in range(2 * period, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx, plus_di, minus_di


class Strategy(BaseStrategy):
    name = "cursor_opus46max_023"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 885     # 14:45
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_cross_level", default=25.0, low=20.0, high=30.0),
            TunableParam("di_spread_min", default=10.0, low=5.0, high=15.0),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.004),
            TunableParam("trailing_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("trailing_activate", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_lvl = params.get("adx_cross_level", 25.0)
        di_spread = params.get("di_spread_min", 10.0)
        tgt_pct = params.get("target_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        trail_pct = params.get("trailing_pct", 0.002)
        trail_act = params.get("trailing_activate", 0.003)

        n = len(df)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── ADX with DI lines ──
        adx, plus_di, minus_di = _compute_adx_full(high, low, close, 14)

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── ADX cross above threshold ──
        adx_cross_up = np.zeros(n, dtype=np.bool_)
        adx_rising = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if adx[i] > adx_lvl and adx[i - 1] <= adx_lvl:
                adx_cross_up[i] = True
            if adx[i] > adx[i - 1]:
                adx_rising[i] = True

        # ── False breakout filter: skip if ADX crossed and fell back 2+ times
        #    in the last 60 bars ──
        adx_fell_back = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if adx[i] < adx_lvl and adx[i - 1] >= adx_lvl:
                adx_fell_back[i] = 1
        false_breakout_count = np.zeros(n, dtype=np.int32)
        for i in range(n):
            start = max(0, i - 59)
            false_breakout_count[i] = np.sum(adx_fell_back[start:i + 1])
        fb_ok = false_breakout_count < 2

        # ── Filters ──
        time_ok = (time_mins >= 585) & (time_mins <= 885)

        # ── Entries ──
        long_entry = (
            adx_cross_up
            & adx_rising
            & (plus_di > minus_di)
            & ((plus_di - minus_di) > di_spread)
            & (close > vwap)
            & vol_ok & fb_ok & time_ok
        )
        short_entry = (
            adx_cross_up
            & adx_rising
            & (minus_di > plus_di)
            & ((minus_di - plus_di) > di_spread)
            & (close < vwap)
            & vol_ok & fb_ok & time_ok
        )

        # ── Signal exit: ADX declining AND DI lines converging ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            adx_declining = adx[i] < adx[i - 1]
            di_converging_l = (plus_di[i] - minus_di[i]) < (plus_di[i - 1] - minus_di[i - 1])
            di_converging_s = (minus_di[i] - plus_di[i]) < (minus_di[i - 1] - plus_di[i - 1])
            if adx_declining and di_converging_l:
                sig_exit_long[i] = True
            if adx[i] < 20:
                sig_exit_long[i] = True
            if adx_declining and di_converging_s:
                sig_exit_short[i] = True
            if adx[i] < 20:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=60,
        )
```

### 9c. Complex — VWAP Overshoot Fade (cursor_gpt54_vwap_overshoot_fade_001)

RSI(2) + VWAP + ADX + BB + relative volume + index return + opening range filter + breakeven.

```python
"""VWAP Overshoot Fade 001 — cursor_gpt54_vwap_overshoot_fade_001

Thesis: Exploits short-lived intraday dislocations away from session VWAP in
liquid Indian equities.  Opening inventory imbalances and retail chase flow
create temporary overshoots in otherwise balanced tapes.

Entry long: close <= VWAP*0.995, RSI(2) <= 8, rel_vol >= 1.2, ADX(14) <= 22,
    close > BB lower, abs(index_ret_15) <= 0.004, bullish bar.
Entry short: mirror with close >= VWAP*1.005, RSI(2) >= 92.
Target: VWAP (use_target_indicator).  Stop: 0.45%.  BE after +0.25%.
Time stop: 10 bars.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_rsi_wilder(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


def _compute_adx(high, low, close, period):
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    if n < period * 2:
        return adx
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr_s = np.zeros(n, dtype=np.float64)
    pdm_s = np.zeros(n, dtype=np.float64)
    mdm_s = np.zeros(n, dtype=np.float64)
    atr_s[period] = np.sum(tr[1:period + 1])
    pdm_s[period] = np.sum(plus_dm[1:period + 1])
    mdm_s[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
        pdm_s[i] = pdm_s[i - 1] - pdm_s[i - 1] / period + plus_dm[i]
        mdm_s[i] = mdm_s[i - 1] - mdm_s[i - 1] / period + minus_dm[i]
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if atr_s[i] > 0:
            plus_di[i] = 100.0 * pdm_s[i] / atr_s[i]
            minus_di[i] = 100.0 * mdm_s[i] / atr_s[i]
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    start = 2 * period
    if start < n:
        adx[start] = np.mean(dx[period:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


class Strategy(BaseStrategy):
    name = "vwap_overshoot_fade_001"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 675     # 11:15
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_long", default=0.005, low=0.003, high=0.010),
            TunableParam("vwap_dev_short", default=0.005, low=0.003, high=0.010),
            TunableParam("rsi_long_thresh", default=8.0, low=3.0, high=15.0),
            TunableParam("rsi_short_thresh", default=92.0, low=85.0, high=97.0),
            TunableParam("adx_max", default=22.0, low=16.0, high=28.0),
            TunableParam("rel_vol_min", default=1.2, low=1.0, high=2.0),
            TunableParam("index_ret_max", default=0.004, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.0045, low=0.003, high=0.007),
            TunableParam("breakeven_pct", default=0.0025, low=0.002, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vwap_dev_l = params.get("vwap_dev_long", 0.005)
        vwap_dev_s = params.get("vwap_dev_short", 0.005)
        rsi_l = params.get("rsi_long_thresh", 8.0)
        rsi_s = params.get("rsi_short_thresh", 92.0)
        adx_max = params.get("adx_max", 22.0)
        rel_vol_min = params.get("rel_vol_min", 1.2)
        idx_ret_max = params.get("index_ret_max", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.0045)
        be_pct = params.get("breakeven_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        index_close = df["index_close"].to_numpy().astype(np.float64)

        atr = _compute_atr(high, low, close, 14)
        rsi_fast = _compute_rsi_wilder(close, 2)
        adx = _compute_adx(high, low, close, 14)

        # ── VWAP ──
        vwap_df = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = vwap_df["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── BB (SMA20, 2*std) ──
        close_sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        close_sma20 = np.nan_to_num(close_sma20, nan=0.0)
        close_std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        close_std20 = np.nan_to_num(close_std20, nan=0.0)
        bb_upper = close_sma20 + 2.0 * close_std20
        bb_lower = close_sma20 - 2.0 * close_std20

        # ── Relative volume ──
        vol_sma20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        vol_sma20 = np.nan_to_num(vol_sma20, nan=1.0)
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        rel_vol = volume / vol_sma20

        # ── Index return 15 bars ──
        idx_ret_15 = np.zeros(n, dtype=np.float64)
        for i in range(15, n):
            if index_close[i - 15] > 0:
                idx_ret_15[i] = index_close[i] / index_close[i - 15] - 1.0

        # ── OR range filter ──
        unique_days = np.unique(day_ids)
        or_range_ok = np.ones(n, dtype=np.bool_)
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) == 0:
                continue
            or_bars = idx[time_mins[idx] < 570]
            if len(or_bars) == 0:
                continue
            or_rng = np.max(high[or_bars]) - np.min(low[or_bars])
            atr_val = atr[or_bars[-1]] if atr[or_bars[-1]] > 0 else atr[idx[-1]]
            if atr_val > 0 and or_rng > 1.8 * atr_val:
                or_range_ok[idx] = False

        # ── Time filter: 09:25-11:15 (565-675) ──
        time_ok = (time_mins >= 565) & (time_mins <= 675)

        bullish = close > open_
        bearish = close < open_

        long_entry = (
            (close <= vwap * (1 - vwap_dev_l)) &
            (rsi_fast <= rsi_l) &
            (rel_vol >= rel_vol_min) &
            (adx <= adx_max) &
            (close > bb_lower) &
            (np.abs(idx_ret_15) <= idx_ret_max) &
            bullish &
            time_ok &
            or_range_ok
        )

        short_entry = (
            (close >= vwap * (1 + vwap_dev_s)) &
            (rsi_fast >= rsi_s) &
            (rel_vol >= rel_vol_min) &
            (adx <= adx_max) &
            (close < bb_upper) &
            (np.abs(idx_ret_15) <= idx_ret_max) &
            bearish &
            time_ok &
            or_range_ok
        )

        # ── Signal exit: close crosses VWAP OR RSI crosses 55/45 against ──
        sig_exit_long = ((close >= vwap) & (vwap > 0)) | (rsi_fast >= 55)
        sig_exit_short = ((close <= vwap) & (vwap > 0)) | (rsi_fast <= 45)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap,
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=10,
        )
```

---

## 10. Decisions & Assumptions

These are non-obvious choices made during implementation. Follow them for consistency.

### Architecture
1. **Each strategy file is self-contained.** Helper functions like `_compute_atr`, `_compute_rsi`
   are duplicated in every file that uses them. This is intentional — no shared utility module,
   no import dependencies between strategy files. A file can be deleted without breaking others.

2. **The class must be named `Strategy`** (not the strategy name). The loader does
   `mod.Strategy()` to instantiate it.

3. **`strategy.name` must match the JSON `name` field** from the corresponding strategy JSON.
   This is how the pipeline maps JSON files to Python implementations.

4. **Helper functions go OUTSIDE the class**, at module level, prefixed with `_`.

### Indicators
5. **RSI and ATR use Wilder's smoothing** (exponential with factor `(period-1)/period`),
   NOT simple moving average smoothing. This matches standard TA libraries.

6. **EMA uses standard exponential** with `k = 2/(period+1)`, computed via loop for
   full control. Alternatively `df["close"].ewm_mean(span=N)` via Polars is acceptable.

7. **VWAP typical price = (H+L+C)/3**, NOT (O+H+L+C)/4. This matches most institutional
   VWAP definitions.

8. **Bollinger Band std uses Polars `rolling_std`** which defaults to ddof=1 (sample std).

### Signal logic
9. **Entry masks should be conservative.** Multiple conditions ANDed together. A strategy
   that fires too often generates too many trades and gets filtered out.

10. **Signal exits fire on EVERY bar where the condition is true**, not just on transitions.
    The state machine only acts on signal_exit when in a position, so setting it True on
    every bar past a threshold is safe and simpler than detecting crossovers.

11. **For crossover detection** (e.g., "ADX crosses above 25"), use a loop comparing
    bar[i] vs bar[i-1]. NumPy boolean vectorization can also work:
    `cross_up = (arr[1:] > threshold) & (arr[:-1] <= threshold)`.

12. **`is_long_only = False`** for most strategies. If True, `short_entry` should be
    `np.zeros(n, dtype=np.bool_)`.

### Exit parameters
13. **If both `stop_loss_pct` and `stop_loss_atr_mult` are nonzero**, only `stop_loss_pct`
    is used (it's checked first in the state machine). Same for target. Don't set both
    unless you intend the pct to dominate.

14. **`use_target_indicator=True` overrides `target_pct` and `target_atr_mult`.** When set,
    the state machine uses the `target_indicator` array value on each bar as the target.
    For VWAP-target strategies, pass `target_indicator=vwap` and `use_target_indicator=True`.

15. **`time_stop_bars=0` means the default (120 bars).** Always set an explicit time stop
    appropriate for the strategy. Mean reversion strategies: 10-20. Trend following: 45-60+.

16. **`breakeven_pct`** moves the stop to entry price once profit exceeds this threshold.
    Common value: 0.0025 (0.25%). Only works for the `stop_loss_pct` stop, not ATR stop.

### Data handling
17. **VIX column may have NaN.** Replace with 99.0 (blocks VIX filters during missing data).

18. **Volume column may have NaN or zero.** Replace NaN with 1.0, then clip avg_vol to
    min 1.0 before division.

19. **`index_close` is NIFTY 50 close.** When computing relative returns or spreads vs index,
    guard against zero: `if index_close[i-N] > 0: ...`.

20. **Opening Range (OR)** is typically first 15 minutes: bars where `time_minutes < 570`.
    The OR high/low is the max(high)/min(low) of those bars within each day.

### Session times
21. **Mean reversion strategies** usually have narrow session windows (09:20-11:00) because
    they exploit opening volatility.

22. **Trend strategies** usually have wider windows (09:45-14:30+) because trends develop
    over longer periods.

23. **Closing-range strategies** use late session windows like `session_start=870, session_end=929`.

### Naming
24. **Module filename convention**: `{source}_{family}_{variant}.py`. Examples:
    - `cursor_opus46max_001.py` — Opus batch, strategy #1
    - `cursor_gpt54_vwap_overshoot_fade_001.py` — GPT54 batch, VWAP overshoot variant 1
    - `gemini31pro_gap_fade_reversion_v103.py` — Gemini batch, gap fade variant 103

25. **The `assumptions` list** documents any approximations or proxy logic used when the
    strategy concept requires data not available in the DataFrame:
    ```python
    assumptions = [
        "Sector rotation proxied via index_close relative return",
        "Institutional flow approximated from volume patterns",
    ]
    ```

### Performance
26. **Loops are acceptable** for indicator computation. The hot path (state machine) is
    already Numba-compiled. Strategy compute() runs once per stock per optimization trial.
    Don't prematurely optimize with Numba in strategy files.

27. **`max_lookback = 200`** is the default warmup. If your strategy uses longer indicators
    (e.g., 200-bar SMA), set `max_lookback` accordingly on the class. The state machine
    skips the first `warmup_bars` bars.
