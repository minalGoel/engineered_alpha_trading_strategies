# Bug Catalog

## Summary
33+ bugs discovered across 4 audit passes on the equity pipeline, plus 14 bugs on the options pipeline refactor, plus the critical strike-locking architecture flaw. Organized by category and severity. Every bug here was actually encountered and caused wrong results, crashes, or silent data corruption.

## Timestamp & Timezone Bugs

### UTC→IST Conversion (CRITICAL — found in Audit Pass 3)
**File**: `data_loader.py`
**What**: `replace_time_zone(None)` strips the UTC label WITHOUT converting to IST. All time-based logic operated on UTC hours — market open 09:15 IST appeared as 03:45.
**Impact**: Only 45/375 bars per day passed session checks. EOD flatten never triggered. Most strategies produced zero trades.
**Fix**: `convert_time_zone("Asia/Kolkata").replace_time_zone(None)`
**Lesson**: This survived TWO audit passes because nobody ran on real data until pass 3. Unit tests on synthetic data can't catch timezone bugs.

### Int8 Overflow in time_minutes (HIGH)
**File**: `data_loader.py`
**What**: Polars `dt.hour()` returns Int8 (max 127). `9 * 60 = 540` silently overflowed to 28.
**Fix**: Cast to Int32 before multiplication.

### EC2 Missing Timezone (HIGH)
**File**: `ec2_run.sh`
**What**: EC2 defaults to UTC. Pipeline constants assumed IST.
**Fix**: `export TZ=Asia/Kolkata` and install `tzdata`.

### run_local.sh Missing Timezone (MEDIUM)
**File**: `run_local.sh`
**What**: `TZ=Asia/Kolkata` set in ec2_run.sh but not run_local.sh.
**Fix**: Added `export TZ=Asia/Kolkata`.

## State Machine Bugs

### Short Mask Default to All-True (CRITICAL)
**File**: `backtester.py`
**What**: When a strategy has no short conditions (long-only), `short_mask` defaulted to `np.ones(n)` → shorted on EVERY bar.
**Fix**: `short_mask = np.zeros(n)` when no short conditions.

### ATR=0 Instant Stop-Out (HIGH)
**File**: `state_machine.py`
**What**: During warmup, `nan_to_num(nan=0.0)` converts NaN ATR to 0. ATR-based stop = `entry ± 0 = entry`. Since `low <= entry` is almost always true, stop triggers next bar.
**Fix**: Added `and atr_arr[i] > 0` check on all ATR-based stop/target calculations.

### Open Position at End of Data Dropped (CRITICAL)
**File**: `state_machine.py`
**What**: Position open at last bar is silently dropped — PnL never recorded.
**Fix**: Force-close open position at last bar with EOD reason.

### Session End Not Capped at EOD (HIGH)
**File**: `backtester.py`
**What**: `session_end_minutes` not capped at EOD flatten time — entries possible after 15:20.
**Fix**: `capped_session_end = min(effective_end, eod_flatten_minutes)`.

## Data Handling Bugs

### NaN in Price Arrays (HIGH)
**File**: `backtester.py`
**What**: NaN in OHLC arrays from missing data produces NaN PnL, corrupting all metrics.
**Fix**: Forward-fill NaN in Polars BEFORE converting to numpy. Use `pl.col(c).forward_fill()` not a Python loop (saves 33 minutes on full run).

### Duplicate Timestamps (HIGH)
**File**: `data_loader.py`
**What**: Raw data can have duplicate bars per timestamp, causing spurious trades.
**Fix**: `df.unique(subset=["datetime"], keep="last")`.

### Drawdown Sort Order (CRITICAL)
**File**: `metrics.py`
**What**: Drawdown computed on cumulative PnL in DataFrame iteration order, not chronological exit order.
**Fix**: Sort trades by `exit_time` before computing cumulative PnL.

## JSON/Serialization Bugs

### orjson + NaN/Infinity (CRITICAL)
**File**: `phases.py`, `backtester.py`
**What**: `orjson.dumps` crashes on NaN and Infinity values in metrics dicts.
**Fix**: Recursive `_sanitise_for_json()` that converts NaN→None, Infinity→None. Also added `math.isinf()` check (NaN filter `val == val` catches NaN but not Infinity since `inf == inf` is True).

### Profit Factor Divide-by-Zero (MEDIUM)
**File**: `metrics.py`
**What**: Zero losing trades → divide by zero.
**Fix**: Return `float('inf')` explicitly.

## Expiry & Calendar Bugs

### Weekday Thursday = 4, Not 3 (CRITICAL)
**File**: `backtester.py`
**What**: Polars weekday has Mon=1...Sun=7. Code used `weekday != 3` (Wednesday) instead of `weekday != 4` (Thursday) for expiry filtering.
**Fix**: Changed to `weekday != 4`.

## Optimization Bugs

### All Trials Fail → optimized_sharpe=-999 (HIGH)
**File**: `optimizer.py`
**What**: When all 100 Optuna trials fail (return -999), this propagates downstream.
**Fix**: Guard: `study.best_value > -900`, fall back to default params.

### study.best_trial Raises ValueError (HIGH)
**File**: `optimizer.py`
**What**: When ALL trials crash (not return bad value, but actually raise), accessing `study.best_trial` throws ValueError.
**Fix**: Wrapped in try/except ValueError.

### Improvement Ratio Division by Zero (HIGH)
**File**: `optimizer.py`
**What**: `improvement_ratio = optimized_sharpe / default_sharpe` with default_sharpe near zero.
**Fix**: Guard with `default_sharpe > 0.01`.

## Parallelism Bugs

### Numba Thread Contention (HIGH)
**File**: `run_all.py`
**What**: Without `NUMBA_NUM_THREADS=1`, Numba spawns internal threads per worker → 16×N thread contention.
**Fix**: `os.environ["NUMBA_NUM_THREADS"] = "1"` at import time.

### Nested joblib Deadlock Risk (HIGH)
**What**: Both outer (strategy) and inner (stock) pools using "loky" backend → process-inside-process deadlock on some systems.
**Fix**: Outer = loky, inner = threading.

## Pipeline Logic Bugs

### Phase Restart Reprocesses Everything (HIGH)
**File**: `run_all.py`
**What**: `--phase 3` reprocessed all 358 strategies even if already done.
**Fix**: Skip check for strategies with terminal verdicts on disk.

### Hardcoded trading_days=748 (HIGH)
**File**: `phases.py`
**What**: Phase restart hardcodes `trading_days = 748` instead of computing from actual data.
**Fix**: Estimate from actual stock data.

### Look-Ahead Bias in Indicator (CRITICAL)
**File**: `intraday_vol_pattern_v1.py`
**What**: Expected vol computed from ALL data including future bars.
**Fix**: Use only prior-day data for expected vol estimates.

## Options-Specific Bugs (Refactored Pipeline)

### Pipeline Never Actually Backtested (CRITICAL)
**File**: `run_all.py`
**What**: Only counted signals, never built option premium arrays or ran state machine.
**Fix**: Added `build_option_premium_arrays()` and `backtest_strategy()`.

### LOO-CV Was Fake (CRITICAL)
**File**: `optimizer.py`
**What**: Marked any day with signals as "profitable" instead of computing actual PnL.
**Fix**: CV now runs full backtest to get actual PnL.

### Optimizer Objective Was Signal Count (CRITICAL)
**File**: `optimizer.py`
**What**: Optimized for most signals, not best risk-adjusted returns.
**Fix**: Objective returns after-cost annualized Sharpe.

### Sharpe/Drawdown Used Gross PnL (HIGH — 3 instances)
**File**: `metrics.py`
**What**: Three separate places used pre-cost PnL when costs were active.
**Fix**: All use `net_pnl_quick()`.

### Test Suite 100% Broken (HIGH)
**File**: `tests/test_pipeline.py`
**What**: All 17 tests imported OLD equity modules that no longer existed.
**Fix**: Rewrote all tests for current option pipeline.

## Corrections Log
- The bug catalog grew from 8 (audit 1) → 25 (audit 2) → 29 (audit 3) → 33+ (audit 4) → 47+ (options refactor). Each pass found bugs the previous ones missed, primarily because each pass used a different methodology.
