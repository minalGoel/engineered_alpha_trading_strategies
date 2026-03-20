#!/usr/bin/env python3
"""Audit script for batch: strategies cursor_opus46max 013-022."""

import sys, os, importlib, traceback
import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DATA_PATH = os.path.join(ROOT, "data", "stocks_1min.parquet")

STRATEGIES = [
    "cursor_opus46max_013",
    "cursor_opus46max_014",
    "cursor_opus46max_015",
    "cursor_opus46max_016",
    "cursor_opus46max_017",
    "cursor_opus46max_018",
    "cursor_opus46max_019",
    "cursor_opus46max_020",
    "cursor_opus46max_021",
    "cursor_opus46max_022",
]

# ── Load and prepare data ──────────────────────────────────────────────────────
print("Loading data...")
df_raw = pl.read_parquet(DATA_PATH)

# Pick one stock
ticker = df_raw["symbol"].unique().sort().to_list()[0]
df = df_raw.filter(pl.col("symbol") == ticker).sort("timestamp")
print(f"Using ticker: {ticker}, rows: {len(df)}")

# day_id from date
df = df.with_columns(
    pl.col("timestamp").dt.date().rank("dense").cast(pl.Int32).alias("day_id")
)

# time_minutes in IST (UTC+5:30 = +330 minutes)
# IMPORTANT: cast to Int32 BEFORE multiplication to avoid Int8 overflow
df = df.with_columns(
    (pl.col("timestamp").dt.hour().cast(pl.Int32) * 60
     + pl.col("timestamp").dt.minute().cast(pl.Int32)
     + 330).alias("time_minutes")
)

# Add VIX - filter to single symbol (e.g., INDIA VIX)
df_vix_raw = pl.read_parquet(os.path.join(ROOT, "data", "india_vix_1min.parquet"))
df_vix = df_vix_raw.sort("timestamp").select([
    pl.col("timestamp").alias("__vix_dt"),
    pl.col("close").alias("vix")
])
df = df.join(df_vix, left_on="timestamp", right_on="__vix_dt", how="left")
df = df.with_columns(pl.col("vix").fill_null(15.0))

# Add index_close - use NIFTY only to avoid 5x duplicate join
df_idx_raw = pl.read_parquet(os.path.join(ROOT, "data", "index_1min.parquet"))
# Use NIFTY (the main index)
df_idx = df_idx_raw.filter(pl.col("symbol") == "NIFTY").sort("timestamp").select([
    pl.col("timestamp").alias("__idx_dt"),
    pl.col("close").alias("index_close")
])
df = df.join(df_idx, left_on="timestamp", right_on="__idx_dt", how="left")
df = df.with_columns(
    pl.col("index_close").fill_null(strategy="forward").fill_null(20000.0)
)

print(f"Final columns: {df.columns}")
print(f"time_minutes range: {df['time_minutes'].min()} - {df['time_minutes'].max()}")
print(f"Bars per day check (should be ~375): {len(df) / df['day_id'].n_unique():.0f}")

# Use 60 days for adequate signal statistics
day_ids = df["day_id"].unique().sort().to_list()
print(f"Total days: {len(day_ids)}, using first 60")
df = df.filter(pl.col("day_id").is_in(day_ids[:60]))
print(f"Filtered shape: {df.shape}")

n_df = len(df)


def check_signals(name, sig, n, strat):
    issues = []

    arrays = {
        "long_entry": sig.long_entry,
        "short_entry": sig.short_entry,
        "signal_exit_long": sig.signal_exit_long,
        "signal_exit_short": sig.signal_exit_short,
        "atr_arr": sig.atr_arr,
        "target_indicator": sig.target_indicator,
    }

    # 1. Array lengths
    for aname, arr in arrays.items():
        if len(arr) != n:
            issues.append(f"LENGTH: {aname} len={len(arr)} != {n}")

    # 2. Dtypes
    for aname in ["long_entry", "short_entry", "signal_exit_long", "signal_exit_short"]:
        arr = arrays[aname]
        if arr.dtype not in [np.bool_, np.int8]:
            issues.append(f"DTYPE: {aname} dtype={arr.dtype} (expected bool/int8)")

    for aname in ["atr_arr", "target_indicator"]:
        arr = arrays[aname]
        if arr.dtype != np.float64:
            issues.append(f"DTYPE: {aname} dtype={arr.dtype} (expected float64)")

    # 3. No NaN in scalar params
    for fname in ["stop_loss_pct", "target_pct", "trailing_stop_pct", "breakeven_pct"]:
        val = getattr(sig, fname, 0.0)
        if np.isnan(val):
            issues.append(f"NAN_SCALAR: {fname}={val}")

    # 3b. No NaN in float arrays
    for aname in ["atr_arr", "target_indicator"]:
        arr = arrays[aname]
        nans = int(np.sum(np.isnan(arr)))
        if nans > 0:
            issues.append(f"NAN: {aname} has {nans} NaN values")

    # 4. No NaN in entry/exit masks
    for aname in ["long_entry", "short_entry", "signal_exit_long", "signal_exit_short"]:
        arr = arrays[aname]
        arr_f = arr.astype(np.float64)
        nans = int(np.sum(np.isnan(arr_f)))
        if nans > 0:
            issues.append(f"NAN: {aname} has {nans} NaN values")

    # 5. Signal sanity
    le_count = int(np.sum(sig.long_entry))
    se_count = int(np.sum(sig.short_entry))

    if not strat.unparseable:
        le_pct = le_count / n * 100
        se_pct = se_count / n * 100
        if le_count == 0:
            issues.append(f"ZERO_SIGNALS: long_entry has 0 True values")
        elif le_pct > 50:
            issues.append(f"TOO_MANY_SIGNALS: long_entry={le_pct:.1f}% of bars")
        if se_count == 0 and not strat.is_long_only:
            issues.append(f"ZERO_SIGNALS: short_entry has 0 True values")
        elif se_pct > 50 and not strat.is_long_only:
            issues.append(f"TOO_MANY_SIGNALS: short_entry={se_pct:.1f}% of bars")

    # 8. is_long_only consistency
    if strat.is_long_only:
        if np.any(sig.short_entry):
            issues.append("LONG_ONLY_VIOLATION: short_entry has True values but is_long_only=True")

    return issues, le_count, se_count


# ── Run each strategy ──────────────────────────────────────────────────────────
for strat_name in STRATEGIES:
    print(f"\n{'='*60}")
    print(f"Auditing: {strat_name}")
    print(f"{'='*60}")

    try:
        mod = importlib.import_module(f"pipeline.strategies.{strat_name}")
        strat = mod.Strategy()
        print(f"  Import OK | unparseable={strat.unparseable}")
    except Exception as e:
        print(f"  IMPORT FAILED: {e}")
        traceback.print_exc()
        print(f"AUDIT_RESULT: {strat_name} STATUS=UNFIXABLE SIGNALS=0,0 ISSUES=1")
        continue

    if strat.unparseable:
        try:
            params = {p.name: p.default for p in strat.tunable_params()}
            sig = strat.compute(df, params)
            issues, le_count, se_count = check_signals(strat_name, sig, n_df, strat)
            status = "PASS" if not issues else "FIXED"
            print(f"  unparseable=True, signals={le_count},{se_count}, issues={issues}")
            print(f"AUDIT_RESULT: {strat_name} STATUS={status} SIGNALS={le_count},{se_count} ISSUES={len(issues)}")
        except Exception as e:
            print(f"  compute() FAILED: {e}")
            print(f"AUDIT_RESULT: {strat_name} STATUS=UNFIXABLE SIGNALS=0,0 ISSUES=1")
        continue

    try:
        params = {p.name: p.default for p in strat.tunable_params()}
        sig = strat.compute(df, params)
        print(f"  compute() OK")
    except Exception as e:
        print(f"  compute() FAILED: {e}")
        traceback.print_exc()
        print(f"AUDIT_RESULT: {strat_name} STATUS=UNFIXABLE SIGNALS=0,0 ISSUES=1")
        continue

    issues, le_count, se_count = check_signals(strat_name, sig, n_df, strat)

    print(f"  long_entry={le_count} ({le_count/n_df*100:.2f}%)")
    print(f"  short_entry={se_count} ({se_count/n_df*100:.2f}%)")
    print(f"  stop_loss_pct={sig.stop_loss_pct:.5f}, target_pct={sig.target_pct:.5f}")
    print(f"  atr_arr non-zero: {np.sum(sig.atr_arr > 0)}")

    if issues:
        for issue in issues:
            print(f"  BUG: {issue}")
        status = "FIXED"
    else:
        status = "PASS"

    print(f"AUDIT_RESULT: {strat_name} STATUS={status} SIGNALS={le_count},{se_count} ISSUES={len(issues)}")

print("\nDone.")
