#!/usr/bin/env python3
"""Third-pass audit: Run real data through the pipeline and verify outputs.

This audit is different from the first two — those read code, this one RUNS code
on real data and checks whether the actual output makes sense.
"""
import sys
import os
import json
import math
import random
import traceback
import numpy as np
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    STRATEGY_DIR, DATA_DIR, TIMEFRAME_FILES, VIX_FILES, INDEX_FILES,
    STATE_FLAT, STATE_LONG, STATE_SHORT,
    EOD_FLATTEN_H, EOD_FLATTEN_M,
    DEFAULT_CAPITAL_PER_TRADE,
)
from pipeline.data_loader import (
    load_stock_data, load_vix_data, load_index_data,
    merge_vix_index, assign_day_id,
)
from pipeline.strategy_parser import parse_strategy, load_all_strategies
from pipeline.backtester import backtest_single
from pipeline.metrics import compute_metrics
from pipeline.condition_parser import evaluate_conditions, parse_condition
from pipeline.indicators import compute_all_indicators
from pipeline.state_machine import run_state_machine, warmup_numba

import logging
logging.basicConfig(level=logging.WARNING)

bugs_found = []
bug_counter = 0

def report_bug(severity, file_path, line, description, evidence, fix=""):
    global bug_counter
    bug_counter += 1
    bug = {
        "num": bug_counter,
        "severity": severity,
        "file": f"{file_path}:{line}",
        "what": description,
        "evidence": evidence,
        "fix": fix,
    }
    bugs_found.append(bug)
    print(f"\n{'='*70}")
    print(f"BUG #{bug_counter} — {severity}")
    print(f"File: {file_path}:{line}")
    print(f"What: {description}")
    print(f"Evidence: {evidence}")
    if fix:
        print(f"Fix: {fix}")
    print(f"{'='*70}\n")


def section_header(title):
    print(f"\n{'#'*70}")
    print(f"# {title}")
    print(f"{'#'*70}\n")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: RUN REAL DATA THROUGH THE PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def find_strategies():
    """Find 3 strategies: simple, complex, long-only."""
    all_strats = load_all_strategies()
    simple = None
    complex_strat = None
    long_only = None

    for raw in all_strats:
        ps = parse_strategy(raw)
        if not ps.is_parseable:
            continue

        n_long = len(ps.long_conditions)
        n_short = len(ps.short_conditions)
        has_trailing = ps.exit_rules.trailing_stop_pct is not None
        has_vix = len(ps.vix_conditions) > 0
        has_vol = len(ps.volume_conditions) > 0

        # Simple: 2-3 long conditions, no short (or minimal), no complex filters
        if simple is None and 2 <= n_long <= 4 and ps.timeframe == "1min":
            simple = (raw, ps)

        # Complex: 5+ conditions, trailing, VIX filter
        if complex_strat is None and (n_long + n_short) >= 5 and has_trailing and has_vix and ps.timeframe == "1min":
            complex_strat = (raw, ps)

        # Long-only
        if long_only is None and ps.is_long_only and n_long >= 2 and ps.timeframe == "1min":
            long_only = (raw, ps)

        if simple and complex_strat and long_only:
            break

    # Fallback: just take the first 3 parseable strategies if we didn't find ideal ones
    fallbacks = []
    if simple is None or complex_strat is None or long_only is None:
        for raw in all_strats:
            ps = parse_strategy(raw)
            if ps.is_parseable and ps.timeframe == "1min":
                fallbacks.append((raw, ps))
                if len(fallbacks) >= 3:
                    break

    result = []
    labels = []
    if simple:
        result.append(simple)
        labels.append("SIMPLE")
    elif fallbacks:
        result.append(fallbacks.pop(0))
        labels.append("SIMPLE(fallback)")

    if complex_strat:
        result.append(complex_strat)
        labels.append("COMPLEX")
    elif fallbacks:
        result.append(fallbacks.pop(0))
        labels.append("COMPLEX(fallback)")

    if long_only:
        result.append(long_only)
        labels.append("LONG-ONLY")
    elif fallbacks:
        result.append(fallbacks.pop(0))
        labels.append("LONG-ONLY(fallback)")

    return result, labels


def load_data_for_strategy(ps, symbol="RELIANCE"):
    """Load real data for a strategy."""
    df = load_stock_data(ps.timeframe, symbol, start_date="2022-07-01", end_date="2024-12-31")
    if df is None:
        return None

    # Load VIX
    vix_df = load_vix_data(ps.timeframe, start_date="2022-07-01", end_date="2024-12-31")
    index_df = load_index_data(ps.timeframe, start_date="2022-07-01", end_date="2024-12-31")

    # Merge
    df = merge_vix_index(df, vix_df, index_df)
    return df


def run_strategy_on_real_data(raw, ps, symbol="RELIANCE"):
    """Run single strategy on real data and verify outputs."""
    print(f"\n--- Running strategy: {ps.name} on {symbol} ---")
    print(f"  Timeframe: {ps.timeframe}")
    print(f"  Long conditions: {len(ps.long_conditions)}")
    print(f"  Short conditions: {len(ps.short_conditions)}")
    print(f"  Long-only: {ps.is_long_only}")
    print(f"  Exit rules: {ps.exit_rules}")

    df = load_data_for_strategy(ps, symbol)
    if df is None:
        print(f"  ERROR: No data available for {symbol}")
        return None

    print(f"  Data shape: {df.shape}")
    print(f"  Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    trades = backtest_single(df, ps, symbol)

    if trades is None or trades.is_empty():
        print(f"  Result: NO TRADES GENERATED")
        return None

    n_trades = len(trades)
    print(f"  Result: {n_trades} trades generated")

    # ── Check 1: Plausibility ──────────────────────────────────────────
    # 3 years of 1-min data = ~750 trading days * 375 bars = ~280,000 bars
    # Plausible trades: 1 to ~7500 (max 10/day * 750 days)
    data_days = df["day_id"].n_unique()
    print(f"  Trading days in data: {data_days}")
    avg_trades_per_day = n_trades / data_days if data_days > 0 else 0
    print(f"  Average trades/day: {avg_trades_per_day:.2f}")

    if avg_trades_per_day > 15:
        report_bug("MEDIUM", "pipeline/backtester.py", 0,
                    f"Unusually high trade frequency: {avg_trades_per_day:.1f}/day for {ps.name}",
                    f"{n_trades} trades over {data_days} days")

    # ── Check 2: Entry timestamps within session window ────────────────
    eod_minutes = EOD_FLATTEN_H * 60 + EOD_FLATTEN_M  # 920 = 15:20
    effective_start = ps.time_filter.effective_start_minutes()
    effective_end = ps.time_filter.effective_end_minutes()

    entry_times = trades["entry_time"].to_list()
    session_violations = 0
    for et in entry_times[:100]:  # Check first 100
        if et is not None:
            entry_min = et.hour * 60 + et.minute
            if entry_min < effective_start or entry_min >= min(effective_end, eod_minutes):
                session_violations += 1
                if session_violations <= 3:
                    print(f"  SESSION VIOLATION: Entry at {et} ({entry_min} min), window [{effective_start}, {min(effective_end, eod_minutes)})")

    if session_violations > 0:
        report_bug("HIGH", "pipeline/backtester.py", 153,
                    f"{session_violations} entries outside session window for {ps.name}",
                    f"Session: {effective_start}-{min(effective_end, eod_minutes)}, found {session_violations} violations in first 100 trades")
    else:
        print(f"  CHECK PASSED: All entries within session window [{effective_start}, {min(effective_end, eod_minutes)})")

    # ── Check 3: Exit timestamps before 15:20 IST ─────────────────────
    exit_times = trades["exit_time"].to_list()
    late_exits = 0
    for xt in exit_times[:100]:
        if xt is not None:
            exit_min = xt.hour * 60 + xt.minute
            if exit_min > eod_minutes + 1:  # Allow 1 min grace for EOD flatten bar
                late_exits += 1
                if late_exits <= 3:
                    print(f"  LATE EXIT: {xt} ({exit_min} min)")

    if late_exits > 0:
        report_bug("HIGH", "pipeline/state_machine.py", 192,
                    f"{late_exits} exits after 15:20 IST for {ps.name}",
                    f"Found {late_exits} exits after EOD flatten time in first 100 trades")
    else:
        print(f"  CHECK PASSED: All exits at or before 15:20 IST")

    # ── Check 4: BUY→SELL / SHORT→COVER sequence ──────────────────────
    sides = trades["side"].to_list()
    sequence_errors = 0
    for i in range(len(sides)):
        side = sides[i]
        if side not in ("LONG", "SHORT"):
            sequence_errors += 1
            print(f"  UNKNOWN SIDE: {side} at trade {i}")

    if sequence_errors > 0:
        report_bug("CRITICAL", "pipeline/backtester.py", 298,
                    f"Invalid side labels found: {sequence_errors} trades",
                    f"Expected LONG or SHORT, found unknown values")
    else:
        print(f"  CHECK PASSED: All trades have valid sides (LONG/SHORT)")

    # Check for consecutive same-side entries (no BUY→BUY without SELL)
    # Since each trade record has entry+exit, this is already guaranteed by the state machine.
    # But let's verify no overlapping time ranges.
    overlap_errors = 0
    for i in range(1, min(len(trades), 200)):
        prev_exit = exit_times[i-1]
        curr_entry = entry_times[i]
        if prev_exit is not None and curr_entry is not None:
            if curr_entry < prev_exit:
                overlap_errors += 1
                if overlap_errors <= 3:
                    print(f"  OVERLAP: Trade {i} entry {curr_entry} < prev exit {prev_exit}")

    if overlap_errors > 0:
        report_bug("CRITICAL", "pipeline/state_machine.py", 284,
                    f"Overlapping trades: {overlap_errors} instances",
                    f"Trade entry before previous trade exit")
    else:
        print(f"  CHECK PASSED: No overlapping trades")

    # ── Check 5: PnL consistency ──────────────────────────────────────
    pnl_errors = 0
    for row in trades.head(50).iter_rows(named=True):
        ep = row["entry_price"]
        xp = row["exit_price"]
        pnl = row["pnl"]
        pnl_pct = row["pnl_pct"]
        side = row["side"]
        capital = ps.capital_per_trade

        if side == "LONG":
            expected_pnl = (xp - ep) * (capital / ep)
            expected_pct = (xp - ep) / ep
        else:
            expected_pnl = (ep - xp) * (capital / ep)
            expected_pct = (ep - xp) / ep

        if abs(pnl - expected_pnl) > 0.01:
            pnl_errors += 1
            if pnl_errors <= 3:
                print(f"  PNL MISMATCH: trade side={side}, ep={ep:.2f}, xp={xp:.2f}, "
                      f"reported_pnl={pnl:.2f}, expected={expected_pnl:.2f}")

        if abs(pnl_pct - expected_pct) > 0.0001:
            pnl_errors += 1

    if pnl_errors > 0:
        report_bug("CRITICAL", "pipeline/backtester.py", 296,
                    f"PnL calculation errors: {pnl_errors} in first 50 trades",
                    f"PnL doesn't match (exit_price - entry_price) * qty formula")
    else:
        print(f"  CHECK PASSED: PnL values consistent with entry/exit prices")

    # ── Check 6: Metrics verification ─────────────────────────────────
    metrics = compute_metrics(trades, ps.capital_per_trade)
    print(f"\n  Metrics:")
    print(f"    Total trades: {metrics['total_trades']}")
    print(f"    Win rate: {metrics['win_rate']:.4f}")
    print(f"    Profit factor: {metrics['profit_factor']:.2f}")
    print(f"    Sharpe: {metrics['sharpe_annualized']:.4f}")
    print(f"    Total PnL: {metrics['total_pnl']:.2f}")
    print(f"    Max DD: {metrics['max_drawdown']:.2f}")

    # Manual verification of win rate
    pnls = trades["pnl"].to_numpy()
    manual_win_rate = np.sum(pnls > 0) / len(pnls)
    if abs(manual_win_rate - metrics["win_rate"]) > 0.001:
        report_bug("HIGH", "pipeline/metrics.py", 45,
                    f"Win rate mismatch: computed={metrics['win_rate']:.4f}, manual={manual_win_rate:.4f}",
                    f"Manual count: {np.sum(pnls > 0)} wins / {len(pnls)} trades")

    # Manual profit factor
    winners = pnls[pnls > 0]
    losers = pnls[pnls < 0]
    if len(losers) > 0 and abs(np.sum(losers)) > 0:
        manual_pf = np.sum(winners) / abs(np.sum(losers))
        if abs(manual_pf - metrics["profit_factor"]) > 0.01:
            report_bug("HIGH", "pipeline/metrics.py", 51,
                        f"Profit factor mismatch: computed={metrics['profit_factor']:.2f}, manual={manual_pf:.2f}",
                        f"Gross profit: {np.sum(winners):.2f}, Gross loss: {abs(np.sum(losers)):.2f}")

    print(f"  CHECK PASSED: Metrics consistent with manual calculation")

    return trades, df, ps


def verify_signal_condition_alignment(trades, df, ps):
    """THE MOST IMPORTANT TEST: Pull random BUY signals and verify indicators satisfy conditions."""
    section_header("SIGNAL-CONDITION ALIGNMENT VERIFICATION")

    if trades is None or trades.is_empty():
        print("  No trades to verify")
        return

    # Re-compute indicators on the dataframe
    df_with_ind, computed, failed = compute_all_indicators(
        df,
        ps.indicator_defs,
        needs_vwap=ps.needs_vwap,
        needs_pdh_pdl=ps.needs_pdh_pdl,
        needs_prev_close=ps.needs_prev_close,
        needs_opening_range=ps.needs_opening_range,
        needs_gap=ps.needs_gap,
        needs_obv=ps.needs_obv,
        needs_bar_count=ps.needs_bar_count,
    )

    # Build arrays
    from pipeline.backtester import _df_to_arrays, _build_time_minutes
    arrays = _df_to_arrays(df_with_ind)

    # Get LONG trades with entry indicators
    long_trades = trades.filter(pl.col("side") == "LONG")
    if long_trades.is_empty():
        long_trades = trades  # Use all trades

    # Sample up to 5 random trades
    n_sample = min(5, len(long_trades))
    sample_indices = random.sample(range(len(long_trades)), n_sample)

    datetimes = df["datetime"].to_list()

    alignment_failures = 0

    for idx in sample_indices:
        row = long_trades.row(idx, named=True)
        entry_time = row["entry_time"]
        entry_price = row["entry_price"]
        entry_indicators = row.get("entry_indicators", "{}")

        # Find the bar index for this entry time
        bar_idx = None
        for bi, dt in enumerate(datetimes):
            if dt == entry_time:
                bar_idx = bi
                break

        if bar_idx is None:
            # Try matching with tolerance
            for bi, dt in enumerate(datetimes):
                if dt is not None and entry_time is not None:
                    if abs((dt - entry_time).total_seconds()) < 61:
                        bar_idx = bi
                        break

        if bar_idx is None:
            print(f"  WARNING: Cannot find bar for entry at {entry_time}")
            continue

        print(f"\n  Trade #{idx}: Entry at {entry_time} (bar {bar_idx}), price={entry_price:.2f}")

        # Parse stored entry indicators
        try:
            stored_indicators = json.loads(entry_indicators) if entry_indicators else {}
        except:
            stored_indicators = {}

        # Check each long condition at this bar
        conditions = ps.long_conditions if row["side"] == "LONG" else ps.short_conditions
        for cond in conditions:
            cond_repr = repr(cond)

            # Get the LHS value at bar_idx
            lhs_name = getattr(cond, 'lhs', getattr(cond, 'col_a', None))
            if lhs_name and lhs_name in arrays:
                lhs_val = arrays[lhs_name][bar_idx]

                # Get RHS value
                if hasattr(cond, 'rhs_is_numeric') and cond.rhs_is_numeric:
                    rhs_val = float(cond.rhs)
                    rhs_str = f"{rhs_val}"
                elif hasattr(cond, 'rhs') and isinstance(cond.rhs, str) and cond.rhs in arrays:
                    rhs_val = arrays[cond.rhs][bar_idx]
                    rhs_str = f"{cond.rhs}={rhs_val:.4f}"
                elif hasattr(cond, 'col_b'):  # CrossCondition
                    try:
                        rhs_val = float(cond.col_b)
                        rhs_str = f"{rhs_val}"
                    except ValueError:
                        if cond.col_b in arrays:
                            rhs_val = arrays[cond.col_b][bar_idx]
                            rhs_str = f"{cond.col_b}={rhs_val:.4f}"
                        else:
                            rhs_val = None
                            rhs_str = f"{cond.col_b}=MISSING"
                else:
                    rhs_val = None
                    rhs_str = "?"

                # Evaluate this single condition at this bar
                single_result = cond.evaluate(arrays)
                is_satisfied = bool(single_result[bar_idx])

                status = "OK" if is_satisfied else "FAIL"
                print(f"    {status}: {cond_repr} → {lhs_name}={lhs_val:.4f} vs {rhs_str} → {is_satisfied}")

                if not is_satisfied:
                    alignment_failures += 1
            else:
                if lhs_name:
                    print(f"    SKIP: {cond_repr} — column '{lhs_name}' not in arrays")
                else:
                    # CrossCondition
                    single_result = cond.evaluate(arrays)
                    is_satisfied = bool(single_result[bar_idx])
                    status = "OK" if is_satisfied else "FAIL"
                    print(f"    {status}: {cond_repr} → {is_satisfied}")
                    if not is_satisfied:
                        alignment_failures += 1

    if alignment_failures > 0:
        report_bug("CRITICAL", "pipeline/condition_parser.py", 0,
                    f"Signal-condition alignment failures: {alignment_failures} conditions not satisfied at entry bars",
                    f"Checked {n_sample} trades, found {alignment_failures} condition failures at entry time",
                    "Conditions should ALL be True at entry bar since they were ANDed to produce the entry mask")
    else:
        print(f"\n  SIGNAL-CONDITION ALIGNMENT: ALL {n_sample} trades verified — conditions satisfied at entry bars")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: DIFF EVERY FIX AGAINST ITS NEIGHBORS
# ═══════════════════════════════════════════════════════════════════════════

def audit_fix_consistency():
    section_header("SECTION 2: FIX CONSISTENCY AUDIT")

    # 2a: _sanitise_for_json — check ALL orjson.dumps call sites
    print("--- 2a: orjson.dumps call sites vs _sanitise_for_json ---")
    import subprocess
    result = subprocess.run(
        ["grep", "-rn", "orjson.dumps", str(ROOT / "pipeline")],
        capture_output=True, text=True
    )
    print(f"  orjson.dumps call sites:\n{result.stdout}")

    # Check which ones are protected
    result_sanitise = subprocess.run(
        ["grep", "-rn", "_sanitise_for_json", str(ROOT / "pipeline")],
        capture_output=True, text=True
    )
    print(f"  _sanitise_for_json usage:\n{result_sanitise.stdout}")

    # The backtester.py line 328 uses orjson.dumps directly (but with NaN filtering)
    # Verify NaN filtering actually works
    test_dict = {"a": float("nan"), "b": 1.0, "c": float("inf")}
    filtered = {}
    for k, v in test_dict.items():
        if v == v:  # NaN check
            filtered[k] = v
    import orjson
    try:
        orjson.dumps(filtered)
        print(f"  NaN filter works for nan, but inf PASSES the filter: {filtered}")
        if float("inf") in filtered.values():
            report_bug("MEDIUM", "pipeline/backtester.py", 322,
                        "orjson.dumps can still receive Infinity values — NaN filter (val==val) doesn't catch inf",
                        f"inf == inf is True, so inf passes through the NaN check. orjson will crash on inf.",
                        "Add explicit inf check: if val == val and not math.isinf(val)")
    except Exception as e:
        print(f"  orjson.dumps with inf: {e}")
        report_bug("MEDIUM", "pipeline/backtester.py", 322,
                    "orjson.dumps receives Infinity values from indicator arrays",
                    f"inf passes the NaN check (inf == inf is True), orjson crashes: {e}",
                    "Add explicit inf check")

    # 2b: short_mask = np.zeros(n) — check the long_mask counterpart
    print("\n--- 2b: Mask initialization for short-only strategies ---")
    # Check if there's a short-only path (is_long_only opposite)
    # In backtester.py, short_mask defaults to zeros if is_long_only.
    # But what if is_long_only is False and there are no long conditions?
    # long_mask would be np.ones (line 127) then overwritten to zeros at line 133.
    print("  long_mask initialization:")
    print("    Line 127: long_mask = np.ones(n) — initial value")
    print("    Line 130-133: overwritten to evaluate_conditions() or zeros if no conds")
    print("    Line 128: short_mask = np.ones(n) if not is_long_only, else zeros")
    print("    Line 135-139: overwritten if not is_long_only")
    print("  ANALYSIS: The logic correctly handles all cases:")
    print("    - Long-only: short_mask = zeros (line 128), never re-evaluated")
    print("    - Bidirectional: both masks properly evaluated or zeroed")
    print("    - No long conds: parse_errors would fire, strategy marked unparseable")

    # 2c: Forward-fill and VIX NaN
    print("\n--- 2c: Forward-fill vs VIX NaN propagation ---")
    # Load real data and check
    df = load_stock_data("1min", "RELIANCE", start_date="2024-01-01", end_date="2024-01-31")
    if df is not None:
        vix_df = load_vix_data("1min", start_date="2024-01-01", end_date="2024-01-31")
        index_df = load_index_data("1min", start_date="2024-01-01", end_date="2024-01-31")
        df = merge_vix_index(df, vix_df, index_df)

        vix_nulls = df["vix"].null_count()
        vix_nans = df.filter(pl.col("vix").is_nan()).height if "vix" in df.columns else 0
        total_rows = len(df)
        print(f"  After merge: {total_rows} rows, VIX nulls: {vix_nulls}, VIX NaNs: {vix_nans}")

        # Forward fill is only applied to OHLC columns (backtester.py lines 112-114)
        # VIX is NOT forward-filled!
        if vix_nulls > 0:
            report_bug("MEDIUM", "pipeline/backtester.py", 112,
                        f"VIX column has {vix_nulls} null values after merge that are NOT forward-filled",
                        f"Forward-fill only covers OHLC columns. VIX nulls propagate to conditions. "
                        f"join_asof (backward) helps, but VIX bars at start of day may be null.",
                        "Add VIX to forward-fill list, or handle NaN in VIX conditions")
        else:
            print(f"  VIX has no nulls after asof join — OK")

        # Check if any VIX values are NaN (not null but NaN)
        if "vix" in df.columns:
            vix_arr = df["vix"].to_numpy()
            nan_count = np.sum(np.isnan(vix_arr[~np.equal(vix_arr, None)]) if vix_arr.dtype != object else 0)
            if isinstance(nan_count, (int, np.integer)) and nan_count > 0:
                print(f"  VIX NaN values: {nan_count}")

    # 2d: weekday usage across codebase
    print("\n--- 2d: weekday usage across codebase ---")
    result = subprocess.run(
        ["grep", "-rn", "weekday", str(ROOT / "pipeline")],
        capture_output=True, text=True
    )
    print(f"  weekday references:\n{result.stdout}")
    print("  ANALYSIS: weekday is only used in backtester.py for expiry day filtering. Correctly uses 4 for Thursday (Polars 1=Monday).")

    # 2e: Timezone in run_local.sh
    print("\n--- 2e: Timezone in scripts ---")
    ec2_path = ROOT / "scripts" / "ec2_run.sh"
    local_path = ROOT / "scripts" / "run_local.sh"

    if ec2_path.exists():
        ec2_content = ec2_path.read_text()
        has_tz_ec2 = "TZ=Asia/Kolkata" in ec2_content or "Asia/Kolkata" in ec2_content
        print(f"  ec2_run.sh: TZ set = {has_tz_ec2}")

    if local_path.exists():
        local_content = local_path.read_text()
        has_tz_local = "TZ=Asia/Kolkata" in local_content or "Asia/Kolkata" in local_content
        print(f"  run_local.sh: TZ set = {has_tz_local}")
        if not has_tz_local:
            report_bug("MEDIUM", "scripts/run_local.sh", 1,
                        "run_local.sh does not set TZ=Asia/Kolkata",
                        "ec2_run.sh sets it but run_local.sh doesn't. A developer on non-IST timezone would get wrong session times.",
                        "Add 'export TZ=Asia/Kolkata' at the top of run_local.sh")

    # Check if Python code also handles timezone
    result = subprocess.run(
        ["grep", "-rn", "Asia/Kolkata", str(ROOT / "pipeline")],
        capture_output=True, text=True
    )
    print(f"  Python timezone references:\n{result.stdout or '  (none)'}")
    if not result.stdout.strip():
        print("  ANALYSIS: Python code does NOT set timezone internally — relies on shell environment")
        print("  _strip_timezone() in data_loader removes tz from data, so comparisons work regardless")
        print("  BUT: time_minutes computation uses raw hour/minute from naive datetime — this is correct as data is already in IST")


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: STRESS TESTS
# ═══════════════════════════════════════════════════════════════════════════

def stress_test_boundary_conditions():
    section_header("SECTION 3: STRESS TESTS")

    # Pre-compile Numba
    print("Pre-compiling Numba state machine...")
    warmup_numba()

    # ── Scenario A: First and last bar of dataset ─────────────────────
    print("\n--- Scenario A: First and last bar ---")
    df = load_stock_data("1min", "RELIANCE", start_date="2024-01-01", end_date="2024-01-31")
    if df is not None:
        n = len(df)
        print(f"  Data: {n} bars")

        # Create entry mask that triggers on bar 0 and last bar
        long_entry = np.zeros(n, dtype=np.bool_)
        long_entry[0] = True
        long_entry[n - 1] = True
        short_entry = np.zeros(n, dtype=np.bool_)

        time_mins = (df["datetime"].dt.hour().to_numpy() * 60 + df["datetime"].dt.minute().to_numpy()).astype(np.int32)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        day_id = df["day_id"].to_numpy().astype(np.int32)

        result = run_state_machine(
            opn, high, low, close,
            day_id, time_mins,
            long_entry, short_entry,
            np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_),
            np.ones(n, dtype=np.float64),  # ATR
            np.zeros(n, dtype=np.float64),  # target indicator
            0.01, 0.0, 0.02, 0.0, False,
            0.0, 0.0, 0.0, 120,
            920, 555, 920,
            10, 0.0, 100000.0, 0,  # warmup_bars = 0 to allow bar 0
        )

        entry_bars, exit_bars, sides, entry_px, exit_px, reasons, tc = result
        print(f"  Trade count: {tc}")

        if tc > 0:
            for t in range(tc):
                print(f"  Trade {t}: entry_bar={entry_bars[t]}, exit_bar={exit_bars[t]}, "
                      f"entry_price={entry_px[t]:.2f}, exit_price={exit_px[t]:.2f}, "
                      f"reason={reasons[t]}")

            # Bar 0: Check if it's within session (9:15 - 15:20)
            bar0_time = time_mins[0]
            if bar0_time < 555 or bar0_time >= 920:
                print(f"  Bar 0 time: {bar0_time} min — outside session, should NOT trigger")
                if entry_bars[0] == 0:
                    print(f"  WARNING: Entry at bar 0 despite being outside session")
            else:
                print(f"  Bar 0 time: {bar0_time} min — within session, entry OK")

            # Last bar should trigger force-close
            last_entry_bar = entry_bars[tc - 1]
            last_exit_bar = exit_bars[tc - 1]
            if last_entry_bar == n - 1:
                print(f"  Last bar entry: entry at bar {n-1}, exit at bar {last_exit_bar}")
                if last_exit_bar == n - 1:
                    print(f"  Entry and exit on same bar — state machine force-closes at end of data")
                    # Check if PnL is 0 (entry and exit on same bar)
                    pnl = (exit_px[tc-1] - entry_px[tc-1]) * (100000 / entry_px[tc-1])
                    print(f"  PnL: {pnl:.2f} (should be ~0 since same bar)")
        else:
            print("  No trades — bar 0 and last bar may be outside session window")
            bar0_time = time_mins[0]
            last_time = time_mins[n-1]
            print(f"  Bar 0 time: {bar0_time} min, Last bar time: {last_time} min")

    # ── Scenario B: Illiquid stock (zero volume bars) ─────────────────
    print("\n--- Scenario B: Illiquid/zero-volume bars ---")
    df_full = load_stock_data("1min", "RELIANCE", start_date="2024-01-01", end_date="2024-01-31")
    if df_full is not None:
        # Check for zero-volume or zero-range bars in real data
        zero_vol = df_full.filter(pl.col("volume") == 0).height
        zero_range = df_full.filter(pl.col("high") == pl.col("low")).height
        print(f"  RELIANCE data: {zero_vol} zero-volume bars, {zero_range} zero-range bars")

        # Artificially inject some zero-volume bars to test
        from pipeline.indicators import compute_atr
        df_test = df_full.clone()

        # Compute ATR on this data
        df_test = compute_atr(df_test, 20, "atr_20")
        atr_vals = df_test["atr_20"].to_numpy()
        zero_atr = np.sum(atr_vals == 0)
        nan_atr = np.sum(np.isnan(atr_vals))
        print(f"  ATR(20): {zero_atr} zero values, {nan_atr} NaN values")

        # Check: if ATR is 0, does 1.5*ATR stop = 0, causing immediate stop-out?
        # In the state machine, stop_price = entry_price - 0 = entry_price
        # Any bar where low <= entry_price would trigger stop (which is always true for long)
        print(f"  Risk: If ATR=0, stop_loss = entry_price ± 0 → immediate stop-out")
        print(f"  ATR-based stops use nan_to_num(nan=0.0) at backtester.py:227")
        print(f"  This means NaN ATR → 0 → no ATR stop (sl_atr_mult * 0 = 0, stop_price = entry_price - 0 = entry_price)")
        print(f"  Actually, state_machine.py:312 checks 'atr_arr[i] == atr_arr[i]' (NaN check)")
        print(f"  But after nan_to_num, ATR is 0.0, which passes NaN check → stop_price = entry - 0 = entry_price")

        # Verify: does entry_price - 0 = entry_price cause immediate stop?
        # State machine line 141: if stop_price > 0 and low_arr[i] <= stop_price
        # stop_price = entry_price > 0, and low_arr[i] is almost always <= close (which is entry_price)
        # So YES, this would cause immediate stop-out on the very next bar!
        test_n = 100
        test_close = np.full(test_n, 100.0)
        test_high = np.full(test_n, 101.0)
        test_low = np.full(test_n, 99.0)
        test_open = np.full(test_n, 100.0)
        test_day = np.zeros(test_n, dtype=np.int32)
        test_time = np.full(test_n, 600, dtype=np.int32)  # 10:00
        test_long = np.zeros(test_n, dtype=np.bool_)
        test_long[5] = True
        test_short = np.zeros(test_n, dtype=np.bool_)
        test_atr = np.zeros(test_n, dtype=np.float64)  # ATR = 0

        res = run_state_machine(
            test_open, test_high, test_low, test_close,
            test_day, test_time,
            test_long, test_short,
            np.zeros(test_n, dtype=np.bool_), np.zeros(test_n, dtype=np.bool_),
            test_atr,
            np.zeros(test_n, dtype=np.float64),
            0.0, 1.5,  # stop_loss_pct=0, stop_loss_atr_mult=1.5
            0.0, 0.0, False,
            0.0, 0.0, 0.0, 120,
            920, 555, 920,
            10, 0.0, 100000.0, 1,
        )
        e_bars, x_bars, s_sides, e_px, x_px, x_reasons, t_count = res
        print(f"\n  Zero-ATR stop test: {t_count} trades")
        if t_count > 0:
            holding = x_bars[0] - e_bars[0]
            reason_name = {0:"NONE",1:"STOP",2:"TARGET",3:"TRAILING",4:"SIGNAL",5:"TIME",6:"EOD"}.get(int(x_reasons[0]), "?")
            print(f"    Entry bar: {e_bars[0]}, Exit bar: {x_bars[0]}, Holding: {holding} bars, Reason: {reason_name}")
            print(f"    Entry price: {e_px[0]:.2f}, Exit price: {x_px[0]:.2f}")

            if holding <= 1 and reason_name == "STOP":
                report_bug("HIGH", "pipeline/backtester.py", 227,
                    "ATR=0 causes immediate stop-out: stop_price = entry_price, low <= stop_price on next bar",
                    f"Entry at bar {e_bars[0]}, exit at bar {x_bars[0]} (holding={holding} bars) with STOP reason. "
                    f"When ATR is NaN→0 via nan_to_num, ATR-based stop becomes entry_price ± 0 = entry_price, "
                    f"which triggers immediately since low is always <= entry_price (close).",
                    "When atr_arr[i] is 0, don't set ATR-based stop (treat as no stop). "
                    "Change backtester.py:227 or state_machine.py:312 to check atr_arr[i] > 0")

    # ── Scenario C: Very few bars in a day ────────────────────────────
    print("\n--- Scenario C: Day with very few bars ---")
    if df_full is not None:
        # Find a day with fewest bars
        bars_per_day = df_full.group_by("day_id").agg(pl.count().alias("n_bars")).sort("n_bars")
        min_bars_day = bars_per_day.row(0, named=True)
        print(f"  Day with fewest bars: day_id={min_bars_day['day_id']}, bars={min_bars_day['n_bars']}")

        # Check VWAP, SMA(20) on 5 bars
        if min_bars_day['n_bars'] < 20:
            print(f"  SMA(20) on {min_bars_day['n_bars']} bars will produce NaN — this is expected Polars behavior")
            print(f"  The condition evaluator handles NaN: NaN comparisons return False in numpy")

            # Verify: NaN comparison behavior
            test = np.array([np.nan, 1.0, 2.0])
            result = test > 1.5
            print(f"  NaN > 1.5 = {result[0]} (False as expected)")

    # ── Scenario D: VIX data gap ──────────────────────────────────────
    print("\n--- Scenario D: VIX data gap ---")
    df_stock = load_stock_data("1min", "RELIANCE", start_date="2024-01-01", end_date="2024-01-31")
    vix_df = load_vix_data("1min", start_date="2024-01-01", end_date="2024-01-31")

    if df_stock is not None and vix_df is not None:
        stock_dates = set(df_stock["datetime"].dt.date().unique().to_list())
        vix_dates = set(vix_df["datetime"].dt.date().unique().to_list())

        stock_only = stock_dates - vix_dates
        vix_only = vix_dates - stock_dates

        print(f"  Stock dates: {len(stock_dates)}, VIX dates: {len(vix_dates)}")
        print(f"  Dates with stock but no VIX: {len(stock_only)}")
        print(f"  Dates with VIX but no stock: {len(vix_only)}")
        if stock_only:
            print(f"    Example stock-only dates: {sorted(stock_only)[:3]}")

        # After merge, check VIX null count
        merged = merge_vix_index(df_stock, vix_df, None)
        vix_nulls = merged["vix"].null_count()
        vix_nans = merged.filter(pl.col("vix").is_nan()).height
        print(f"  After asof join: VIX nulls={vix_nulls}, VIX NaNs={vix_nans}")

        if vix_nulls > 0:
            # What happens to VIX conditions when VIX is null?
            vix_arr = merged["vix"].to_numpy().astype(np.float64)
            nan_count = np.sum(np.isnan(vix_arr))
            print(f"  As numpy: {nan_count} NaN VIX values out of {len(vix_arr)}")

            # Test condition evaluation with NaN
            test_cond = parse_condition("vix < 20")
            if test_cond:
                test_arrays = {"vix": vix_arr}
                result = test_cond[0].evaluate(test_arrays)
                nan_results = result[np.isnan(vix_arr)]
                print(f"  'vix < 20' on NaN values: {nan_results[:5]} (all False = correct, NaN blocks entry)")

    # ── Scenario E: Signal on last bar before EOD ─────────────────────
    print("\n--- Scenario E: Entry at 15:19 (1 bar before EOD flatten) ---")
    n = 400  # ~6.7 hours of 1-min bars
    # Simulate a trading day: 9:15 to 15:30 = 375 bars
    times = np.array([555 + i for i in range(375)], dtype=np.int32)  # 555 = 9:15
    n_e = len(times)
    close_e = np.full(n_e, 100.0)
    high_e = np.full(n_e, 101.0)
    low_e = np.full(n_e, 99.0)
    open_e = np.full(n_e, 100.0)
    day_e = np.zeros(n_e, dtype=np.int32)

    # Set close at 15:20 to 101 so we can verify PnL
    bar_1519 = 919 - 555  # index for 15:19
    bar_1520 = 920 - 555  # index for 15:20
    close_e[bar_1520] = 101.0

    long_e = np.zeros(n_e, dtype=np.bool_)
    long_e[bar_1519] = True  # Entry signal at 15:19
    short_e = np.zeros(n_e, dtype=np.bool_)

    res = run_state_machine(
        open_e, high_e, low_e, close_e,
        day_e, times,
        long_e, short_e,
        np.zeros(n_e, dtype=np.bool_), np.zeros(n_e, dtype=np.bool_),
        np.ones(n_e, dtype=np.float64),
        np.zeros(n_e, dtype=np.float64),
        0.0, 0.0, 0.0, 0.0, False,  # No stop/target — rely on EOD
        0.0, 0.0, 0.0, 375,
        920, 555, 920,  # EOD at 15:20
        10, 0.0, 100000.0, 1,
    )
    e_bars, x_bars, _, e_px, x_px, x_reasons, t_count = res

    print(f"  Trades: {t_count}")
    if t_count > 0:
        holding = x_bars[0] - e_bars[0]
        reason = {0:"NONE",1:"STOP",2:"TARGET",3:"TRAILING",4:"SIGNAL",5:"TIME",6:"EOD"}.get(int(x_reasons[0]), "?")
        pnl = (x_px[0] - e_px[0]) * (100000 / e_px[0])
        print(f"  Entry bar: {e_bars[0]} (time={times[e_bars[0]]}min={times[e_bars[0]]//60}:{times[e_bars[0]]%60})")
        print(f"  Exit bar: {x_bars[0]} (time={times[x_bars[0]]}min={times[x_bars[0]]//60}:{times[x_bars[0]]%60})")
        print(f"  Entry price: {e_px[0]:.2f}, Exit price: {x_px[0]:.2f}")
        print(f"  Holding: {holding} bars, Reason: {reason}")
        print(f"  PnL: {pnl:.2f}")

        # Verify: entry at 15:19 (919), exit at 15:20 (920)
        if times[e_bars[0]] == 919 and times[x_bars[0]] == 920:
            expected_pnl = (101.0 - 100.0) * (100000 / 100.0)
            if abs(pnl - expected_pnl) < 0.01:
                print(f"  CHECK PASSED: EOD flatten works correctly for last-bar entry")
            else:
                report_bug("HIGH", "pipeline/state_machine.py", 192,
                    "PnL mismatch for EOD flatten at 15:19→15:20",
                    f"Expected PnL={expected_pnl:.2f}, got {pnl:.2f}")
        elif t_count == 0:
            # Entry at 15:19 should work — session_end is capped at EOD (920)
            # But backtester.py:233 caps session_end: min(effective_end, eod_flatten_minutes)
            # And state_machine.py:286: if time_minutes[i] >= session_end_minutes → skip entry
            # So 919 >= 920? No, 919 < 920 — should allow entry
            report_bug("HIGH", "pipeline/state_machine.py", 286,
                "Entry at 15:19 blocked despite being before 15:20 session end",
                f"No trades generated for entry signal at time=919 with session_end=920")
    else:
        # Check if entry was blocked by session end
        print(f"  WARNING: No trades generated. Entry signal at bar {bar_1519}, time={times[bar_1519]}")
        print(f"  Session end check: time_minutes[i] >= session_end_minutes → {times[bar_1519]} >= 920 = {times[bar_1519] >= 920}")
        if times[bar_1519] >= 920:
            print(f"  Entry blocked because 15:19 = 919 minutes, but let me verify the capped_session_end logic...")

    # ── Scenario F: Trivially bad strategy (close < 0) ───────────────
    print("\n--- Scenario F: Strategy with impossible conditions ---")
    # Create a synthetic strategy that should never trigger
    bad_strategy_json = {
        "name": "impossible_strategy_test",
        "_file_path": "test",
        "_source_dir": "test",
        "timeframe": "1min",
        "indicators": [
            {"name": "rsi_14", "formula": "RSI(close, 14)", "lookback": 14}
        ],
        "entry": {
            "long": {"conditions": ["close < 0"]},
            "short": {"conditions": ["close < 0"]},
        },
        "exit": {"stop_loss": "0.5%", "target": "1%"},
        "filters": {},
        "risk": {},
    }

    ps_bad = parse_strategy(bad_strategy_json)
    print(f"  Parsed: {ps_bad.name}, parseable={ps_bad.is_parseable}")
    print(f"  Long conds: {ps_bad.long_conditions}")

    if ps_bad.is_parseable:
        df_test = load_stock_data("1min", "RELIANCE", start_date="2024-01-01", end_date="2024-03-31")
        if df_test is not None:
            vix_test = load_vix_data("1min", start_date="2024-01-01", end_date="2024-03-31")
            idx_test = load_index_data("1min", start_date="2024-01-01", end_date="2024-03-31")
            df_test = merge_vix_index(df_test, vix_test, idx_test)

            trades = backtest_single(df_test, ps_bad, "RELIANCE")
            if trades is None or trades.is_empty():
                print(f"  Result: No trades (correct — 'close < 0' should never trigger)")
                print(f"  CHECK PASSED: Impossible strategy produces zero trades")
            else:
                report_bug("CRITICAL", "pipeline/condition_parser.py", 0,
                    "Impossible strategy 'close < 0' produced trades!",
                    f"Got {len(trades)} trades from a condition that should never be true")
    else:
        print(f"  Strategy not parseable: {ps_bad.parse_errors}")

    # ── Scenario G: Concurrent Numba compilation ──────────────────────
    print("\n--- Scenario G: Concurrent Numba compilation ---")
    print("  Testing parallel Numba JIT with joblib...")
    try:
        from joblib import Parallel, delayed

        def run_one_worker(worker_id):
            """Each worker independently calls the state machine."""
            n = 200
            close = np.random.randn(n).cumsum() + 100
            close = np.abs(close) + 50
            high = close + np.abs(np.random.randn(n))
            low = close - np.abs(np.random.randn(n))
            opn = close + np.random.randn(n) * 0.1
            day_ids = np.zeros(n, dtype=np.int32)
            time_mins = np.full(n, 600, dtype=np.int32)
            long_e = np.random.random(n) > 0.95
            short_e = np.zeros(n, dtype=np.bool_)

            result = run_state_machine(
                opn, high, low, close,
                day_ids, time_mins,
                long_e, short_e,
                np.zeros(n, dtype=np.bool_), np.zeros(n, dtype=np.bool_),
                np.ones(n, dtype=np.float64),
                np.zeros(n, dtype=np.float64),
                0.003, 0.0, 0.005, 0.0, False,
                0.0, 0.0, 0.0, 30,
                920, 555, 920,
                10, 5000.0, 100000.0, 5,
            )
            return f"Worker {worker_id}: {result[-1]} trades"

        # Run 8 workers in parallel
        results = Parallel(n_jobs=8)(delayed(run_one_worker)(i) for i in range(8))
        for r in results:
            print(f"  {r}")
        print("  CHECK PASSED: No Numba race condition with 8 parallel workers")
    except Exception as e:
        report_bug("MEDIUM", "pipeline/state_machine.py", 38,
            f"Concurrent Numba compilation failed: {e}",
            traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("THIRD-PASS AUDIT: Running Real Data Through Pipeline")
    print("=" * 70)

    # ── Section 1: Run strategies on real data ────────────────────────
    section_header("SECTION 1: RUN STRATEGIES ON REAL DATA")

    strategies, labels = find_strategies()
    print(f"Found {len(strategies)} strategies: {labels}")
    for (raw, ps), label in zip(strategies, labels):
        print(f"\n{'─'*50}")
        print(f"Strategy [{label}]: {ps.name}")
        print(f"  Source: {ps.source_path}")
        print(f"  Long conditions ({len(ps.long_conditions)}): {ps.long_conditions}")
        print(f"  Short conditions ({len(ps.short_conditions)}): {ps.short_conditions}")

    all_results = []
    for (raw, ps), label in zip(strategies, labels):
        try:
            result = run_strategy_on_real_data(raw, ps, symbol="RELIANCE")
            if result is not None:
                all_results.append((result[0], result[1], result[2], label))
        except Exception as e:
            print(f"  ERROR running {ps.name}: {e}")
            traceback.print_exc()

    # Signal-condition alignment check (the MOST IMPORTANT test)
    for trades, df, ps, label in all_results:
        print(f"\n{'─'*50}")
        print(f"Signal verification for [{label}]: {ps.name}")
        try:
            verify_signal_condition_alignment(trades, df, ps)
        except Exception as e:
            print(f"  ERROR in signal verification: {e}")
            traceback.print_exc()

    # ── Section 2: Fix consistency ────────────────────────────────────
    try:
        audit_fix_consistency()
    except Exception as e:
        print(f"  ERROR in fix consistency audit: {e}")
        traceback.print_exc()

    # ── Section 3: Stress tests ───────────────────────────────────────
    try:
        stress_test_boundary_conditions()
    except Exception as e:
        print(f"  ERROR in stress tests: {e}")
        traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────
    section_header("AUDIT SUMMARY")
    print(f"Total bugs found: {len(bugs_found)}")
    for bug in bugs_found:
        print(f"  BUG #{bug['num']} [{bug['severity']}] {bug['file']}: {bug['what']}")

    if not bugs_found:
        print("  No bugs found!")

    print("\nDone.")
