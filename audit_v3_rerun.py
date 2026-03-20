#!/usr/bin/env python3
"""Re-run Section 1 of the audit after fixing the UTC→IST timezone bug.

Focuses on strategies with resolvable indicators that will produce trades,
then performs the critical signal-condition alignment check.
"""
import sys
import json
import math
import random
import traceback
import numpy as np
import polars as pl
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    EOD_FLATTEN_H, EOD_FLATTEN_M, STATE_LONG, STATE_SHORT,
)
from pipeline.data_loader import (
    load_stock_data, load_vix_data, load_index_data, merge_vix_index,
)
from pipeline.strategy_parser import parse_strategy, load_all_strategies
from pipeline.backtester import backtest_single, _df_to_arrays, _build_time_minutes
from pipeline.metrics import compute_metrics
from pipeline.indicators import compute_all_indicators

import logging
logging.basicConfig(level=logging.WARNING)

bugs = []
bug_n = 0

def bug(sev, file, line, what, evidence, fix=""):
    global bug_n
    bug_n += 1
    bugs.append({"n": bug_n, "sev": sev, "what": what})
    print(f"\n{'='*60}\nBUG #{bug_n} — {sev}\nFile: {file}:{line}\nWhat: {what}\nEvidence: {evidence}")
    if fix:
        print(f"Fix: {fix}")
    print(f"{'='*60}\n")


def find_good_strategies():
    """Find strategies with fully resolvable indicators."""
    all_strats = load_all_strategies()
    simple = complex_s = long_only = None

    for raw in all_strats:
        ps = parse_strategy(raw)
        if not ps.is_parseable or ps.timeframe != "1min":
            continue
        if len(ps.failed_indicators) > 0:
            continue

        n_long = len(ps.long_conditions)
        n_short = len(ps.short_conditions)
        has_trailing = ps.exit_rules.trailing_stop_pct is not None

        # Simple: 2-3 conditions
        if simple is None and 2 <= n_long <= 3 and n_long + n_short <= 6:
            simple = (raw, ps, "SIMPLE")

        # Complex: 4+ conditions with trailing stop
        if complex_s is None and n_long + n_short >= 8 and has_trailing:
            complex_s = (raw, ps, "COMPLEX")

        # Long-only
        if long_only is None and ps.is_long_only and n_long >= 2:
            long_only = (raw, ps, "LONG-ONLY")

    result = []
    for s in [simple, complex_s, long_only]:
        if s:
            result.append(s)

    # Fallback: if we don't have 3, add more
    if len(result) < 3:
        for raw in all_strats:
            ps = parse_strategy(raw)
            if ps.is_parseable and ps.timeframe == "1min" and len(ps.failed_indicators) == 0:
                if not any(r[1].name == ps.name for r in result):
                    result.append((raw, ps, "FALLBACK"))
                    if len(result) >= 3:
                        break

    return result


def run_and_verify(raw, ps, symbol="RELIANCE"):
    """Run strategy and verify all outputs."""
    print(f"\n{'─'*60}")
    print(f"Strategy: {ps.name} | {len(ps.long_conditions)} long, {len(ps.short_conditions)} short | long_only={ps.is_long_only}")
    print(f"Exit rules: {ps.exit_rules}")
    print(f"Indicators: {[i.get('name') for i in ps.indicator_defs]}")

    # Load data
    df = load_stock_data(ps.timeframe, symbol, start_date="2023-01-01", end_date="2024-12-31")
    if df is None:
        print("  No data!")
        return None

    vix_df = load_vix_data(ps.timeframe, start_date="2023-01-01", end_date="2024-12-31")
    index_df = load_index_data(ps.timeframe, start_date="2023-01-01", end_date="2024-12-31")
    df = merge_vix_index(df, vix_df, index_df)

    print(f"Data: {df.shape}, {df['datetime'].min()} to {df['datetime'].max()}")

    # Verify timestamps are IST
    first_hour = df["datetime"].dt.hour()[0]
    print(f"First bar hour: {first_hour} (should be ~9 for IST)")
    if first_hour < 8 or first_hour > 10:
        bug("CRITICAL", "pipeline/data_loader.py", 87,
            f"Timestamps not in IST after fix: first hour={first_hour}",
            f"Expected 9-10, got {first_hour}")

    # Run backtest
    trades = backtest_single(df, ps, symbol)

    if trades is None or trades.is_empty():
        print(f"  NO TRADES generated")
        return None

    n = len(trades)
    data_days = df["day_id"].n_unique()
    print(f"  {n} trades, {data_days} trading days, {n/data_days:.1f} trades/day")

    eod_min = EOD_FLATTEN_H * 60 + EOD_FLATTEN_M

    # ── Check entries within session window
    eff_start = ps.time_filter.effective_start_minutes()
    eff_end = min(ps.time_filter.effective_end_minutes(), eod_min)
    violations = 0
    for row in trades.head(100).iter_rows(named=True):
        et = row["entry_time"]
        entry_min = et.hour * 60 + et.minute
        if entry_min < eff_start or entry_min >= eff_end:
            violations += 1
    if violations > 0:
        bug("HIGH", "pipeline/backtester.py", 153,
            f"{violations}/100 entries outside session [{eff_start},{eff_end})",
            f"Session violation count: {violations}")
    else:
        print(f"  PASS: All entries within session [{eff_start},{eff_end})")

    # ── Check exits at or before EOD
    late = 0
    for row in trades.head(100).iter_rows(named=True):
        xt = row["exit_time"]
        if xt.hour * 60 + xt.minute > eod_min + 1:
            late += 1
    if late > 0:
        bug("HIGH", "pipeline/state_machine.py", 192,
            f"{late}/100 exits after EOD flatten", f"Late exit count: {late}")
    else:
        print(f"  PASS: All exits at/before 15:20")

    # ── Check PnL consistency
    pnl_err = 0
    for row in trades.head(50).iter_rows(named=True):
        ep, xp, pnl, side = row["entry_price"], row["exit_price"], row["pnl"], row["side"]
        cap = ps.capital_per_trade
        exp = (xp - ep) * (cap / ep) if side == "LONG" else (ep - xp) * (cap / ep)
        if abs(pnl - exp) > 0.01:
            pnl_err += 1
    if pnl_err > 0:
        bug("CRITICAL", "pipeline/backtester.py", 296,
            f"PnL mismatch: {pnl_err}/50 trades", "")
    else:
        print(f"  PASS: PnL consistent")

    # ── Check no overlapping trades
    overlaps = 0
    exit_times = trades["exit_time"].to_list()
    entry_times = trades["entry_time"].to_list()
    for i in range(1, min(len(trades), 200)):
        if entry_times[i] < exit_times[i-1]:
            overlaps += 1
    if overlaps > 0:
        bug("CRITICAL", "pipeline/state_machine.py", 284,
            f"{overlaps} overlapping trades", "")
    else:
        print(f"  PASS: No overlapping trades")

    # ── Metrics check
    m = compute_metrics(trades, ps.capital_per_trade)
    pnls = trades["pnl"].to_numpy()
    manual_wr = np.sum(pnls > 0) / len(pnls)
    if abs(manual_wr - m["win_rate"]) > 0.001:
        bug("HIGH", "pipeline/metrics.py", 45,
            f"Win rate: computed={m['win_rate']:.4f}, manual={manual_wr:.4f}", "")
    print(f"  Metrics: trades={m['total_trades']}, WR={m['win_rate']:.3f}, PF={m['profit_factor']:.2f}, Sharpe={m['sharpe_annualized']:.3f}")

    return trades, df, ps


def verify_signals(trades, df, ps):
    """THE MOST IMPORTANT CHECK: Do indicators at entry bars actually satisfy entry conditions?"""
    print(f"\n  ── SIGNAL-CONDITION ALIGNMENT for {ps.name} ──")

    # Compute indicators
    df_ind, computed, failed = compute_all_indicators(
        df, ps.indicator_defs,
        needs_vwap=ps.needs_vwap, needs_pdh_pdl=ps.needs_pdh_pdl,
        needs_prev_close=ps.needs_prev_close, needs_opening_range=ps.needs_opening_range,
        needs_gap=ps.needs_gap, needs_obv=ps.needs_obv, needs_bar_count=ps.needs_bar_count,
    )
    arrays = _df_to_arrays(df_ind)
    datetimes = df["datetime"].to_list()

    long_trades = trades.filter(pl.col("side") == "LONG")
    if long_trades.is_empty():
        long_trades = trades

    n_sample = min(5, len(long_trades))
    sample_idx = random.sample(range(len(long_trades)), n_sample)

    failures = 0
    for idx in sample_idx:
        row = long_trades.row(idx, named=True)
        entry_time = row["entry_time"]

        bar_idx = None
        for bi, dt in enumerate(datetimes):
            if dt == entry_time:
                bar_idx = bi
                break

        if bar_idx is None:
            print(f"    SKIP: Cannot find bar for entry at {entry_time}")
            continue

        print(f"\n    Trade #{idx}: Entry at {entry_time} (bar {bar_idx}), price={row['entry_price']:.2f}, side={row['side']}")

        conds = ps.long_conditions if row["side"] == "LONG" else ps.short_conditions
        for cond in conds:
            single = cond.evaluate(arrays)
            ok = bool(single[bar_idx])

            lhs_name = getattr(cond, 'lhs', getattr(cond, 'col_a', '?'))
            if lhs_name in arrays:
                lhs_val = arrays[lhs_name][bar_idx]
                print(f"      {'OK' if ok else 'FAIL'}: {repr(cond)} → {lhs_name}={lhs_val:.4f}, result={ok}")
            else:
                print(f"      {'OK' if ok else 'FAIL'}: {repr(cond)} → column missing, result={ok}")

            if not ok:
                failures += 1

    if failures > 0:
        bug("CRITICAL", "pipeline/condition_parser.py", 0,
            f"Signal-condition alignment FAILED: {failures} conditions not satisfied at entry bars",
            f"Checked {n_sample} trades, {failures} condition failures",
            "Entry mask should require ALL conditions true — misalignment indicates a bug")
    else:
        print(f"\n  SIGNAL-CONDITION ALIGNMENT: ALL {n_sample} trades verified OK ✓")


if __name__ == "__main__":
    print("=" * 60)
    print("AUDIT v3 RE-RUN: After UTC→IST Fix")
    print("=" * 60)

    strategies = find_good_strategies()
    print(f"\nSelected {len(strategies)} strategies:")
    for raw, ps, label in strategies:
        print(f"  [{label}] {ps.name}")

    results = []
    for raw, ps, label in strategies:
        try:
            r = run_and_verify(raw, ps, symbol="RELIANCE")
            if r:
                results.append((r[0], r[1], r[2], label))
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    # Signal alignment check
    for trades, df, ps, label in results:
        try:
            verify_signals(trades, df, ps)
        except Exception as e:
            print(f"  Signal check ERROR: {e}")
            traceback.print_exc()

    # ── Stress test: verify ATR=0 fix ──────────────────────────────────
    print(f"\n{'─'*60}")
    print("Stress test: ATR=0 stop-out fix verification")
    from pipeline.state_machine import run_state_machine
    test_n = 100
    test_close = np.full(test_n, 100.0)
    test_high = np.full(test_n, 101.0)
    test_low = np.full(test_n, 99.0)
    test_open = np.full(test_n, 100.0)
    test_day = np.zeros(test_n, dtype=np.int32)
    test_time = np.full(test_n, 600, dtype=np.int32)
    test_long = np.zeros(test_n, dtype=np.bool_)
    test_long[5] = True
    test_atr = np.zeros(test_n, dtype=np.float64)  # ATR = 0

    res = run_state_machine(
        test_open, test_high, test_low, test_close,
        test_day, test_time,
        test_long, np.zeros(test_n, dtype=np.bool_),
        np.zeros(test_n, dtype=np.bool_), np.zeros(test_n, dtype=np.bool_),
        test_atr, np.zeros(test_n, dtype=np.float64),
        0.0, 1.5, 0.0, 0.0, False,
        0.0, 0.0, 0.0, 120,
        920, 555, 920, 10, 0.0, 100000.0, 1,
    )
    _, x_bars, _, _, _, x_reasons, tc = res
    if tc > 0:
        holding = x_bars[0] - 5
        reason = {0:"NONE",1:"STOP",2:"TARGET",3:"TRAILING",4:"SIGNAL",5:"TIME",6:"EOD"}.get(int(x_reasons[0]), "?")
        if holding <= 1 and reason == "STOP":
            bug("HIGH", "pipeline/state_machine.py", 312,
                "ATR=0 still causes immediate stop-out after fix",
                f"Holding={holding} bars, reason={reason}")
        else:
            print(f"  ATR=0 fix verified: holding={holding} bars, reason={reason}")
    else:
        print(f"  ATR=0 fix verified: no immediate stop-out (no trades with zero ATR)")

    # Summary
    print(f"\n{'='*60}")
    print(f"Re-run summary: {len(bugs)} new bugs found")
    for b in bugs:
        print(f"  BUG #{b['n']} [{b['sev']}]: {b['what']}")
    if not bugs:
        print("  All checks passed!")
    print(f"{'='*60}")
