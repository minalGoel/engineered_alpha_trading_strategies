#!/usr/bin/env python3
"""Main pipeline entry point for 5-second index option trading.

Usage:
    python pipeline/run_all.py
    python pipeline/run_all.py --parallel-strategies 8
    python pipeline/run_all.py --strategy tick_momentum_v1
    python pipeline/run_all.py --dry-run
"""
from __future__ import annotations
import os
import argparse
import logging
import sys
import time
import importlib
import orjson
from pathlib import Path

# Prevent Numba from spawning its own thread pool inside each worker.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from pipeline.config import (
    RESULTS_DIR, OUTPUTS_DIR, STRATEGY_DIR,
    DEFAULT_PARALLEL_STRATEGIES, DEFAULT_CORES_PER_STRATEGY,
    NIFTY_LOT, BANKNIFTY_LOT, NIFTY_SYMBOL, BANKNIFTY_SYMBOL,
)
from pipeline.data_loader import (
    build_data_inventory, load_all_data, add_atm_strike, integration_test,
)
from pipeline.option_utils import get_strike_step
from pipeline.state_machine import run_state_machine, warmup_numba, EXIT_REASON_MAP
from pipeline.leaderboard import build_leaderboard
from pipeline.metrics import compute_metrics
from pipeline.cost_model import get_lot_size

# ── Logging setup ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


def _save_json(path: Path, data):
    """Save dict to JSON, sanitising NaN/Infinity for orjson."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitised = _sanitise_for_json(data)
    path.write_bytes(orjson.dumps(sanitised, option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS))


def _sanitise_for_json(obj):
    """Recursively replace NaN/Infinity with None for orjson compatibility."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitise_for_json(v) for v in obj]
    return obj


import numpy as np
import polars as pl


def build_option_premium_arrays(
    spot_df: pl.DataFrame,
    option_df: pl.DataFrame,
    underlying: str,
) -> tuple:
    """Build per-bar ATM CE and PE option premium arrays aligned to spot bars.

    For each spot bar, looks up the ATM strike's nearest-expiry CE and PE
    close premium using an asof-join (last option bar at or before spot time).

    Returns (option_ce_close, option_pe_close, is_expiry) — all length n.
    """
    n = len(spot_df)
    option_ce_close = np.zeros(n, dtype=np.float64)
    option_pe_close = np.zeros(n, dtype=np.float64)
    is_expiry = np.zeros(n, dtype=np.bool_)

    if option_df is None or option_df.is_empty():
        return option_ce_close, option_pe_close, is_expiry

    expiry_dates = set(option_df["expiry"].unique().to_list())
    day_ids = spot_df["day_id"].to_numpy()
    atm_strike = spot_df["atm_strike"].to_numpy()
    datetimes = spot_df["datetime"].to_list()
    session_dates = spot_df["session_date"].to_list()

    # Mark expiry days
    for i in range(n):
        if session_dates[i] in expiry_dates:
            is_expiry[i] = True

    # Process day-by-day for efficiency
    for day_val in sorted(spot_df["day_id"].unique().to_list()):
        day_mask = day_ids == day_val
        day_indices = np.where(day_mask)[0]
        if len(day_indices) == 0:
            continue

        sd = session_dates[day_indices[0]]

        # Get options for this session date
        day_opts = option_df.filter(pl.col("session_date") == sd)
        if day_opts.is_empty():
            continue

        # Pick nearest expiry
        nearest_expiry = day_opts["expiry"].min()
        day_opts = day_opts.filter(pl.col("expiry") == nearest_expiry)

        # Get unique ATM strikes used on this day
        day_atm_set = set(atm_strike[day_indices])

        for strike_val in day_atm_set:
            for opt_type, out_arr in [("CE", option_ce_close), ("PE", option_pe_close)]:
                opts = day_opts.filter(
                    (pl.col("strike") == strike_val) &
                    (pl.col("option_type") == opt_type)
                ).sort("datetime")

                if opts.is_empty():
                    continue

                opt_times = opts["datetime"].to_list()
                opt_closes = opts["close"].to_numpy()

                # For each spot bar with this ATM strike, find the last option close
                strike_indices = day_indices[atm_strike[day_indices] == strike_val]
                opt_idx = 0
                for si in strike_indices:
                    bar_time = datetimes[si]
                    # Advance opt_idx to last entry at or before bar_time
                    while opt_idx < len(opt_times) - 1 and opt_times[opt_idx + 1] <= bar_time:
                        opt_idx += 1
                    if opt_idx < len(opt_times) and opt_times[opt_idx] <= bar_time:
                        out_arr[si] = opt_closes[opt_idx]

    return option_ce_close, option_pe_close, is_expiry


def backtest_strategy(strategy, spot_df, option_df, vix_df, lot_size, params=None):
    """Run full backtest: compute signals → state machine → metrics.

    Returns (trades_df, metrics_dict) or (None, empty_metrics) on failure.
    """
    from pipeline.config import (
        EOD_FLATTEN_H, EOD_FLATTEN_M,
        EXPIRY_FLATTEN_H, EXPIRY_FLATTEN_M,
    )
    from pipeline.metrics import _empty_metrics

    if params is None:
        params = {tp.name: tp.default for tp in strategy.tunable_params()}

    # Compute signals
    signals = strategy.compute(spot_df, option_df, vix_df, params)
    n = len(spot_df)

    # Build option premium arrays
    option_ce_close, option_pe_close, is_expiry_arr = build_option_premium_arrays(
        spot_df, option_df, strategy.underlying,
    )

    # Run state machine
    eod_mins = EOD_FLATTEN_H * 60 + EOD_FLATTEN_M
    exp_mins = EXPIRY_FLATTEN_H * 60 + EXPIRY_FLATTEN_M

    result = run_state_machine(
        spot_close=spot_df["close"].to_numpy().astype(np.float64),
        option_ce_close=option_ce_close,
        option_pe_close=option_pe_close,
        day_id=spot_df["day_id"].to_numpy().astype(np.int32),
        time_minutes=spot_df["time_minutes"].to_numpy().astype(np.int32),
        buy_ce=signals.buy_ce,
        buy_pe=signals.buy_pe,
        sell_ce=signals.sell_ce,
        sell_pe=signals.sell_pe,
        stop_points=signals.stop_points,
        target_points=signals.target_points,
        time_stop_bars=signals.time_stop_bars,
        eod_flatten_minutes=eod_mins,
        session_start_minutes=strategy.session_start_minutes,
        session_end_minutes=strategy.session_end_minutes,
        max_trades_per_day=signals.max_trades_per_day,
        lot_size=lot_size,
        warmup_bars=strategy.max_lookback,
        is_expiry=is_expiry_arr,
        expiry_flatten_minutes=exp_mins,
    )

    entry_bars, exit_bars, sides, entry_prems, exit_prems, exit_reasons, pnls, trade_count = result

    if trade_count == 0:
        return None, _empty_metrics()

    # Build trades DataFrame
    dt_list = spot_df["datetime"].to_list()
    trades_df = pl.DataFrame({
        "entry_bar": entry_bars[:trade_count],
        "exit_bar": exit_bars[:trade_count],
        "side": sides[:trade_count],
        "entry_premium": entry_prems[:trade_count],
        "exit_premium": exit_prems[:trade_count],
        "exit_reason": [EXIT_REASON_MAP.get(int(r), "?") for r in exit_reasons[:trade_count]],
        "pnl": pnls[:trade_count],
        "entry_time": [dt_list[int(b)] for b in entry_bars[:trade_count]],
        "exit_time": [dt_list[int(b)] for b in exit_bars[:trade_count]],
        "holding_bars": (exit_bars[:trade_count] - entry_bars[:trade_count]).astype(np.int64),
    })

    total_days = spot_df["day_id"].n_unique()
    metrics = compute_metrics(trades_df, lot_size=lot_size, total_trading_days=total_days)

    return trades_df, metrics


def _load_strategy_class(strategy_name: str):
    """Dynamically import a strategy class from pipeline/strategies/."""
    try:
        mod = importlib.import_module(f"pipeline.strategies.{strategy_name}")
        return mod.Strategy()
    except (ImportError, AttributeError) as e:
        log.warning("Cannot load strategy class '%s': %s", strategy_name, e)
        return None


def _get_qualified_strategies() -> list[dict]:
    """Load qualified strategy list from triage output."""
    triage_path = OUTPUTS_DIR / "strategy_triage.json"
    if not triage_path.exists():
        log.error("No strategy_triage.json found. Run Phase 1 first.")
        return []

    triage = orjson.loads(triage_path.read_bytes())
    return triage.get("qualified_strategies", [])


def main():
    parser = argparse.ArgumentParser(description="5-Second Option Trading Pipeline")
    parser.add_argument("--parallel-strategies", type=int,
                        default=DEFAULT_PARALLEL_STRATEGIES)
    parser.add_argument("--strategy", type=str, default=None,
                        help="Run only this strategy name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and strategies, compute nothing")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("5-Second Option Trading Pipeline — Starting")
    log.info("=" * 70)

    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Data discovery ──
    log.info("Phase 1: Data discovery and integration test")
    inventory = build_data_inventory()
    _save_json(OUTPUTS_DIR / "data_inventory.json", inventory)
    log.info("Phase 1 complete — spot=%s, option=%s",
             inventory.get("spot_exists"), inventory.get("option_exists"))

    # Integration test: verify UTC→IST conversion
    integration_test()

    # ── Load all data ──
    log.info("Loading all 5-second data...")
    all_data = load_all_data()
    log.info("Data loaded: %d trading days", all_data["trading_days"])

    # Add ATM strike columns
    for key, symbol in [("nifty_spot", NIFTY_SYMBOL), ("banknifty_spot", BANKNIFTY_SYMBOL)]:
        if all_data[key] is not None:
            step = get_strike_step(symbol)
            all_data[key] = add_atm_strike(all_data[key], step)

    # ── Phase 2: Load qualified strategies ──
    log.info("Phase 2: Loading qualified strategies")
    qualified = _get_qualified_strategies()
    if args.strategy:
        qualified = [s for s in qualified if s["name"] == args.strategy]
        if not qualified:
            log.error("Strategy '%s' not found in qualified list", args.strategy)
            sys.exit(1)

    log.info("Qualified strategies: %d", len(qualified))

    if args.dry_run:
        log.info("DRY RUN — loading complete, not executing backtests")
        for q in qualified:
            name = q["name"]
            strat = _load_strategy_class(name)
            status = "OK" if strat and not strat.unparseable else "FAIL"
            log.info("  [%s] %s — underlying=%s",
                     status, name, strat.underlying if strat else "?")
        return

    # ── Warmup Numba ──
    log.info("Warming up Numba JIT compiler...")
    warmup_numba()
    log.info("Numba warmup complete")

    # ── Process strategies ──
    log.info("Processing %d strategies", len(qualified))
    all_results = []

    for idx, q in enumerate(qualified, 1):
        name = q["name"]
        log.info("[%d/%d] Processing %s", idx, len(qualified), name)

        strategy = _load_strategy_class(name)
        if strategy is None:
            all_results.append({"name": name, "verdict": "FAILED_LOAD_ERROR"})
            continue

        # Determine which underlying data to use
        underlying = strategy.underlying.upper()
        if underlying in ("NIFTY", "NSE:NIFTY50-INDEX"):
            spot_df = all_data["nifty_spot"]
            option_df = all_data["nifty_options"]
            lot_size = NIFTY_LOT
        elif underlying in ("BANKNIFTY", "NSE:NIFTYBANK-INDEX"):
            spot_df = all_data["banknifty_spot"]
            option_df = all_data["banknifty_options"]
            lot_size = BANKNIFTY_LOT
        else:
            # BOTH — run on NIFTY by default, can be extended
            spot_df = all_data["nifty_spot"]
            option_df = all_data["nifty_options"]
            lot_size = NIFTY_LOT

        vix_df = all_data["vix"]

        if spot_df is None or option_df is None:
            log.warning("No data for %s underlying=%s", name, underlying)
            all_results.append({"name": name, "verdict": "FAILED_NO_DATA"})
            continue

        try:
            # Run full backtest
            trades_df, metrics = backtest_strategy(
                strategy, spot_df, option_df, vix_df, lot_size,
            )

            n_trades = metrics.get("total_trades", 0)
            sharpe = metrics.get("sharpe_annualized", 0.0)
            total_pnl = metrics.get("total_pnl", 0.0)
            log.info("  Trades: %d, Sharpe: %.4f, Net PnL: INR %.0f",
                     n_trades, sharpe, total_pnl)

            all_results.append({
                "name": name,
                "verdict": "BACKTEST_OK" if n_trades > 0 else "ZERO_TRADES",
                "underlying": underlying,
                "metrics": metrics,
            })

        except Exception as e:
            log.error("  Error: %s", e)
            all_results.append({
                "name": name,
                "verdict": "FAILED_COMPUTE_ERROR",
                "error": str(e),
            })

    # ── Summary ──
    elapsed = time.time() - start_time
    log.info("=" * 70)
    log.info("Pipeline complete in %.1f seconds", elapsed)
    verdict_counts = {}
    for r in all_results:
        v = r["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    for verdict, count in sorted(verdict_counts.items()):
        log.info("  %s: %d", verdict, count)
    log.info("=" * 70)

    _save_json(OUTPUTS_DIR / "pipeline_log.json", {
        "total_strategies": len(qualified),
        "results": {r["name"]: r["verdict"] for r in all_results},
        "verdict_counts": verdict_counts,
        "elapsed_seconds": round(elapsed, 1),
    })


if __name__ == "__main__":
    main()
