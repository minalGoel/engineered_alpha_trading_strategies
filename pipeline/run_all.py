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
from pipeline.state_machine import warmup_numba
from pipeline.leaderboard import build_leaderboard

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
            # Compute signals with default params
            default_params = {tp.name: tp.default for tp in strategy.tunable_params()}
            signals = strategy.compute(spot_df, option_df, vix_df, default_params)

            n_ce = int(signals.buy_ce.sum())
            n_pe = int(signals.buy_pe.sum())
            log.info("  Signals: %d CE, %d PE", n_ce, n_pe)

            all_results.append({
                "name": name,
                "verdict": "SMOKE_TEST_OK",
                "underlying": underlying,
                "ce_signals": n_ce,
                "pe_signals": n_pe,
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
