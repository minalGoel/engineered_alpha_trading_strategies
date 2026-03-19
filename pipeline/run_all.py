#!/usr/bin/env python3
"""Main pipeline entry point.

Usage:
    python pipeline/run_all.py
    python pipeline/run_all.py --parallel-strategies 16 --cores-per-strategy 4
    python pipeline/run_all.py --strategy vwap_mean_reversion_v1
    python pipeline/run_all.py --phase 3 --strategy my_strategy
    python pipeline/run_all.py --dry-run
"""
from __future__ import annotations
import os
import argparse
import logging
import sys
import time
import orjson
from pathlib import Path
from joblib import Parallel, delayed

# Prevent Numba from spawning its own thread pool inside each worker.
# Without this, 16 strategies × Numba's default threads = massive over-subscription.
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

from pipeline.config import (
    RESULTS_DIR, OUTPUTS_DIR,
    DEFAULT_PARALLEL_STRATEGIES, DEFAULT_CORES_PER_STRATEGY,
)
from pipeline.data_loader import build_data_inventory, get_available_symbols
from pipeline.strategy_parser import load_all_strategies, parse_strategy, build_indicator_coverage
from pipeline.phases import run_strategy_pipeline
from pipeline.leaderboard import build_leaderboard, build_master_signals
from pipeline.state_machine import warmup_numba

# ── Logging setup ───────────────────────────────────────────────────────────
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def main():
    parser = argparse.ArgumentParser(description="Engineered Alpha Trading Pipeline")
    parser.add_argument("--parallel-strategies", type=int,
                        default=DEFAULT_PARALLEL_STRATEGIES,
                        help="Number of strategies to run in parallel")
    parser.add_argument("--cores-per-strategy", type=int,
                        default=DEFAULT_CORES_PER_STRATEGY,
                        help="Cores per strategy for stock-level parallelism")
    parser.add_argument("--phase", type=int, default=1,
                        help="Start from this phase (load prior results from disk)")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Run only this strategy name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse everything, compute nothing")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("Engineered Alpha Trading Pipeline — Starting")
    log.info("parallel=%d, cores_per=%d, phase=%d, strategy=%s, dry_run=%s",
             args.parallel_strategies, args.cores_per_strategy,
             args.phase, args.strategy, args.dry_run)
    log.info("=" * 70)

    start_time = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Data discovery ─────────────────────────────────────────
    if args.phase <= 1:
        log.info("Phase 1: Data discovery and inventory")
        inventory = build_data_inventory()
        _save_json(OUTPUTS_DIR / "data_inventory.json", inventory)
        log.info("Phase 1 complete — %d symbols, timeframes: %s",
                 len(inventory.get("symbols", [])),
                 list(inventory.get("timeframes", {}).keys()))
    else:
        log.info("Phase 1: Loading from disk")
        inv_path = OUTPUTS_DIR / "data_inventory.json"
        if inv_path.exists():
            inventory = orjson.loads(inv_path.read_bytes())
        else:
            inventory = build_data_inventory()
            _save_json(OUTPUTS_DIR / "data_inventory.json", inventory)

    available_symbols = inventory.get("symbols", [])
    if not available_symbols:
        available_symbols = get_available_symbols("1min")
    log.info("Available symbols: %d", len(available_symbols))

    # ── Phase 2: Strategy parsing ───────────────────────────────────────
    if args.phase <= 2:
        log.info("Phase 2: Strategy parsing and indicator coverage")
        all_strategies = load_all_strategies()

        if args.strategy:
            all_strategies = [s for s in all_strategies if s.get("name") == args.strategy]
            if not all_strategies:
                log.error("Strategy '%s' not found", args.strategy)
                sys.exit(1)

        coverage = build_indicator_coverage(all_strategies)
        _save_json(OUTPUTS_DIR / "indicator_coverage.json", coverage)
        log.info("Phase 2 complete — %d parseable, %d unparseable",
                 coverage["parseable"], coverage["unparseable"])
    else:
        log.info("Phase 2: Loading from disk")
        all_strategies = load_all_strategies()
        if args.strategy:
            all_strategies = [s for s in all_strategies if s.get("name") == args.strategy]
        cov_path = OUTPUTS_DIR / "indicator_coverage.json"
        if cov_path.exists():
            coverage = orjson.loads(cov_path.read_bytes())

    if args.dry_run:
        log.info("DRY RUN — parsing complete, not executing backtests")
        # Print summary
        for strat in all_strategies:
            ps = parse_strategy(strat)
            status = "OK" if ps.is_parseable else "FAIL"
            log.info("  [%s] %s — long=%d, short=%d, params=%d, warnings=%d, errors=%d",
                     status, ps.name,
                     len(ps.long_conditions), len(ps.short_conditions),
                     len(ps.tunable_params),
                     len(ps.parse_warnings), len(ps.parse_errors))
        return

    # ── Warmup Numba ────────────────────────────────────────────────────
    log.info("Warming up Numba JIT compiler...")
    warmup_numba()
    log.info("Numba warmup complete")

    # ── Phases 3-7: Process strategies ──────────────────────────────────
    log.info("Phases 3-7: Processing %d strategies", len(all_strategies))

    # Filter to parseable strategies
    parseable = []
    for strat in all_strategies:
        ps = parse_strategy(strat)
        if ps.is_parseable:
            parseable.append(strat)
        elif args.phase <= 2:
            # Already handled in Phase 2 — create verdict for unparseable
            name = strat.get("name", "unknown")
            strat_dir = RESULTS_DIR / name
            strat_dir.mkdir(parents=True, exist_ok=True)
            _save_json(strat_dir / "strategy.json", strat)
            _save_json(strat_dir / "verdict.json", {
                "verdict": "FAILED_PARSE_ERROR",
                "reason": f"Parse errors: {ps.parse_errors}",
            })

    log.info("Parseable strategies: %d/%d", len(parseable), len(all_strategies))

    # Process strategies
    total = len(parseable)
    all_results = []

    # When restarting from a later phase, skip strategies that already have a verdict
    if args.phase > 1:
        to_process = []
        for strat in parseable:
            name = strat.get("name", "unknown")
            verdict_path = RESULTS_DIR / name / "verdict.json"
            if verdict_path.exists() and args.phase > 3:
                try:
                    existing = orjson.loads(verdict_path.read_bytes())
                    v = existing.get("verdict", "")
                    # Skip strategies that already reached a terminal verdict
                    if v in ("VALIDATED", "FAILED_OOS", "FAILED_NARROW",
                             "FAILED_OVERFIT", "FAILED_NO_TRADES", "FAILED_PARSE_ERROR"):
                        log.info("Skipping %s — already has verdict: %s", name, v)
                        all_results.append({"name": name, "verdict": v,
                                            "family": strat.get("_dedup_metadata", {}).get("family", ""),
                                            "tags": strat.get("tags", [])})
                        continue
                except Exception:
                    pass
            to_process.append(strat)
        log.info("After skip check: %d/%d strategies to process", len(to_process), len(parseable))
    else:
        to_process = list(parseable)

    total = len(to_process)

    if args.parallel_strategies <= 1 or args.strategy:
        # Sequential processing
        for idx, strat in enumerate(to_process, 1):
            result = run_strategy_pipeline(
                strat, available_symbols,
                strategy_idx=idx, total_strategies=total,
                n_cores=args.cores_per_strategy,
                start_phase=args.phase,
            )
            all_results.append(result)
    else:
        # Parallel processing using joblib
        def _process(strat, idx):
            return run_strategy_pipeline(
                strat, available_symbols,
                strategy_idx=idx, total_strategies=total,
                n_cores=args.cores_per_strategy,
                start_phase=args.phase,
            )

        results = Parallel(
            n_jobs=args.parallel_strategies,
            backend="loky",
            verbose=10,
        )(
            delayed(_process)(strat, idx)
            for idx, strat in enumerate(to_process, 1)
        )
        all_results = list(results)

    # Add unparseable strategies to results
    for strat in all_strategies:
        ps = parse_strategy(strat)
        if not ps.is_parseable:
            all_results.append({
                "name": strat.get("name", "unknown"),
                "verdict": "FAILED_PARSE_ERROR",
                "family": strat.get("_dedup_metadata", {}).get("family", ""),
                "tags": strat.get("tags", []),
            })

    # ── Phase 8: Leaderboard ────────────────────────────────────────────
    log.info("Phase 8: Building leaderboard")
    leaderboard_json, leaderboard_md, winning_json, family_summary = build_leaderboard(all_results)

    _save_json(OUTPUTS_DIR / "leaderboard.json", leaderboard_json)
    (OUTPUTS_DIR / "leaderboard.md").write_text(leaderboard_md, encoding="utf-8")
    _save_json(OUTPUTS_DIR / "winning_strategies.json", winning_json)
    _save_json(OUTPUTS_DIR / "family_summary.json", family_summary)

    # Build master signals parquet
    validated = [r for r in all_results if r["verdict"] == "VALIDATED"]
    if validated:
        import polars as pl
        master_signals = build_master_signals(validated)
        if master_signals is not None:
            master_signals.write_parquet(OUTPUTS_DIR / "master_signals.parquet")
            log.info("Master signals: %d signals from %d strategies",
                     len(master_signals), len(validated))

    # Save pipeline log
    pipeline_log = {
        "total_strategies": len(all_strategies),
        "parseable": len(parseable),
        "results": {r["name"]: r["verdict"] for r in all_results},
        "verdict_counts": {},
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    for r in all_results:
        v = r["verdict"]
        pipeline_log["verdict_counts"][v] = pipeline_log["verdict_counts"].get(v, 0) + 1
    _save_json(OUTPUTS_DIR / "pipeline_log.json", pipeline_log)

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    log.info("=" * 70)
    log.info("Pipeline complete in %.1f minutes", elapsed / 60)
    log.info("Verdict breakdown:")
    for verdict, count in sorted(pipeline_log["verdict_counts"].items()):
        log.info("  %s: %d", verdict, count)
    log.info("Validated strategies: %d", len(validated))
    if validated:
        log.info("Top 5:")
        sorted_valid = sorted(validated, key=lambda x: x.get("score", 0), reverse=True)
        for i, v in enumerate(sorted_valid[:5], 1):
            log.info("  %d. %s (score=%.2f)", i, v["name"], v.get("score", 0))
    log.info("=" * 70)


if __name__ == "__main__":
    main()
