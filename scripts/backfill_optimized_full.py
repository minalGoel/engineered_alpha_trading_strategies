"""One-time backfill: generate 'optimized_full' runs for all strategies.

Recovers the consensus (optimized) params from existing sensitivity runs
and runs a full-period backtest, storing results in the ResultStore.

Usage:
    python scripts/backfill_optimized_full.py              # all strategies
    python scripts/backfill_optimized_full.py --strategy vwap_rsi_volume_hybrid_v1  # single
    python scripts/backfill_optimized_full.py --dry-run     # preview only
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def _recover_consensus_params(store, strategy_id: str) -> dict[str, float] | None:
    """Recover the consensus (optimized) baseline params from sensitivity runs.

    Each sensitivity run perturbs ONE param by ±20%. The unperturbed value
    (appearing in 10+ of 12 sensitivity runs) is the consensus baseline.
    """
    import sqlite3
    with store._connect() as conn:
        rows = conn.execute(
            """
            SELECT p.param_name, p.param_value, COUNT(*) as freq
            FROM parameters p
            JOIN runs r ON p.run_id = r.run_id
            WHERE r.strategy_id = ? AND r.is_sensitivity_run = 1
            GROUP BY p.param_name, p.param_value
            ORDER BY p.param_name, freq DESC
            """,
            (strategy_id,),
        ).fetchall()

    if not rows:
        return None

    # For each param, pick the value with the highest frequency (the unperturbed baseline)
    params: dict[str, float] = {}
    for row in rows:
        name = row["param_name"]
        if name not in params:  # first row per param = highest freq
            params[name] = float(row["param_value"])

    return params if params else None


def _strategies_needing_backfill(store, single: str | None = None) -> list[str]:
    """Find strategies that have sensitivity runs but no optimized_full run."""
    with store._connect() as conn:
        if single:
            # Check if this specific strategy needs backfill
            has_opt = conn.execute(
                "SELECT 1 FROM runs WHERE strategy_id = ? AND notes = 'optimized_full' LIMIT 1",
                (single,),
            ).fetchone()
            if has_opt:
                return []
            has_sens = conn.execute(
                "SELECT 1 FROM runs WHERE strategy_id = ? AND is_sensitivity_run = 1 LIMIT 1",
                (single,),
            ).fetchone()
            return [single] if has_sens else []

        rows = conn.execute(
            """
            SELECT DISTINCT r.strategy_id
            FROM runs r
            WHERE r.is_sensitivity_run = 1
              AND r.strategy_id NOT IN (
                  SELECT strategy_id FROM runs WHERE notes = 'optimized_full'
              )
            ORDER BY r.strategy_id
            """
        ).fetchall()
    return [r["strategy_id"] for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Backfill optimized_full runs")
    parser.add_argument("--strategy", type=str, default=None, help="Single strategy to backfill")
    parser.add_argument("--dry-run", action="store_true", help="Preview without running backtests")
    args = parser.parse_args()

    from pipeline.result_store import get_store
    from pipeline.data_loader import load_all_data, add_atm_strike
    from pipeline.option_utils import get_strike_step
    from pipeline.config import NIFTY_SYMBOL, BANKNIFTY_SYMBOL
    from pipeline.run_all import (
        backtest_strategy, _save_strategy_run, _resolve_data, _load_strategy_class,
    )
    from pipeline.state_machine import warmup_numba

    store = get_store()
    candidates = _strategies_needing_backfill(store, args.strategy)

    if not candidates:
        log.info("No strategies need backfill (all have optimized_full runs already).")
        return

    log.info("Found %d strategies needing optimized_full backfill", len(candidates))

    if args.dry_run:
        for sid in candidates:
            params = _recover_consensus_params(store, sid)
            log.info("  [DRY] %s → %d params recovered: %s",
                     sid, len(params) if params else 0, params)
        return

    # Load market data once
    log.info("Loading 5-second market data...")
    all_data = load_all_data()
    for key, symbol in [("nifty_spot", NIFTY_SYMBOL), ("banknifty_spot", BANKNIFTY_SYMBOL)]:
        if all_data[key] is not None:
            step = get_strike_step(symbol)
            all_data[key] = add_atm_strike(all_data[key], step)

    warmup_numba()
    vix_df = all_data["vix"]

    success = 0
    failed = 0
    skipped = 0

    for i, sid in enumerate(candidates, 1):
        log.info("[%d/%d] Processing %s", i, len(candidates), sid)

        # Recover consensus params
        params = _recover_consensus_params(store, sid)
        if not params:
            log.warning("  Skipped — no sensitivity params found")
            skipped += 1
            continue

        # Load strategy
        strategy = _load_strategy_class(sid)
        if strategy is None:
            log.warning("  Skipped — strategy class not found")
            skipped += 1
            continue

        # Resolve data
        try:
            spot_df, option_df, lot_size = _resolve_data(strategy, all_data)
        except Exception as e:
            log.warning("  Skipped — data resolution failed: %s", e)
            skipped += 1
            continue

        if spot_df is None or option_df is None:
            log.warning("  Skipped — no market data for %s", strategy.underlying)
            skipped += 1
            continue

        # Run backtest
        try:
            trades_df, metrics = backtest_strategy(
                strategy, spot_df, option_df, vix_df, lot_size, params,
            )
        except Exception as e:
            log.error("  FAILED backtest: %s", e)
            failed += 1
            continue

        if trades_df is None or trades_df.is_empty():
            log.warning("  Skipped — no trades produced")
            skipped += 1
            continue

        # Save
        try:
            rid = _save_strategy_run(
                strategy, trades_df, metrics, spot_df, option_df, vix_df,
                lot_size, params, notes="optimized_full",
                split="full", is_optimized=True,
            )
            if rid:
                sharpe = metrics.get("sharpe_annualized", 0)
                total_pnl = metrics.get("total_pnl", 0)
                n_trades = metrics.get("total_trades", 0)
                log.info("  OK — Sharpe %.2f, %d trades, PnL ₹%.0f", sharpe, n_trades, total_pnl)
                success += 1
            else:
                log.error("  FAILED — save returned empty run_id")
                failed += 1
        except Exception as e:
            log.error("  FAILED save: %s", e)
            failed += 1

    log.info("=" * 60)
    log.info("Backfill complete: %d success, %d failed, %d skipped (of %d total)",
             success, failed, skipped, len(candidates))


if __name__ == "__main__":
    main()
