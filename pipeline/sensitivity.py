"""Sensitivity analysis — Chan's core data-snooping test.

For each strategy's parameter set, perturbs each parameter ±20% and re-runs
the backtest.  A robust strategy should degrade gracefully (no cliff-drops).
Strategies where any single ±20% perturbation causes >50% Sharpe degradation
are flagged as fragile.

Usage (standalone):
    python -m pipeline.sensitivity                      # all qualified strategies
    python -m pipeline.sensitivity --strategy tick_v1   # single strategy

Usage (from pipeline code):
    from pipeline.sensitivity import run_sensitivity
    result = run_sensitivity(strategy, spot_df, option_df, vix_df, lot_size, params)
"""
from __future__ import annotations

import logging
import math
import numpy as np
from typing import Optional

from pipeline.strategies.base import BaseStrategy

log = logging.getLogger(__name__)

# ── Tunables ────────────────────────────────────────────────────────────────
PERTURBATION_PCT = 0.20        # ±20% of parameter value
SHARPE_DEGRADATION_LIMIT = 0.50  # >50% Sharpe drop = fragile
MIN_TRADES_FOR_SENSITIVITY = 10  # need at least this many trades to evaluate


def run_sensitivity(
    strategy: BaseStrategy,
    spot_df,
    option_df,
    vix_df,
    lot_size: int,
    params: dict[str, float],
    perturbation_pct: float = PERTURBATION_PCT,
    capital: Optional[float] = None,
) -> dict:
    """Run ±perturbation_pct sensitivity analysis on each parameter.

    Args:
        strategy: Loaded strategy object.
        spot_df, option_df, vix_df: Market data DataFrames.
        lot_size: Contract lot size.
        params: Baseline parameter dict (optimized or default).
        perturbation_pct: Fraction to perturb each param (default 0.20).
        capital: Capital per entry (None = use config default).

    Returns:
        Dict with keys:
            baseline_sharpe: float
            baseline_trades: int
            param_results: {param_name: {+pct: {sharpe, trades, pct_change}, -pct: ...}}
            worst_degradation_pct: float (most negative pct_change across all perturbations)
            worst_param: str (which param caused worst degradation)
            fragile: bool (True if worst_degradation > SHARPE_DEGRADATION_LIMIT)
            verdict: "ROBUST" | "FRAGILE" | "INSUFFICIENT_TRADES"
    """
    from pipeline.run_all import backtest_strategy

    tp_list = strategy.tunable_params()
    if not tp_list:
        return {
            "baseline_sharpe": 0.0,
            "baseline_trades": 0,
            "param_results": {},
            "worst_degradation_pct": 0.0,
            "worst_param": None,
            "fragile": False,
            "verdict": "NO_TUNABLE_PARAMS",
        }

    # ── Baseline run ────────────────────────────────────────────────────────
    _, baseline_metrics = backtest_strategy(
        strategy, spot_df, option_df, vix_df, lot_size, params, capital=capital,
    )
    baseline_sharpe = baseline_metrics.get("sharpe_annualized", 0.0)
    baseline_trades = baseline_metrics.get("total_trades", 0)

    if baseline_trades < MIN_TRADES_FOR_SENSITIVITY:
        return {
            "baseline_sharpe": baseline_sharpe,
            "baseline_trades": baseline_trades,
            "param_results": {},
            "worst_degradation_pct": 0.0,
            "worst_param": None,
            "fragile": False,
            "verdict": "INSUFFICIENT_TRADES",
        }

    # ── Perturb each parameter ──────────────────────────────────────────────
    tp_bounds = {tp.name: (tp.low, tp.high) for tp in tp_list}
    param_results = {}
    # Cache: (pname, direction) -> (perturbed_params, trades_df, metrics)
    # Kept so the save hook reuses results instead of re-running backtests.
    _run_cache: dict[tuple, tuple] = {}
    worst_degradation = 0.0
    worst_param = None

    for pname in params:
        if pname not in tp_bounds:
            continue

        lo_bound, hi_bound = tp_bounds[pname]
        base_val = params[pname]
        results_for_param = {}

        for direction, sign in [(f"+{int(perturbation_pct*100)}%", 1), (f"-{int(perturbation_pct*100)}%", -1)]:
            perturbed_val = base_val * (1.0 + sign * perturbation_pct)

            # Clamp to bounds
            perturbed_val = max(lo_bound, min(hi_bound, perturbed_val))

            # Skip if clamping made it identical to baseline
            if abs(perturbed_val - base_val) < 1e-12:
                results_for_param[direction] = {
                    "value": round(perturbed_val, 8),
                    "clamped": True,
                    "sharpe": baseline_sharpe,
                    "trades": baseline_trades,
                    "pct_change": 0.0,
                }
                continue

            perturbed_params = dict(params)
            perturbed_params[pname] = perturbed_val

            try:
                perturbed_trades, perturbed_metrics = backtest_strategy(
                    strategy, spot_df, option_df, vix_df, lot_size,
                    perturbed_params, capital=capital,
                )
                p_sharpe = perturbed_metrics.get("sharpe_annualized", 0.0)
                p_trades = perturbed_metrics.get("total_trades", 0)
                # Cache for save hook — no re-running needed
                _run_cache[(pname, direction)] = (perturbed_params, perturbed_trades, perturbed_metrics)
            except Exception as e:
                log.warning("Sensitivity error for %s %s=%s: %s",
                            strategy.name, pname, perturbed_val, e)
                p_sharpe = 0.0
                p_trades = 0

            # Compute degradation
            if abs(baseline_sharpe) > 0.01:
                pct_change = (p_sharpe - baseline_sharpe) / abs(baseline_sharpe)
            else:
                pct_change = 0.0

            results_for_param[direction] = {
                "value": round(perturbed_val, 8),
                "clamped": False,
                "sharpe": round(p_sharpe, 4),
                "trades": p_trades,
                "pct_change": round(pct_change, 4),
            }

            if pct_change < worst_degradation:
                worst_degradation = pct_change
                worst_param = pname

        param_results[pname] = results_for_param

    fragile = abs(worst_degradation) > SHARPE_DEGRADATION_LIMIT

    result = {
        "baseline_sharpe": round(baseline_sharpe, 4),
        "baseline_trades": baseline_trades,
        "param_results": param_results,
        "worst_degradation_pct": round(worst_degradation, 4),
        "worst_param": worst_param,
        "fragile": fragile,
        "verdict": "FRAGILE" if fragile else "ROBUST",
    }

    # ── Save each perturbed run to the ResultStore ────────────────────────────
    # Pass the already-computed run cache so we never re-run backtests.
    _save_sensitivity_runs_to_store(
        strategy, spot_df, option_df, vix_df, lot_size,
        params, param_results, capital=capital, run_cache=_run_cache,
    )

    return result


def _save_sensitivity_runs_to_store(
    strategy,
    spot_df,
    option_df,
    vix_df,
    lot_size: int,
    baseline_params: dict,
    param_results: dict,
    capital=None,
    run_cache: dict | None = None,
):
    """Save each sensitivity perturbation run to the ResultStore.

    Called after run_sensitivity() computes results. Writes is_sensitivity_run=True
    runs to the same schema as regular backtest runs.

    run_cache: dict keyed (pname, direction) -> (perturbed_params, trades_df, metrics)
               from run_sensitivity(). When provided, backtests are NOT re-run — results
               are read directly from the cache, cutting sensitivity compute time in half.
    """
    try:
        from pipeline.result_store import (
            get_store, build_run_record, build_result_record,
            build_feature_records, build_split_record, COST_MODEL_VERSION,
            BACKTEST_ENGINE,
        )
        from pipeline.run_all import backtest_strategy
    except ImportError:
        return

    if capital is None:
        from pipeline.cost_model import CAPITAL_PER_ENTRY
        capital = CAPITAL_PER_ENTRY

    store = get_store()
    total_days = spot_df["day_id"].n_unique() if (
        spot_df is not None and "day_id" in spot_df.columns
    ) else 12
    _cache = run_cache or {}

    for pname, directions in param_results.items():
        for direction, info in directions.items():
            if info.get("clamped"):
                continue
            perturbed_value = info.get("value")
            if perturbed_value is None:
                continue

            # ── Use pre-computed result from cache if available ──────────────
            cache_key = (pname, direction)
            if cache_key in _cache:
                perturbed_params, trades_df, metrics = _cache[cache_key]
            else:
                # Fallback: re-run (only happens if called outside run_sensitivity)
                perturbed_params = dict(baseline_params)
                perturbed_params[pname] = perturbed_value
                try:
                    trades_df, metrics = backtest_strategy(
                        strategy, spot_df, option_df, vix_df, lot_size,
                        perturbed_params, capital=capital,
                    )
                except Exception as exc:
                    log.warning("Sensitivity save: backtest error for %s %s=%s: %s",
                                strategy.name, pname, perturbed_value, exc)
                    continue

            run_record = build_run_record(
                strategy_id=strategy.name,
                spot_df=spot_df,
                backtest_engine=BACKTEST_ENGINE,
                cost_model_version=COST_MODEL_VERSION,
                is_sensitivity_run=True,
                notes=f"sensitivity:{pname}:{direction}",
            )

            param_records = [
                {
                    "param_name": k,
                    "param_value": v,
                    "param_type": "entry",
                    "is_optimized": 0,
                }
                for k, v in perturbed_params.items()
            ]
            # Mark the perturbed parameter
            for pr in param_records:
                if pr["param_name"] == pname:
                    pr["is_optimized"] = 0  # not optimised, just perturbed

            feature_records = build_feature_records(strategy)
            result_record = build_result_record(
                metrics, split="full", fold_number=0, total_trading_days=total_days,
            )
            split_record = build_split_record(spot_df, fold_number=0, split_type="fixed")

            signal_arrays = None
            try:
                signals = strategy.compute(spot_df, option_df, vix_df, perturbed_params)
                signal_arrays = {
                    "buy_ce": signals.buy_ce,
                    "buy_pe": signals.buy_pe,
                    "stop_points": signals.stop_points,
                    "target_points": signals.target_points,
                }
            except Exception:
                pass

            try:
                store.save_run(
                    run_record=run_record,
                    parameters=param_records,
                    features=feature_records,
                    splits=[split_record],
                    results=[result_record],
                    trades=trades_df,
                    lot_size=lot_size,
                    capital=capital,
                    signal_arrays=signal_arrays,
                )
            except Exception as exc:
                log.warning("Sensitivity save error for %s %s: %s",
                            strategy.name, pname, exc)


def format_sensitivity_report(name: str, result: dict) -> str:
    """Format a single strategy's sensitivity result as readable text."""
    lines = [f"\n{'─'*70}", f"  {name}  —  {result['verdict']}"]
    lines.append(f"  Baseline: Sharpe={result['baseline_sharpe']:.4f}, "
                 f"Trades={result['baseline_trades']}")

    if result["verdict"] in ("NO_TUNABLE_PARAMS", "INSUFFICIENT_TRADES"):
        lines.append(f"  Skipped: {result['verdict']}")
        return "\n".join(lines)

    for pname, directions in result["param_results"].items():
        lines.append(f"  ├─ {pname}")
        for direction, info in directions.items():
            tag = ""
            if info.get("clamped"):
                tag = " [clamped]"
            elif abs(info["pct_change"]) > SHARPE_DEGRADATION_LIMIT:
                tag = " *** CLIFF ***"
            lines.append(f"  │    {direction}: val={info['value']:.6f}  "
                         f"Sharpe={info['sharpe']:+.4f}  "
                         f"Δ={info['pct_change']:+.1%}  "
                         f"trades={info['trades']}{tag}")

    if result["fragile"]:
        lines.append(f"  ╰─ FRAGILE: worst param={result['worst_param']}, "
                     f"degradation={result['worst_degradation_pct']:+.1%}")
    else:
        lines.append(f"  ╰─ ROBUST: worst degradation={result['worst_degradation_pct']:+.1%}")

    return "\n".join(lines)


# ── CLI entry point ─────────────────────────────────────────────────────────
def main():
    import argparse
    import sys
    import os
    import time as _time
    import orjson
    from pathlib import Path

    os.environ.setdefault("NUMBA_NUM_THREADS", "1")

    from pipeline.config import (
        NIFTY_LOT, BANKNIFTY_LOT, NIFTY_SYMBOL, BANKNIFTY_SYMBOL,
        OUTPUTS_DIR,
    )
    from pipeline.data_loader import load_all_data, add_atm_strike
    from pipeline.option_utils import get_strike_step
    from pipeline.state_machine import warmup_numba
    from pipeline.run_all import _load_strategy_class, _get_qualified_strategies, _save_json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    parser = argparse.ArgumentParser(description="Sensitivity analysis (Chan's ±20% test)")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Run only this strategy name")
    parser.add_argument("--perturbation", type=float, default=PERTURBATION_PCT,
                        help=f"Perturbation fraction (default {PERTURBATION_PCT})")
    args = parser.parse_args()

    log.info("Loading data...")
    all_data = load_all_data()
    for key, symbol in [("nifty_spot", NIFTY_SYMBOL), ("banknifty_spot", BANKNIFTY_SYMBOL)]:
        if all_data[key] is not None:
            step = get_strike_step(symbol)
            all_data[key] = add_atm_strike(all_data[key], step)

    warmup_numba()

    qualified = _get_qualified_strategies()
    if args.strategy:
        qualified = [s for s in qualified if s["name"] == args.strategy]
        if not qualified:
            log.error("Strategy '%s' not found", args.strategy)
            sys.exit(1)

    log.info("Running sensitivity analysis on %d strategies (±%.0f%%)",
             len(qualified), args.perturbation * 100)

    all_results = {}
    n_robust = 0
    n_fragile = 0
    n_skipped = 0
    start = _time.time()

    for idx, q in enumerate(qualified, 1):
        name = q["name"]
        strategy = _load_strategy_class(name)
        if strategy is None:
            log.warning("[%d/%d] Cannot load %s", idx, len(qualified), name)
            continue

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
            spot_df = all_data["nifty_spot"]
            option_df = all_data["nifty_options"]
            lot_size = NIFTY_LOT

        vix_df = all_data["vix"]

        if spot_df is None or option_df is None:
            continue

        # Use default params as baseline (pre-optimization)
        default_params = {tp.name: tp.default for tp in strategy.tunable_params()}

        log.info("[%d/%d] %s (%d params)", idx, len(qualified), name, len(default_params))

        result = run_sensitivity(
            strategy, spot_df, option_df, vix_df, lot_size,
            default_params, perturbation_pct=args.perturbation,
        )

        all_results[name] = result
        print(format_sensitivity_report(name, result))

        verdict = result["verdict"]
        if verdict == "ROBUST":
            n_robust += 1
        elif verdict == "FRAGILE":
            n_fragile += 1
        else:
            n_skipped += 1

    elapsed = _time.time() - start

    # ── Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SENSITIVITY ANALYSIS COMPLETE — {elapsed:.1f}s")
    print(f"  Robust:  {n_robust}")
    print(f"  Fragile: {n_fragile}")
    print(f"  Skipped: {n_skipped}")
    print(f"{'='*70}")

    # Save results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _save_json(OUTPUTS_DIR / "sensitivity_results.json", {
        "perturbation_pct": args.perturbation,
        "degradation_limit": SHARPE_DEGRADATION_LIMIT,
        "summary": {
            "total": len(all_results),
            "robust": n_robust,
            "fragile": n_fragile,
            "skipped": n_skipped,
        },
        "strategies": all_results,
    })
    log.info("Results saved to %s", OUTPUTS_DIR / "sensitivity_results.json")


if __name__ == "__main__":
    main()
