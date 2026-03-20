"""Optuna-based parameter optimization for option strategies.

Optimizes ONLY numeric thresholds (not indicator periods).
Indicators are computed ONCE; each trial only changes condition masks.
Uses leave-one-day-out cross-validation for 12-day dataset.
"""
from __future__ import annotations
import optuna
import numpy as np
import polars as pl
import logging
import time
from typing import Optional

from pipeline.config import (
    OPTUNA_TRIALS, OPTUNA_TIMEOUT_SECS, MIN_TRADES_FULL,
    OVERFIT_SHARPE_RATIO, TOTAL_TRADING_DAYS, MIN_PROFITABLE_DAYS,
)
from pipeline.strategies.base import BaseStrategy
from pipeline.metrics import compute_sharpe_from_trades

log = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _estimate_trading_days(spot_df: pl.DataFrame) -> int:
    """Estimate total trading days from the data."""
    if "day_id" in spot_df.columns:
        return spot_df["day_id"].n_unique()
    if "datetime" in spot_df.columns:
        return spot_df["datetime"].cast(pl.Date).n_unique()
    return TOTAL_TRADING_DAYS


def run_leave_one_day_out_cv(
    strategy: BaseStrategy,
    spot_df: pl.DataFrame,
    option_df: pl.DataFrame,
    vix_df: pl.DataFrame,
    lot_size: int,
) -> tuple[list[dict], bool]:
    """Leave-one-day-out cross-validation with DEFAULT parameters.

    Returns (day_results, passed_cv).
    Strategy must be profitable on >= MIN_PROFITABLE_DAYS of TOTAL_TRADING_DAYS.
    """
    day_ids = sorted(spot_df["day_id"].unique().to_list())
    day_results = []
    days_profitable = 0

    default_params = {tp.name: tp.default for tp in strategy.tunable_params()}

    # Import backtest function for actual PnL computation
    from pipeline.run_all import backtest_strategy

    for test_day_id in day_ids:
        # Test on one day
        test_spot = spot_df.filter(pl.col("day_id") == test_day_id)

        if test_spot.is_empty():
            continue

        # Get test date for reporting
        test_date = ""
        if "session_date" in test_spot.columns:
            test_date = str(test_spot["session_date"].head(1).to_list()[0])

        try:
            # Run full backtest on test day
            trades_df, metrics = backtest_strategy(
                strategy, test_spot, option_df, vix_df, lot_size, default_params,
            )

            n_trades = metrics.get("total_trades", 0)
            day_pnl = metrics.get("total_pnl", 0.0)

            day_results.append({
                "day": test_day_id,
                "date": test_date,
                "pnl": day_pnl,
                "trades": n_trades,
                "profitable": day_pnl > 0,
            })

        except Exception as e:
            log.warning("CV error on day %d for %s: %s", test_day_id, strategy.name, e)
            day_results.append({
                "day": test_day_id,
                "date": test_date,
                "pnl": 0.0,
                "trades": 0,
                "profitable": False,
            })

    days_profitable = sum(1 for d in day_results if d.get("profitable", False))
    passed = days_profitable >= MIN_PROFITABLE_DAYS

    return day_results, passed


def run_optimization(
    strategy: BaseStrategy,
    spot_df: pl.DataFrame,
    option_df: pl.DataFrame,
    vix_df: pl.DataFrame,
    default_sharpe: float,
    total_trading_days: int,
    lot_size: int,
    progress_callback=None,
) -> dict:
    """Run Optuna optimization on full training data.

    Returns optimization results dict.
    """
    tp_list = strategy.tunable_params()
    tunable = {tp.name: (tp.default, tp.low, tp.high) for tp in tp_list}

    if not tunable:
        log.info("No tunable parameters for %s, using defaults", strategy.name)
        return {
            "default_params": {},
            "optimized_params": {},
            "param_bounds": {},
            "default_sharpe": round(default_sharpe, 4),
            "optimized_sharpe": round(default_sharpe, 4),
            "sharpe_improvement_ratio": 1.0,
            "overfit_flag": False,
            "optuna_trials_completed": 0,
            "optuna_best_trial": 0,
            "optuna_timeout_hit": False,
        }

    default_params = {k: v[0] for k, v in tunable.items()}
    param_bounds = {k: [v[1], v[2]] for k, v in tunable.items()}

    from pipeline.run_all import backtest_strategy

    def objective(trial: optuna.Trial) -> float:
        overrides = {}
        for name, (default, lo, hi) in tunable.items():
            overrides[name] = trial.suggest_float(name, lo, hi)

        try:
            trades_df, metrics = backtest_strategy(
                strategy, spot_df, option_df, vix_df, lot_size, overrides,
            )
            n_trades = metrics.get("total_trades", 0)
            if n_trades < MIN_TRADES_FULL:
                return -999.0
            sharpe = metrics.get("sharpe_annualized", -999.0)
            return sharpe
        except Exception:
            return -999.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    start_time = time.time()

    try:
        study.optimize(
            objective,
            n_trials=OPTUNA_TRIALS,
            timeout=OPTUNA_TIMEOUT_SECS,
            show_progress_bar=False,
        )
    except Exception as e:
        log.warning("Optuna error for %s: %s", strategy.name, e)

    elapsed = time.time() - start_time
    timeout_hit = elapsed >= OPTUNA_TIMEOUT_SECS - 5

    try:
        best_trial = study.best_trial
        best_value = study.best_value
        if best_trial is not None and best_value is not None and best_value > -900:
            optimized_params = study.best_params
            optimized_sharpe = best_value
            best_trial_num = best_trial.number
        else:
            optimized_params = default_params
            optimized_sharpe = default_sharpe
            best_trial_num = 0
    except ValueError:
        optimized_params = default_params
        optimized_sharpe = default_sharpe
        best_trial_num = 0

    if default_sharpe > 0.01:
        improvement_ratio = optimized_sharpe / default_sharpe
    else:
        improvement_ratio = 1.0
    overfit_flag = improvement_ratio > OVERFIT_SHARPE_RATIO

    return {
        "default_params": {k: round(v, 6) for k, v in default_params.items()},
        "optimized_params": {k: round(v, 6) for k, v in optimized_params.items()},
        "param_bounds": {k: [round(v[0], 6), round(v[1], 6)] for k, v in param_bounds.items()},
        "default_sharpe": round(default_sharpe, 4),
        "optimized_sharpe": round(optimized_sharpe, 4),
        "sharpe_improvement_ratio": round(improvement_ratio, 2),
        "overfit_flag": overfit_flag,
        "optuna_trials_completed": len(study.trials),
        "optuna_best_trial": best_trial_num,
        "optuna_timeout_hit": timeout_hit,
    }
