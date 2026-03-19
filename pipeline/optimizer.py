"""Optuna-based parameter optimization.

Optimizes ONLY numeric thresholds (not indicator periods).
Indicators are computed ONCE; each trial only changes condition masks.
"""
from __future__ import annotations
import optuna
import numpy as np
import polars as pl
import logging
import time
from typing import Optional
from joblib import Parallel, delayed

from pipeline.config import (
    OPTUNA_TRIALS, OPTUNA_TIMEOUT_SECS, MIN_TRADES_FULL,
    OVERFIT_SHARPE_RATIO, CV_FOLDS, MIN_CV_FOLDS_PROFITABLE,
)
from pipeline.strategy_parser import ParsedStrategy
from pipeline.backtester import backtest_single
from pipeline.metrics import compute_sharpe_from_trades, compute_metrics

log = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _estimate_trading_days(df: pl.DataFrame) -> int:
    """Estimate total trading days from the data."""
    if "day_id" in df.columns:
        return df["day_id"].n_unique()
    if "datetime" in df.columns:
        return df["datetime"].cast(pl.Date).n_unique()
    return 252


def run_default_backtest(
    strategy: ParsedStrategy,
    stock_data: dict[str, pl.DataFrame],
    n_jobs: int = 4,
) -> tuple[Optional[pl.DataFrame], int]:
    """Run backtest with DEFAULT parameters across all stocks.

    Returns (combined_trades_df, total_trading_days).
    """
    all_trades = []
    max_trading_days = 0

    def _run_one(symbol: str, df: pl.DataFrame):
        return backtest_single(df, strategy, symbol)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_run_one)(sym, df)
        for sym, df in stock_data.items()
    )

    for sym, result in zip(stock_data.keys(), results):
        if result is not None and not result.is_empty():
            all_trades.append(result)

    # Estimate trading days from any stock's data
    for df in stock_data.values():
        td = _estimate_trading_days(df)
        max_trading_days = max(max_trading_days, td)
        break

    if not all_trades:
        return None, max_trading_days

    combined = pl.concat(all_trades)
    return combined, max_trading_days


def run_cv_validation(
    strategy: ParsedStrategy,
    stock_data: dict[str, pl.DataFrame],
    n_jobs: int = 4,
) -> tuple[list[dict], bool]:
    """Run 5-fold cross-validation with DEFAULT parameters.

    Returns (fold_results, passed_cv).
    """
    fold_results = []
    folds_profitable = 0

    for fold_idx, (start, end) in enumerate(CV_FOLDS):
        fold_trades = []

        for sym, df in stock_data.items():
            # Filter to fold period
            fold_df = df.filter(
                (pl.col("datetime") >= pl.lit(start).str.to_datetime()) &
                (pl.col("datetime") <= pl.lit(end + " 23:59:59").str.to_datetime())
            )
            if fold_df.is_empty() or len(fold_df) < strategy.max_lookback:
                continue

            trades = backtest_single(fold_df, strategy, sym)
            if trades is not None and not trades.is_empty():
                fold_trades.append(trades)

        if fold_trades:
            combined = pl.concat(fold_trades)
            total_pnl = combined["pnl"].sum()
            n_trades = len(combined)
        else:
            total_pnl = 0.0
            n_trades = 0

        profitable = total_pnl > 0 and n_trades >= 5
        if profitable:
            folds_profitable += 1

        fold_results.append({
            "fold": fold_idx + 1,
            "period": f"{start} to {end}",
            "pnl": round(float(total_pnl), 2),
            "trades": n_trades,
            "profitable": profitable,
        })

    passed = folds_profitable >= MIN_CV_FOLDS_PROFITABLE
    return fold_results, passed


def run_optimization(
    strategy: ParsedStrategy,
    stock_data: dict[str, pl.DataFrame],
    default_sharpe: float,
    total_trading_days: int,
    n_jobs: int = 4,
    progress_callback=None,
) -> dict:
    """Run Optuna optimization on full training data.

    Returns optimization results dict.
    """
    tunable = strategy.tunable_params
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

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective: maximize Sharpe ratio."""
        # Sample parameters
        overrides = {}
        for name, (default, lo, hi) in tunable.items():
            overrides[name] = trial.suggest_float(name, lo, hi)

        # Run backtest across all stocks with these params
        all_trades = []
        for sym, df in stock_data.items():
            trades = backtest_single(df, strategy, sym, param_overrides=overrides,
                                     skip_indicators=True)
            if trades is not None and not trades.is_empty():
                all_trades.append(trades)

        if not all_trades:
            return -999.0

        combined = pl.concat(all_trades)
        n_trades = len(combined)

        if n_trades < MIN_TRADES_FULL:
            return -999.0

        sharpe = compute_sharpe_from_trades(combined, strategy.capital_per_trade, total_trading_days)

        if progress_callback:
            progress_callback(trial.number, sharpe)

        return sharpe

    # Create study — NO pruning per spec
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    start_time = time.time()
    timeout_hit = False

    try:
        study.optimize(
            objective,
            n_trials=OPTUNA_TRIALS,
            timeout=OPTUNA_TIMEOUT_SECS,
            show_progress_bar=False,
        )
    except Exception as e:
        log.warning("Optuna optimization error for %s: %s", strategy.name, e)

    elapsed = time.time() - start_time
    timeout_hit = elapsed >= OPTUNA_TIMEOUT_SECS - 5

    # Extract results
    # Guard against all-trials-fail: if best_value is the sentinel -999,
    # treat as if no valid trial was found and fall back to defaults.
    if (study.best_trial is not None
            and study.best_value is not None
            and study.best_value > -900):
        optimized_params = study.best_params
        optimized_sharpe = study.best_value
        best_trial_num = study.best_trial.number
    else:
        optimized_params = default_params
        optimized_sharpe = default_sharpe
        best_trial_num = 0

    improvement_ratio = (optimized_sharpe / default_sharpe
                         if default_sharpe > 0 and default_sharpe != 0
                         else 1.0)
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
