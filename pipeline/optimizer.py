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


# ── Result-store save callback ────────────────────────────────────────────────

def make_optuna_save_callback(
    strategy,
    spot_df: pl.DataFrame,
    option_df: pl.DataFrame,
    vix_df: pl.DataFrame,
    lot_size: int,
    study_id: str,
):
    """Return an Optuna callback that saves each trial to the ResultStore.

    Attach to study.optimize(..., callbacks=[make_optuna_save_callback(...)])
    Do not alter study configuration — add callback only.
    """
    try:
        from pipeline.result_store import (
            get_store, build_run_record, build_result_record,
            build_feature_records, build_split_record, COST_MODEL_VERSION,
            BACKTEST_ENGINE,
        )
        from pipeline.run_all import backtest_strategy
    except ImportError:
        return None

    def _callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        # Skip failed/degenerate trials (no trades) and clearly negative Sharpes.
        # We only persist trials that have at least neutral performance — this prevents
        # filling the ResultStore with garbage and halves save-callback overhead.
        if trial.value is None or trial.value < 0.0:
            return
        try:
            params = trial.params
            trades_df, metrics = backtest_strategy(
                strategy, spot_df, option_df, vix_df, lot_size, params,
            )
            store = get_store()

            run_record = build_run_record(
                strategy_id=strategy.name,
                spot_df=spot_df,
                backtest_engine=BACKTEST_ENGINE,
                cost_model_version=COST_MODEL_VERSION,
                optuna_study_id=study_id,
                optuna_trial_number=trial.number,
                notes=f"optuna_trial_{trial.number}",
            )

            param_records = [
                {
                    "param_name": k,
                    "param_value": v,
                    "param_type": "entry",
                    "is_optimized": 1,
                }
                for k, v in params.items()
            ]

            feature_records = build_feature_records(strategy)
            total_days = spot_df["day_id"].n_unique() if "day_id" in spot_df.columns else 12
            result_record = build_result_record(metrics, split="full", fold_number=0,
                                                 total_trading_days=total_days)
            split_record = build_split_record(spot_df, fold_number=0, split_type="fixed")

            # Signal snapshot for entry_signal_values
            signal_arrays = None
            try:
                signals = strategy.compute(spot_df, option_df, vix_df, params)
                signal_arrays = {
                    "buy_ce": signals.buy_ce,
                    "buy_pe": signals.buy_pe,
                    "stop_points": signals.stop_points,
                    "target_points": signals.target_points,
                    "strike_offset": signals.strike_offset,
                }
            except Exception:
                pass

            store.save_run(
                run_record=run_record,
                parameters=param_records,
                features=feature_records,
                splits=[split_record],
                results=[result_record],
                trades=trades_df,
                lot_size=lot_size,
                signal_arrays=signal_arrays,
            )
        except Exception as exc:
            log.warning("Optuna save callback error (trial %d): %s", trial.number, exc)

    return _callback


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
    params: Optional[dict] = None,
) -> tuple[list[dict], bool]:
    """Leave-one-day-out cross-validation.

    Args:
        params: Parameter dict to use. If None, uses strategy defaults.

    Returns (day_results, passed_cv).
    Strategy must be profitable on >= MIN_PROFITABLE_DAYS of TOTAL_TRADING_DAYS.
    """
    day_ids = sorted(spot_df["day_id"].unique().to_list())
    day_results = []
    days_profitable = 0

    default_params = params or {tp.name: tp.default for tp in strategy.tunable_params()}

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
    n_trials: int | None = None,
    timeout_secs: int | None = None,
    save_trials: bool = True,
) -> dict:
    """Run Optuna optimization on the provided data.

    Args:
        n_trials:     Override OPTUNA_TRIALS (e.g. 50 for per-fold in nested CV).
        timeout_secs: Override OPTUNA_TIMEOUT_SECS (e.g. 120 for per-fold).
        save_trials:  If False, skip the ResultStore save callback (nested CV mode
                      — individual train-fold trials are not useful to persist).

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

    import uuid as _uuid
    study_id = str(_uuid.uuid4())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Attach result-store save callback (add only — do not alter study config)
    # Disabled in nested CV mode (save_trials=False) to avoid persisting
    # thousands of train-fold Optuna trials that have no out-of-sample value.
    callbacks = []
    if save_trials:
        save_cb = make_optuna_save_callback(
            strategy, spot_df, option_df, vix_df, lot_size, study_id,
        )
        if save_cb is not None:
            callbacks.append(save_cb)

    _n_trials = n_trials if n_trials is not None else OPTUNA_TRIALS
    _timeout = timeout_secs if timeout_secs is not None else OPTUNA_TIMEOUT_SECS

    start_time = time.time()

    try:
        study.optimize(
            objective,
            n_trials=_n_trials,
            timeout=_timeout,
            show_progress_bar=False,
            callbacks=callbacks,
        )
    except Exception as e:
        log.warning("Optuna error for %s: %s", strategy.name, e)

    elapsed = time.time() - start_time
    timeout_hit = elapsed >= _timeout - 5

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
