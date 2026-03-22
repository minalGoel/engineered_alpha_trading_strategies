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

# Raise file descriptor limit — macOS default is 256, insufficient for parallel strategy loading.
try:
    import resource as _resource
    _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    _target = min(max(_hard, 4096), 65536)
    _resource.setrlimit(_resource.RLIMIT_NOFILE, (_target, _hard))
except Exception:
    pass

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


import json
import multiprocessing as mp
import threading
import concurrent.futures
import numpy as np
import polars as pl


def build_option_premium_grid(
    spot_df: pl.DataFrame,
    option_df: pl.DataFrame,
    underlying: str,
) -> tuple:
    """Build 2D premium grids: ce_grid[n, n_strikes] and pe_grid[n, n_strikes].

    For each spot bar and each strike, stores the latest option close premium
    at or before that bar's timestamp (nearest expiry only).

    This allows the state machine to lock the entry strike and track the
    SAME contract's premium throughout the trade, even if ATM shifts.

    Returns (ce_grid, pe_grid, strikes_arr, atm_strike_idx, is_expiry).
    """
    n = len(spot_df)

    if option_df is None or option_df.is_empty():
        # Return minimal grid
        ce_grid = np.zeros((n, 1), dtype=np.float64)
        pe_grid = np.zeros((n, 1), dtype=np.float64)
        strikes_arr = np.array([0.0])
        atm_strike_idx = np.zeros(n, dtype=np.int32)
        is_expiry = np.zeros(n, dtype=np.bool_)
        return ce_grid, pe_grid, strikes_arr, atm_strike_idx, is_expiry

    # Build strike index
    all_strikes = sorted(option_df["strike"].unique().to_list())
    n_strikes = len(all_strikes)
    strike_to_idx = {}
    for i_s, s in enumerate(all_strikes):
        strike_to_idx[s] = i_s
    strikes_arr = np.array(all_strikes, dtype=np.float64)

    ce_grid = np.zeros((n, n_strikes), dtype=np.float64)
    pe_grid = np.zeros((n, n_strikes), dtype=np.float64)
    is_expiry = np.zeros(n, dtype=np.bool_)

    # ATM strike index per bar
    atm_raw = spot_df["atm_strike"].to_numpy()
    atm_strike_idx = np.zeros(n, dtype=np.int32)
    for i in range(n):
        atm = atm_raw[i]
        if atm in strike_to_idx:
            atm_strike_idx[i] = strike_to_idx[atm]
        else:
            atm_strike_idx[i] = int(np.argmin(np.abs(strikes_arr - atm)))

    # Mark expiry days
    expiry_dates = set(option_df["expiry"].unique().to_list())
    day_ids = spot_df["day_id"].to_numpy()
    session_dates = spot_df["session_date"].to_list()
    datetimes = spot_df["datetime"].to_list()

    for i in range(n):
        if session_dates[i] in expiry_dates:
            is_expiry[i] = True

    # Fill grid day-by-day
    for day_val in sorted(spot_df["day_id"].unique().to_list()):
        day_mask = day_ids == day_val
        day_indices = np.where(day_mask)[0]
        if len(day_indices) == 0:
            continue

        sd = session_dates[day_indices[0]]

        day_opts = option_df.filter(pl.col("session_date") == sd)
        if day_opts.is_empty():
            continue

        nearest_expiry = day_opts["expiry"].min()
        day_opts = day_opts.filter(pl.col("expiry") == nearest_expiry)

        # Get the set of ATM strikes used this day + neighbours (±2 strikes)
        day_atm_set = set(atm_raw[day_indices])
        strikes_needed = set()
        for atm_val in day_atm_set:
            if atm_val in strike_to_idx:
                idx = strike_to_idx[atm_val]
                for offset in range(-2, 3):
                    neighbor = idx + offset
                    if 0 <= neighbor < n_strikes:
                        strikes_needed.add(all_strikes[neighbor])

        # Process each (strike, option_type) combination
        for strike_val in strikes_needed:
            if strike_val not in strike_to_idx:
                continue
            s_idx = strike_to_idx[strike_val]

            for opt_type, grid in [("CE", ce_grid), ("PE", pe_grid)]:
                opts = day_opts.filter(
                    (pl.col("strike") == strike_val) &
                    (pl.col("option_type") == opt_type)
                ).sort("datetime")

                if opts.is_empty():
                    continue

                opt_times = opts["datetime"].to_list()
                opt_closes = opts["close"].to_numpy()

                # Two-pointer fill: for each spot bar, find last option close
                opt_idx = 0
                for si in day_indices:
                    bar_time = datetimes[si]
                    while opt_idx < len(opt_times) - 1 and opt_times[opt_idx + 1] <= bar_time:
                        opt_idx += 1
                    if opt_idx < len(opt_times) and opt_times[opt_idx] <= bar_time:
                        grid[si, s_idx] = opt_closes[opt_idx]

    return ce_grid, pe_grid, strikes_arr, atm_strike_idx, is_expiry


def backtest_strategy(strategy, spot_df, option_df, vix_df, lot_size, params=None, capital=None):
    """Run full backtest: compute signals → state machine → metrics.

    Returns (trades_df, metrics_dict) or (None, empty_metrics) on failure.
    """
    from pipeline.cost_model import CAPITAL_PER_ENTRY
    if capital is None:
        capital = CAPITAL_PER_ENTRY
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

    # Build option premium grid (2D: bars × strikes)
    ce_grid, pe_grid, strikes_arr, atm_strike_idx, is_expiry_arr = build_option_premium_grid(
        spot_df, option_df, strategy.underlying,
    )

    # Run state machine
    eod_mins = EOD_FLATTEN_H * 60 + EOD_FLATTEN_M
    exp_mins = EXPIRY_FLATTEN_H * 60 + EXPIRY_FLATTEN_M

    result = run_state_machine(
        spot_close=spot_df["close"].to_numpy().astype(np.float64),
        ce_grid=ce_grid,
        pe_grid=pe_grid,
        atm_strike_idx=atm_strike_idx,
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

    (entry_bars, exit_bars, sides, entry_prems, exit_prems,
     exit_reasons, pnls, entry_strike_indices, premium_missing, trade_count) = result

    if trade_count == 0:
        return None, _empty_metrics()

    # Build trades DataFrame
    dt_list = spot_df["datetime"].to_list()
    spot_close_arr = spot_df["close"].to_numpy()
    atm_arr = spot_df["atm_strike"].to_numpy()

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
        "entry_strike": [float(strikes_arr[int(si)]) for si in entry_strike_indices[:trade_count]],
        "entry_strike_idx": entry_strike_indices[:trade_count].astype(np.int32),
        "atm_at_entry": [float(atm_arr[int(b)]) for b in entry_bars[:trade_count]],
        "atm_at_exit": [float(atm_arr[int(b)]) for b in exit_bars[:trade_count]],
        "spot_at_entry": [float(spot_close_arr[int(b)]) for b in entry_bars[:trade_count]],
        "spot_at_exit": [float(spot_close_arr[int(b)]) for b in exit_bars[:trade_count]],
        "premium_missing_at_exit": premium_missing[:trade_count],
    })

    total_days = spot_df["day_id"].n_unique()
    metrics = compute_metrics(trades_df, lot_size=lot_size, total_trading_days=total_days, capital=capital)

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
    """Load qualified strategy list from retriage output.

    Reads strategy_retriage.json (291 entries, post-codegen) which has
    fields: name, source_file, underlying, tags, reason.
    Falls back to strategy_triage.json for legacy compatibility.
    """
    retriage_path = OUTPUTS_DIR / "strategy_retriage.json"
    legacy_path = OUTPUTS_DIR / "strategy_triage.json"

    if retriage_path.exists():
        triage_path = retriage_path
    elif legacy_path.exists():
        log.warning("strategy_retriage.json not found, falling back to strategy_triage.json")
        triage_path = legacy_path
    else:
        log.error("No strategy triage file found in outputs/")
        return []

    triage = orjson.loads(triage_path.read_bytes())
    return triage.get("qualified_strategies", [])


def _save_strategy_run(
    strategy,
    trades_df,
    metrics,
    spot_df,
    option_df,
    vix_df,
    lot_size: int,
    params: dict,
    is_sensitivity_run: bool = False,
    optuna_study_id: str = None,
    optuna_trial_number: int = None,
    notes: str = None,
    split: str = "full",
    fold_number: int = 0,
    split_type: str = "fixed",
    test_day_id: int = None,
    is_optimized: bool = False,
) -> str:
    """Save one backtest run to the ResultStore. Returns run_id."""
    try:
        from pipeline.result_store import (
            get_store, build_run_record, build_result_record,
            build_feature_records, build_split_record,
            COST_MODEL_VERSION, BACKTEST_ENGINE,
        )
        from pipeline.cost_model import CAPITAL_PER_ENTRY
    except ImportError as e:
        log.warning("ResultStore not available: %s", e)
        return ""

    store = get_store()
    total_days = spot_df["day_id"].n_unique() if "day_id" in spot_df.columns else 12

    run_record = build_run_record(
        strategy_id=strategy.name,
        spot_df=spot_df,
        backtest_engine=BACKTEST_ENGINE,
        cost_model_version=COST_MODEL_VERSION,
        optuna_study_id=optuna_study_id,
        optuna_trial_number=optuna_trial_number,
        is_sensitivity_run=is_sensitivity_run,
        notes=notes,
    )

    param_records = [
        {
            "param_name": k,
            "param_value": v,
            "param_type": "entry",
            "is_optimized": int(is_optimized),
        }
        for k, v in (params or {}).items()
    ]

    feature_records = build_feature_records(strategy)
    result_record = build_result_record(metrics, split=split, fold_number=fold_number,
                                         total_trading_days=total_days)
    split_record = build_split_record(spot_df, fold_number=fold_number,
                                       split_type=split_type, test_day_id=test_day_id)

    # Capture entry_signal_values via a separate compute() call (pure, no side effects)
    signal_arrays = None
    try:
        signals = strategy.compute(spot_df, option_df, vix_df, params or {})
        signal_arrays = {
            "buy_ce": signals.buy_ce,
            "buy_pe": signals.buy_pe,
            "stop_points": signals.stop_points,
            "target_points": signals.target_points,
            "strike_offset": signals.strike_offset,
        }
    except Exception:
        pass

    try:
        from pipeline.cost_model import CAPITAL_PER_ENTRY
        capital = CAPITAL_PER_ENTRY
        run_id = store.save_run(
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
        return run_id
    except Exception as e:
        log.warning("  ResultStore save failed: %s", e)
        return ""


def _resolve_data(strategy, all_data):
    """Resolve spot_df, option_df, lot_size for a strategy's underlying."""
    underlying = strategy.underlying.upper()
    if underlying in ("NIFTY", "NSE:NIFTY50-INDEX"):
        return all_data["nifty_spot"], all_data["nifty_options"], NIFTY_LOT
    elif underlying in ("BANKNIFTY", "NSE:NIFTYBANK-INDEX"):
        return all_data["banknifty_spot"], all_data["banknifty_options"], BANKNIFTY_LOT
    else:
        return all_data["nifty_spot"], all_data["nifty_options"], NIFTY_LOT


def _run_nested_cv_pipeline(strategy, spot_df, option_df, vix_df, lot_size) -> dict:
    """Nested LOO-CV pipeline: default backtest → {per-fold: optimize on train → test on held-out} → sensitivity.

    For each of 12 folds:
        1. Split: train = 11 days, test = 1 held-out day
        2. Optimize parameters on TRAIN data only (Optuna, 50 trials)
        3. Backtest on FULL data with fold-optimal params (indicators warm up from all days)
        4. Evaluate ONLY trades that entered on the held-out test day

    This eliminates the train-test leakage in the old pipeline where Optuna
    saw all 12 days and then CV "validated" on those same days.

    Returns a summary dict with all stage results.
    """
    from pipeline.config import (
        MIN_TRADES_FULL, MIN_PROFITABLE_DAYS,
        OPTUNA_TRIALS_PER_FOLD, OPTUNA_FOLD_TIMEOUT_SECS,
    )
    from pipeline.optimizer import run_optimization
    from pipeline.sensitivity import run_sensitivity
    from pipeline.metrics import compute_metrics as _compute_metrics, _empty_metrics

    name = strategy.name
    default_params = {tp.name: tp.default for tp in strategy.tunable_params()}
    day_ids = sorted(spot_df["day_id"].unique().to_list())
    total_days = len(day_ids)
    has_tunable = len(strategy.tunable_params()) > 0

    summary = {
        "name": name,
        "stages_completed": [],
        "default_sharpe": 0.0,
        "optimized_sharpe": 0.0,
        "cv_passed": None,
        "cv_profitable_days": 0,
        "cv_fold_details": [],
        "sensitivity_verdict": None,
        "n_optuna_trials": 0,
        "total_runs_saved": 0,
    }

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 1: Default params full backtest (baseline, no optimization)
    # ══════════════════════════════════════════════════════════════════════════
    trades_df, metrics = backtest_strategy(strategy, spot_df, option_df, vix_df, lot_size)
    n_trades = metrics.get("total_trades", 0)
    default_sharpe = metrics.get("sharpe_annualized", 0.0)
    summary["default_sharpe"] = round(default_sharpe, 4)

    if n_trades == 0:
        summary["verdict"] = "ZERO_TRADES"
        return summary

    run_id = _save_strategy_run(
        strategy, trades_df, metrics, spot_df, option_df, vix_df,
        lot_size, default_params, notes="default_params_full", split="full",
    )
    if run_id:
        summary["total_runs_saved"] += 1
    summary["stages_completed"].append("default_backtest")

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 2+3 merged: Nested LOO-CV
    #   For each held-out day:
    #     - Optimize on the OTHER 11 days (Optuna, 50 trials, no trial-level saves)
    #     - Test the fold-optimal params on the held-out day
    # ══════════════════════════════════════════════════════════════════════════
    cv_fold_results = []
    all_fold_params = []
    total_optuna_trials = 0

    for fold_num, test_day_id in enumerate(day_ids):
        # ── Split ────────────────────────────────────────────────────────────
        train_spot = spot_df.filter(pl.col("day_id") != test_day_id)
        train_days = train_spot["day_id"].n_unique()

        # ── Check train viability ────────────────────────────────────────────
        train_trades, train_metrics = backtest_strategy(
            strategy, train_spot, option_df, vix_df, lot_size, default_params,
        )
        train_n_trades = train_metrics.get("total_trades", 0)
        train_sharpe = train_metrics.get("sharpe_annualized", 0.0)

        # ── Optimize on TRAIN only ───────────────────────────────────────────
        fold_best_params = default_params
        fold_optuna_trials = 0

        if train_n_trades >= MIN_TRADES_FULL and has_tunable:
            try:
                fold_opt = run_optimization(
                    strategy, train_spot, option_df, vix_df,
                    train_sharpe, train_days, lot_size,
                    n_trials=OPTUNA_TRIALS_PER_FOLD,
                    timeout_secs=OPTUNA_FOLD_TIMEOUT_SECS,
                    save_trials=False,  # Don't persist train-fold Optuna trials
                )
                fold_best_params = fold_opt.get("optimized_params", default_params)
                fold_optuna_trials = fold_opt.get("optuna_trials_completed", 0)
                total_optuna_trials += fold_optuna_trials
            except Exception as e:
                log.warning("  [%s] Fold %d/%d optimization failed: %s",
                            name, fold_num, total_days, e)

        all_fold_params.append(fold_best_params)

        # ── Test on held-out day ─────────────────────────────────────────────
        # Run the FULL 12-day backtest with fold_best_params so indicators
        # are warm (causal — no look-ahead).  Then filter to test-day trades.
        test_trades_all, _ = backtest_strategy(
            strategy, spot_df, option_df, vix_df, lot_size, fold_best_params,
        )

        # Filter trades to only those ENTERING on the test day
        fold_trades = None
        if test_trades_all is not None and not test_trades_all.is_empty():
            test_spot_day = spot_df.filter(pl.col("day_id") == test_day_id)
            if not test_spot_day.is_empty() and "entry_time" in test_trades_all.columns:
                test_date = test_spot_day["session_date"].head(1).to_list()[0]
                fold_trades = test_trades_all.filter(
                    pl.col("entry_time").cast(pl.Date) == test_date
                )
                if fold_trades.is_empty():
                    fold_trades = None

        # Compute test-day metrics
        if fold_trades is not None:
            fold_metrics = _compute_metrics(fold_trades, lot_size=lot_size, total_trading_days=1)
        else:
            fold_metrics = _empty_metrics()

        day_pnl = fold_metrics.get("total_pnl", 0.0)
        fold_n_trades = fold_metrics.get("total_trades", 0)

        # Save each fold to ResultStore
        test_spot_fold = spot_df.filter(pl.col("day_id") == test_day_id)
        if fold_n_trades > 0 and fold_trades is not None:
            rid = _save_strategy_run(
                strategy, fold_trades, fold_metrics, test_spot_fold,
                option_df, vix_df, lot_size, fold_best_params,
                notes=f"nested_cv_fold_{fold_num}",
                split="test", fold_number=fold_num, split_type="nested_loo",
                test_day_id=test_day_id, is_optimized=has_tunable,
            )
            if rid:
                summary["total_runs_saved"] += 1

        cv_fold_results.append({
            "fold": fold_num,
            "day_id": int(test_day_id),
            "pnl": round(day_pnl, 2),
            "trades": fold_n_trades,
            "train_trades": train_n_trades,
            "train_sharpe": round(train_sharpe, 4),
            "optuna_trials": fold_optuna_trials,
            "profitable": day_pnl > 0,
        })

    # ── Aggregate CV results ─────────────────────────────────────────────────
    days_profitable = sum(1 for d in cv_fold_results if d["profitable"])
    cv_passed = days_profitable >= MIN_PROFITABLE_DAYS
    summary["cv_passed"] = cv_passed
    summary["cv_profitable_days"] = days_profitable
    summary["cv_fold_details"] = cv_fold_results
    summary["n_optuna_trials"] = total_optuna_trials
    summary["stages_completed"].append("nested_cross_validation")

    # Mean out-of-sample PnL across folds
    fold_pnls = [d["pnl"] for d in cv_fold_results if d["trades"] > 0]
    if fold_pnls:
        summary["optimized_sharpe"] = round(sum(fold_pnls) / max(len(fold_pnls), 1), 2)
    log.info("  [%s] Nested CV: %d/%d profitable, passed=%s, %d total Optuna trials",
             name, days_profitable, total_days, cv_passed, total_optuna_trials)

    # ══════════════════════════════════════════════════════════════════════════
    # Stage 4: Sensitivity analysis on consensus params
    # ══════════════════════════════════════════════════════════════════════════
    # Use element-wise median of fold-optimized params as the "consensus" set.
    if has_tunable and n_trades >= 10 and all_fold_params:
        try:
            consensus_params = _median_params(all_fold_params)
            sens_result = run_sensitivity(
                strategy, spot_df, option_df, vix_df, lot_size, params=consensus_params,
            )
            summary["sensitivity_verdict"] = sens_result.get("verdict", "UNKNOWN")
            n_perturbations = sum(
                1 for pname, dirs in sens_result.get("param_results", {}).items()
                for d, info in dirs.items() if not info.get("clamped")
            )
            summary["total_runs_saved"] += n_perturbations
            summary["stages_completed"].append("sensitivity")
        except Exception as e:
            log.warning("  [%s] Sensitivity failed: %s", name, e)

    summary["verdict"] = "NESTED_CV_PIPELINE_OK"
    return summary


def _median_params(param_list: list[dict]) -> dict:
    """Element-wise median across a list of parameter dicts.

    Used to pick "consensus" parameters from 12 CV folds for sensitivity analysis.
    """
    if not param_list:
        return {}
    keys = param_list[0].keys()
    result = {}
    for k in keys:
        vals = sorted(d[k] for d in param_list if k in d)
        if vals:
            mid = len(vals) // 2
            result[k] = vals[mid]
    return result


# ── Checkpoint helpers ──────────────────────────────────────────────────────

def _load_checkpoint(path: Path) -> dict:
    """Load {strategy_id: summary_dict} checkpoint. Returns {} if not found."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    """Atomically write checkpoint JSON."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(checkpoint, indent=2, default=str))
        os.replace(tmp, path)
    except Exception as e:
        log.warning("Checkpoint write failed: %s", e)


# ── ProcessPoolExecutor worker infrastructure ───────────────────────────────
#
# --full mode uses ProcessPoolExecutor (spawn) for TRUE CPU parallelism.
# Each spawned worker loads data from disk independently (no GIL sharing).
# --save mode keeps ThreadPoolExecutor (fast enough for single-pass).

_WORKER_DATA: dict | None = None   # Set by _worker_init in each spawned process


def _worker_init():
    """Called once per spawned worker process. Loads data and warms Numba.

    Takes ~3 seconds (Parquet decompression + JIT compilation).
    Data is ~50MB per process; 9 processes = ~450MB.
    On 16GB M4 with swap this is trivial.
    """
    global _WORKER_DATA
    import resource as _resource
    try:
        _soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
        _target = min(max(_hard, 4096), 65536)
        _resource.setrlimit(_resource.RLIMIT_NOFILE, (_target, _hard))
    except Exception:
        pass

    os.environ.setdefault("NUMBA_NUM_THREADS", "1")

    from pipeline.data_loader import load_all_data, add_atm_strike
    from pipeline.option_utils import get_strike_step
    from pipeline.state_machine import warmup_numba as _warmup

    all_data = load_all_data()
    for key, symbol in [("nifty_spot", NIFTY_SYMBOL), ("banknifty_spot", BANKNIFTY_SYMBOL)]:
        if all_data[key] is not None:
            step = get_strike_step(symbol)
            all_data[key] = add_atm_strike(all_data[key], step)
    _warmup()
    _WORKER_DATA = all_data


def _process_one_strategy(q: dict) -> dict:
    """Top-level function for ProcessPoolExecutor workers (must be picklable).

    Runs one strategy through the full nested CV pipeline.
    Accesses per-process _WORKER_DATA set by _worker_init().
    """
    name = q["name"]
    mode = q.get("mode", "full")

    strategy = _load_strategy_class(name)
    if strategy is None:
        return {"name": name, "verdict": "FAILED_LOAD_ERROR"}

    all_data = _WORKER_DATA
    if all_data is None:
        return {"name": name, "verdict": "FAILED_WORKER_INIT"}

    spot_df, option_df, lot_size = _resolve_data(strategy, all_data)
    vix_df = all_data["vix"]

    if spot_df is None or option_df is None:
        return {"name": name, "verdict": "FAILED_NO_DATA"}

    try:
        if mode == "full":
            t0 = time.time()
            result = _run_nested_cv_pipeline(strategy, spot_df, option_df, vix_df, lot_size)
            result["elapsed_seconds"] = round(time.time() - t0, 1)
            return result

        # Quick mode
        trades_df, metrics = backtest_strategy(strategy, spot_df, option_df, vix_df, lot_size)
        n_trades = metrics.get("total_trades", 0)

        result_entry = {
            "name": name,
            "verdict": "BACKTEST_OK" if n_trades > 0 else "ZERO_TRADES",
            "underlying": strategy.underlying.upper(),
            "metrics": metrics,
        }

        if mode == "save" and n_trades > 0:
            default_params = {tp.name: tp.default for tp in strategy.tunable_params()}
            _save_strategy_run(
                strategy=strategy, trades_df=trades_df, metrics=metrics,
                spot_df=spot_df, option_df=option_df, vix_df=vix_df,
                lot_size=lot_size, params=default_params,
                notes="quick_default_params",
            )

        return result_entry

    except Exception as e:
        return {"name": name, "verdict": "FAILED_COMPUTE_ERROR", "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="5-Second Option Trading Pipeline")
    parser.add_argument("--parallel-strategies", type=int,
                        default=DEFAULT_PARALLEL_STRATEGIES)
    parser.add_argument("--strategy", type=str, default=None,
                        help="Run only this strategy name")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and strategies, compute nothing")
    parser.add_argument("--save", action="store_true",
                        help="Quick mode: default-params backtest only, save to ResultStore")
    parser.add_argument("--full", action="store_true",
                        help="Nested LOO-CV: per-fold optimize on 11 days, test on 1 held-out day. "
                        "Saves all results to ResultStore. ~1 hour for 291 strategies × 9 processes.")
    parser.add_argument("--resume", action="store_true",
                        help="With --full: skip strategies already recorded in "
                        "outputs/full_pipeline_checkpoint.json. Safe to re-run after a crash.")
    parser.add_argument("--dashboard", action="store_true",
                        help="Start the strategy dashboard server")
    parser.add_argument("--audit", action="store_true",
                        help="Run ResultStore.audit() and print report")
    args = parser.parse_args()

    # ── Dashboard mode ──
    if args.dashboard:
        log.info("Starting dashboard server...")
        try:
            from pipeline.dashboard_server import start_server
            start_server()
        except ImportError:
            log.error("dashboard_server not found. Run: pip install fastapi uvicorn")
        return

    # ── Audit mode ──
    if args.audit:
        log.info("Running ResultStore audit...")
        try:
            from pipeline.result_store import get_store
            store = get_store()
            warnings = store.audit()
            print("\n" + "=" * 70)
            print("RESULT STORE AUDIT")
            print("=" * 70)
            for w in warnings:
                print(w)
            print("=" * 70 + "\n")
        except Exception as e:
            log.error("Audit failed: %s", e)
        return

    log.info("=" * 70)
    if args.full:
        mode_label = "NESTED LOO-CV (ProcessPool)"
    elif args.save:
        mode_label = "QUICK + SAVE (ThreadPool)"
    else:
        mode_label = "QUICK (no save)"
    log.info("5-Second Option Trading Pipeline — %s", mode_label)
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

    # ── Load all data (in parent — quick modes use it directly) ──
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

    # ── Warmup Numba (parent process — also caches to disk for child processes) ──
    log.info("Warming up Numba JIT compiler...")
    warmup_numba()
    log.info("Numba warmup complete")

    # ── Checkpoint: load prior progress for --full --resume ──
    CHECKPOINT_FILE = OUTPUTS_DIR / "full_pipeline_checkpoint.json"
    checkpoint: dict = {}
    if args.full and args.resume:
        checkpoint = _load_checkpoint(CHECKPOINT_FILE)
        n_already_done = len(checkpoint)
        if n_already_done:
            log.info("RESUME: %d strategies already completed in checkpoint, skipping them",
                     n_already_done)

    # ═══════════════════════════════════════════════════════════════════════════
    # Full pipeline: ProcessPoolExecutor with spawn
    #   - Each process loads data independently (no GIL contention)
    #   - True CPU parallelism on numpy + Numba
    #   - ~50MB data per process × 9 = 450MB (trivial on 16GB)
    # ═══════════════════════════════════════════════════════════════════════════
    n_workers = max(1, args.parallel_strategies)
    all_results = []

    if args.full:
        # Filter out checkpoint-completed strategies
        to_process = []
        for q in qualified:
            if args.resume and q["name"] in checkpoint:
                all_results.append(checkpoint[q["name"]])
            else:
                q["mode"] = "full"
                to_process.append(q)

        n_already = len(all_results)
        n_remaining = len(to_process)
        log.info("Processing %d strategies with %d worker PROCESSES "
                 "(%d already in checkpoint)",
                 n_remaining, n_workers, n_already)

        if n_remaining > 0:
            log.info("Spawning %d worker processes (each loads ~50MB data + Numba warmup)...",
                     n_workers)
            ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_worker_init,
            ) as pool:
                futures = {pool.submit(_process_one_strategy, q): q["name"]
                           for q in to_process}

                completed = n_already
                for future in concurrent.futures.as_completed(futures):
                    fname = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        log.error("Process error for %s: %s", fname, e)
                        result = {"name": fname, "verdict": "FAILED_PROCESS_ERROR",
                                  "error": str(e)}

                    all_results.append(result)
                    completed += 1

                    # Checkpoint from main process (only process with write access)
                    checkpoint[fname] = result
                    _save_checkpoint(CHECKPOINT_FILE, checkpoint)

                    elapsed_s = result.get("elapsed_seconds", 0)
                    log.info(
                        "[%d/%d] %s — %s | sr=%.2f cv=%s sens=%s saved=%d (%.0fs)",
                        completed, len(qualified), fname,
                        result.get("verdict", "?"),
                        result.get("default_sharpe", 0),
                        result.get("cv_passed", "?"),
                        result.get("sensitivity_verdict", "?"),
                        result.get("total_runs_saved", 0),
                        elapsed_s,
                    )

    else:
        # ═══════════════════════════════════════════════════════════════════════
        # Quick mode: ThreadPoolExecutor (fast enough, avoids spawn overhead)
        # ═══════════════════════════════════════════════════════════════════════
        mode = "save" if args.save else "quick"
        for q in qualified:
            q["mode"] = mode

        log.info("Processing %d strategies with %d worker thread(s)",
                 len(qualified), n_workers)

        results_lock = threading.Lock()
        counter = {"done": 0}

        def _thread_run(q: dict) -> dict:
            name = q["name"]
            strategy = _load_strategy_class(name)
            if strategy is None:
                return {"name": name, "verdict": "FAILED_LOAD_ERROR"}

            spot_df_t, option_df_t, lot_size_t = _resolve_data(strategy, all_data)
            vix_df_t = all_data["vix"]

            if spot_df_t is None or option_df_t is None:
                return {"name": name, "verdict": "FAILED_NO_DATA"}

            try:
                trades_df, metrics = backtest_strategy(
                    strategy, spot_df_t, option_df_t, vix_df_t, lot_size_t,
                )
                n_trades = metrics.get("total_trades", 0)

                with results_lock:
                    counter["done"] += 1
                    idx = counter["done"]
                log.info("[%d/%d] %s — trades=%d sharpe=%.4f pnl=INR%.0f",
                         idx, len(qualified), name, n_trades,
                         metrics.get("sharpe_annualized", 0),
                         metrics.get("total_pnl", 0))

                result_entry = {
                    "name": name,
                    "verdict": "BACKTEST_OK" if n_trades > 0 else "ZERO_TRADES",
                    "metrics": metrics,
                }

                if args.save and n_trades > 0:
                    default_params = {tp.name: tp.default for tp in strategy.tunable_params()}
                    _save_strategy_run(
                        strategy=strategy, trades_df=trades_df, metrics=metrics,
                        spot_df=spot_df_t, option_df=option_df_t, vix_df=vix_df_t,
                        lot_size=lot_size_t, params=default_params,
                        notes="quick_default_params",
                    )

                return result_entry

            except Exception as e:
                return {"name": name, "verdict": "FAILED_COMPUTE_ERROR", "error": str(e)}

        if n_workers == 1:
            for q in qualified:
                all_results.append(_thread_run(q))
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_thread_run, q): q["name"] for q in qualified}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        all_results.append(future.result())
                    except Exception as e:
                        fname = futures[future]
                        log.error("Thread error for %s: %s", fname, e)
                        all_results.append({"name": fname, "verdict": "FAILED_THREAD_ERROR"})

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - start_time
    log.info("=" * 70)
    log.info("Pipeline complete in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)

    verdict_counts = {}
    for r in all_results:
        v = r.get("verdict", "UNKNOWN")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    for verdict, count in sorted(verdict_counts.items()):
        log.info("  %s: %d", verdict, count)

    if args.full:
        total_saved = sum(r.get("total_runs_saved", 0) for r in all_results)
        n_cv_passed = sum(1 for r in all_results if r.get("cv_passed") is True)
        n_robust = sum(1 for r in all_results if r.get("sensitivity_verdict") == "ROBUST")
        total_trials = sum(r.get("n_optuna_trials", 0) for r in all_results)
        log.info("  Total runs saved to ResultStore: %d", total_saved)
        log.info("  Total Optuna trials (across all folds): %d", total_trials)
        log.info("  Nested CV passed: %d / %d", n_cv_passed, len(all_results))
        log.info("  Sensitivity ROBUST: %d / %d", n_robust, len(all_results))

    log.info("=" * 70)

    _save_json(OUTPUTS_DIR / "pipeline_log.json", {
        "mode": "full" if args.full else ("save" if args.save else "quick"),
        "total_strategies": len(qualified),
        "results": {r.get("name", "?"): r.get("verdict", "?") for r in all_results},
        "verdict_counts": verdict_counts,
        "elapsed_seconds": round(elapsed, 1),
    })

    if args.full:
        _save_json(OUTPUTS_DIR / "full_pipeline_results.json", {
            "elapsed_seconds": round(elapsed, 1),
            "validation_method": "nested_leave_one_day_out",
            "parallelism": f"ProcessPoolExecutor(spawn, workers={n_workers})",
            "strategies": all_results,
        })
        log.info("Detailed results saved to %s", OUTPUTS_DIR / "full_pipeline_results.json")


if __name__ == "__main__":
    main()
