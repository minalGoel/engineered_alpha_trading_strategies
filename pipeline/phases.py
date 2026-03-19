"""Phase orchestration — runs the full pipeline for a single strategy.

Each function handles one phase and saves results incrementally.
The main entry point is `run_strategy_pipeline()`.
"""
from __future__ import annotations
import orjson
import shutil
import logging
import time
import signal
import polars as pl
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pipeline.config import (
    RESULTS_DIR, OUTPUTS_DIR, TRAIN_END, TEST_START,
    MIN_TRADES_FULL, MIN_PASSING_STOCKS, OPTUNA_TIMEOUT_SECS,
    OOS_MIN_PROFIT_FACTOR, OOS_MIN_SHARPE,
    OOS_SHARPE_DECAY_LIMIT, OOS_MIN_STOCKS_PROFITABLE_PCT,
)
from pipeline.strategy_parser import ParsedStrategy, parse_strategy
from pipeline.data_loader import (
    load_stock_data, load_vix_data, load_index_data,
    merge_vix_index, assign_day_id,
)
from pipeline.indicators import compute_all_indicators
from pipeline.backtester import backtest_single, build_signals_from_trades
from pipeline.optimizer import run_default_backtest, run_cv_validation, run_optimization
from pipeline.metrics import compute_metrics, compute_per_stock_metrics
from pipeline.stock_filter import filter_all_stocks
from pipeline.report import generate_report

log = logging.getLogger(__name__)


def _save_json(path: Path, data):
    """Save data as JSON using orjson."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def _save_parquet(path: Path, df: pl.DataFrame):
    """Save DataFrame as Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _save_text(path: Path, text: str):
    """Save text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_verdict(verdict: str, reason: str = "") -> dict:
    return {"verdict": verdict, "reason": reason}


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Strategy processing timed out")


def run_strategy_pipeline(
    raw_json: dict,
    available_symbols: list[str],
    strategy_idx: int = 0,
    total_strategies: int = 358,
    n_cores: int = 4,
    start_phase: int = 1,
) -> dict:
    """Run the full pipeline for a single strategy.

    Returns a summary dict with verdict and key metrics.
    """
    name = raw_json.get("name", "unknown")
    strat_dir = RESULTS_DIR / name
    strat_dir.mkdir(parents=True, exist_ok=True)

    # Save strategy copy
    _save_json(strat_dir / "strategy.json", raw_json)

    log_prefix = f"[{strategy_idx}/{total_strategies}] {name}"
    log.info("%s — starting pipeline", log_prefix)

    result = {
        "name": name,
        "verdict": "FAILED_ERROR",
        "family": raw_json.get("_dedup_metadata", {}).get("family", ""),
        "tags": raw_json.get("tags", []),
    }

    try:
        # Set timeout
        try:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(OPTUNA_TIMEOUT_SECS)
        except (ValueError, AttributeError):
            pass  # SIGALRM not available on Windows / threads

        # ── Phase 2: Parse strategy ─────────────────────────────────────
        log.info("%s — Phase 2 — parsing strategy", log_prefix)
        ps = parse_strategy(raw_json)

        if not ps.is_parseable:
            verdict = _make_verdict("FAILED_PARSE_ERROR",
                                    f"Errors: {ps.parse_errors}; Unparsed: {ps.unparsed_conditions}")
            _save_json(strat_dir / "verdict.json", verdict)
            result["verdict"] = "FAILED_PARSE_ERROR"
            _generate_fail_report(strat_dir, name, raw_json, ps, "FAILED_PARSE_ERROR", verdict["reason"])
            return result

        # ── Load data for this strategy's timeframe ─────────────────────
        log.info("%s — Phase 2 — loading data (timeframe=%s)", log_prefix, ps.timeframe)

        vix_df = load_vix_data(ps.timeframe, end_date=TRAIN_END)
        index_df = load_index_data(ps.timeframe, end_date=TRAIN_END) if ps.needs_index else None

        # Load all stocks for training period, compute indicators
        train_stock_data = {}
        for sym in available_symbols:
            df = load_stock_data(ps.timeframe, sym, end_date=TRAIN_END)
            if df is not None and not df.is_empty():
                df = merge_vix_index(df, vix_df, index_df)
                # Compute indicators once
                df, computed, failed = compute_all_indicators(
                    df, ps.indicator_defs,
                    needs_vwap=ps.needs_vwap,
                    needs_pdh_pdl=ps.needs_pdh_pdl,
                    needs_prev_close=ps.needs_prev_close,
                    needs_opening_range=ps.needs_opening_range,
                    needs_gap=ps.needs_gap,
                    needs_obv=ps.needs_obv,
                    needs_bar_count=ps.needs_bar_count,
                )
                if len(df) > ps.max_lookback:
                    train_stock_data[sym] = df

        if not train_stock_data:
            verdict = _make_verdict("FAILED_NO_TRADES", "No stock data available for this timeframe")
            _save_json(strat_dir / "verdict.json", verdict)
            result["verdict"] = "FAILED_NO_TRADES"
            _generate_fail_report(strat_dir, name, raw_json, ps, "FAILED_NO_TRADES", verdict["reason"])
            return result

        log.info("%s — Phase 2 — loaded %d stocks", log_prefix, len(train_stock_data))

        # ── Phase 3a: Default backtest + CV ─────────────────────────────
        log.info("%s — Phase 3a — default backtest on full training set", log_prefix)

        default_trades, trading_days = run_default_backtest(ps, train_stock_data, n_jobs=n_cores)

        if default_trades is None or len(default_trades) < MIN_TRADES_FULL:
            n_trades = len(default_trades) if default_trades is not None else 0
            verdict = _make_verdict("FAILED_NO_TRADES",
                                    f"Only {n_trades} trades on full training set (need {MIN_TRADES_FULL})")
            _save_json(strat_dir / "verdict.json", verdict)
            result["verdict"] = "FAILED_NO_TRADES"
            _generate_fail_report(strat_dir, name, raw_json, ps, "FAILED_NO_TRADES", verdict["reason"])
            return result

        default_metrics = compute_metrics(default_trades, ps.capital_per_trade, trading_days)
        default_sharpe = default_metrics["sharpe_annualized"]
        log.info("%s — Phase 3a — default: %d trades, Sharpe=%.3f",
                 log_prefix, len(default_trades), default_sharpe)

        # Cross-validation
        log.info("%s — Phase 3a — 5-fold CV with default params", log_prefix)
        cv_results, cv_passed = run_cv_validation(ps, train_stock_data, n_jobs=n_cores)

        if not cv_passed:
            profitable_folds = sum(1 for f in cv_results if f["profitable"])
            verdict = _make_verdict("FAILED_OVERFIT",
                                    f"Only {profitable_folds}/5 CV folds profitable with default params")
            _save_json(strat_dir / "verdict.json", verdict)
            result["verdict"] = "FAILED_OVERFIT"

            # Still save optimization.json with CV results
            opt_json = {
                "default_params": {k: v[0] for k, v in ps.tunable_params.items()},
                "optimized_params": {},
                "param_bounds": {},
                "default_sharpe": round(default_sharpe, 4),
                "optimized_sharpe": 0,
                "sharpe_improvement_ratio": 0,
                "overfit_flag": False,
                "cv_fold_results": cv_results,
                "cv_folds_profitable": profitable_folds,
                "optuna_trials_completed": 0,
                "optuna_best_trial": 0,
                "optuna_timeout_hit": False,
            }
            _save_json(strat_dir / "optimization.json", opt_json)
            _generate_fail_report(strat_dir, name, raw_json, ps, "FAILED_OVERFIT",
                                  verdict["reason"], optimization=opt_json, cv_folds=cv_results)
            return result

        # ── Phase 3b: Optuna optimization ───────────────────────────────
        log.info("%s — Phase 3b — Optuna optimization (100 trials)", log_prefix)

        trial_count = [0]
        def progress_cb(trial_num, sharpe):
            trial_count[0] = trial_num + 1
            if (trial_num + 1) % 10 == 0:
                log.info("%s — Phase 3 — trial %d/100 — best Sharpe %.3f",
                         log_prefix, trial_num + 1, sharpe)

        opt_result = run_optimization(
            ps, train_stock_data, default_sharpe, trading_days,
            n_jobs=n_cores, progress_callback=progress_cb,
        )
        opt_result["cv_fold_results"] = cv_results
        opt_result["cv_folds_profitable"] = sum(1 for f in cv_results if f["profitable"])
        _save_json(strat_dir / "optimization.json", opt_result)

        optimized_params = opt_result["optimized_params"]
        optimized_sharpe = opt_result["optimized_sharpe"]
        log.info("%s — Phase 3b — optimized Sharpe=%.3f (default=%.3f, ratio=%.2f)",
                 log_prefix, optimized_sharpe, default_sharpe,
                 opt_result["sharpe_improvement_ratio"])

        # ── Phase 4: Per-stock backtesting ──────────────────────────────
        log.info("%s — Phase 4 — per-stock backtesting", log_prefix)

        per_stock_dir = strat_dir / "training" / "per_stock"
        per_stock_dir.mkdir(parents=True, exist_ok=True)

        all_train_trades = []
        per_stock_metrics = {}

        for sym, df in train_stock_data.items():
            trades = backtest_single(df, ps, sym, param_overrides=optimized_params,
                                     skip_indicators=True)
            if trades is not None and not trades.is_empty():
                all_train_trades.append(trades)
                m = compute_per_stock_metrics(trades, sym, ps.capital_per_trade, trading_days)
                per_stock_metrics[sym] = m
                _save_json(per_stock_dir / f"{sym}.json", m)
            else:
                per_stock_metrics[sym] = {
                    "symbol": sym, "total_trades": 0, "win_rate": 0, "profit_factor": 0,
                    "sharpe": 0, "total_pnl": 0, "max_drawdown": 0, "avg_trade_pnl": 0,
                    "avg_winner": 0, "avg_loser": 0, "avg_holding_bars": 0,
                    "best_trade_pnl": 0, "worst_trade_pnl": 0,
                    "passed_filter": False, "filter_failures": ["no_trades"],
                }
                _save_json(per_stock_dir / f"{sym}.json", per_stock_metrics[sym])

        # ── Phase 5: Stock filtering ────────────────────────────────────
        log.info("%s — Phase 5 — stock filtering", log_prefix)

        stock_filter_json, passing_syms, failing_syms = filter_all_stocks(per_stock_metrics)
        _save_json(strat_dir / "stock_filter.json", stock_filter_json)

        if len(passing_syms) < MIN_PASSING_STOCKS:
            verdict = _make_verdict("FAILED_NARROW",
                                    f"Only {len(passing_syms)} passing stocks (need {MIN_PASSING_STOCKS})")
            _save_json(strat_dir / "verdict.json", verdict)
            result["verdict"] = "FAILED_NARROW"

            # Save training results even for failed strategies
            _save_training_results(strat_dir, all_train_trades, ps, per_stock_metrics,
                                   passing_syms, trading_days)
            _generate_fail_report(strat_dir, name, raw_json, ps, "FAILED_NARROW",
                                  verdict["reason"], optimization=opt_result,
                                  cv_folds=cv_results, stock_filter=stock_filter_json,
                                  per_stock=per_stock_metrics)
            return result

        log.info("%s — Phase 5 — %d/%d stocks pass", log_prefix,
                 len(passing_syms), len(per_stock_metrics))

        # Save training results (passing stocks only for aggregate)
        passing_trades = [t for t in all_train_trades
                          if t["symbol"][0] in set(passing_syms)] if all_train_trades else []
        train_combined = pl.concat(passing_trades) if passing_trades else pl.DataFrame()
        train_metrics = compute_metrics(train_combined, ps.capital_per_trade, trading_days)
        train_metrics["passing_stocks"] = len(passing_syms)
        train_metrics["total_stocks_tested"] = len(per_stock_metrics)

        _save_training_results(strat_dir, all_train_trades, ps, per_stock_metrics,
                               passing_syms, trading_days, train_metrics)

        # ── Phase 6: Out-of-sample validation ───────────────────────────
        log.info("%s — Phase 6 — out-of-sample validation (2025)", log_prefix)

        vix_test = load_vix_data(ps.timeframe, start_date=TEST_START)
        index_test = load_index_data(ps.timeframe, start_date=TEST_START) if ps.needs_index else None

        all_test_trades = []
        test_per_stock = {}

        for sym in passing_syms:
            df = load_stock_data(ps.timeframe, sym, start_date=TEST_START)
            if df is None or df.is_empty():
                continue
            df = merge_vix_index(df, vix_test, index_test)
            df, _, _ = compute_all_indicators(
                df, ps.indicator_defs,
                needs_vwap=ps.needs_vwap,
                needs_pdh_pdl=ps.needs_pdh_pdl,
                needs_prev_close=ps.needs_prev_close,
                needs_opening_range=ps.needs_opening_range,
                needs_gap=ps.needs_gap,
                needs_obv=ps.needs_obv,
                needs_bar_count=ps.needs_bar_count,
            )
            trades = backtest_single(df, ps, sym, param_overrides=optimized_params,
                                     skip_indicators=True)
            if trades is not None and not trades.is_empty():
                all_test_trades.append(trades)
                m = compute_per_stock_metrics(trades, sym, ps.capital_per_trade)
                test_per_stock[sym] = m

        test_combined = pl.concat(all_test_trades) if all_test_trades else pl.DataFrame()
        test_metrics = compute_metrics(test_combined, ps.capital_per_trade)

        # Save test results
        test_dir = strat_dir / "test"
        test_dir.mkdir(parents=True, exist_ok=True)
        if not test_combined.is_empty():
            _save_parquet(test_dir / "trades.parquet", test_combined)
            signals = build_signals_from_trades(test_combined, name)
            if not signals.is_empty():
                _save_parquet(test_dir / "signals.parquet", signals)
        _save_json(test_dir / "metrics.json", test_metrics)
        _save_json(test_dir / "monthly_pnl.json", test_metrics.get("monthly_pnl", {}))

        test_per_stock_dir = test_dir / "per_stock"
        test_per_stock_dir.mkdir(parents=True, exist_ok=True)
        for sym, m in test_per_stock.items():
            _save_json(test_per_stock_dir / f"{sym}.json", m)

        # ── Validate OOS ────────────────────────────────────────────────
        test_pf = test_metrics.get("profit_factor", 0)
        test_sharpe = test_metrics.get("sharpe_annualized", 0)
        train_sharpe_val = train_metrics.get("sharpe_annualized", 0)

        # Check OOS criteria
        oos_failures = []
        if test_pf < OOS_MIN_PROFIT_FACTOR:
            oos_failures.append(f"test PF={test_pf:.2f} < {OOS_MIN_PROFIT_FACTOR}")
        if test_sharpe < OOS_MIN_SHARPE:
            oos_failures.append(f"test Sharpe={test_sharpe:.4f} < {OOS_MIN_SHARPE}")
        if train_sharpe_val > 0 and test_sharpe < train_sharpe_val * OOS_SHARPE_DECAY_LIMIT:
            oos_failures.append(
                f"test Sharpe={test_sharpe:.4f} < 30% of train ({train_sharpe_val * OOS_SHARPE_DECAY_LIMIT:.4f})")

        # Check % of stocks still profitable
        if passing_syms:
            profitable_test_stocks = sum(
                1 for sym in passing_syms
                if test_per_stock.get(sym, {}).get("total_pnl", 0) > 0
            )
            profitable_pct = profitable_test_stocks / len(passing_syms)
            if profitable_pct < OOS_MIN_STOCKS_PROFITABLE_PCT:
                oos_failures.append(
                    f"profitable stocks {profitable_pct:.1%} < {OOS_MIN_STOCKS_PROFITABLE_PCT:.0%}")

        if oos_failures:
            verdict_str = "FAILED_OOS"
            reason = "; ".join(oos_failures)
        else:
            verdict_str = "VALIDATED"
            reason = (f"Test Sharpe={test_sharpe:.4f}, PF={test_pf:.2f}, "
                      f"Sharpe decay={1-test_sharpe/train_sharpe_val:.1%}" if train_sharpe_val > 0
                      else f"Test Sharpe={test_sharpe:.4f}, PF={test_pf:.2f}")

        verdict = _make_verdict(verdict_str, reason)
        _save_json(strat_dir / "verdict.json", verdict)
        result["verdict"] = verdict_str

        log.info("%s — %s: %s", log_prefix, verdict_str, reason)

        # Store metrics for leaderboard
        result["train_metrics"] = train_metrics
        result["test_metrics"] = test_metrics
        result["stock_filter"] = stock_filter_json
        result["optimization"] = opt_result

        # ── Phase 7: Report ─────────────────────────────────────────────
        log.info("%s — Phase 7 — generating report", log_prefix)

        report_md = generate_report(
            strategy_name=name,
            raw_json=raw_json,
            parsed_strategy=ps,
            optimization=opt_result,
            stock_filter=stock_filter_json,
            train_metrics=train_metrics,
            test_metrics=test_metrics if not test_combined.is_empty() else None,
            verdict=verdict_str,
            verdict_reason=reason,
            cv_fold_results=cv_results,
            per_stock_results=per_stock_metrics,
            test_trades_df=test_combined if not test_combined.is_empty() else None,
            assumptions=ps.parse_warnings,
        )
        _save_text(strat_dir / "report.md", report_md)

    except TimeoutError:
        log.warning("%s — TIMEOUT after %d seconds", log_prefix, OPTUNA_TIMEOUT_SECS)
        verdict = _make_verdict("FAILED_TIMEOUT", f"Exceeded {OPTUNA_TIMEOUT_SECS}s safety timeout")
        _save_json(strat_dir / "verdict.json", verdict)
        result["verdict"] = "FAILED_TIMEOUT"

    except Exception as e:
        log.error("%s — UNCAUGHT ERROR: %s", log_prefix, e, exc_info=True)
        verdict = _make_verdict("FAILED_ERROR", str(e)[:500])
        _save_json(strat_dir / "verdict.json", verdict)
        result["verdict"] = "FAILED_ERROR"

    finally:
        # Cancel alarm
        try:
            signal.alarm(0)
        except (ValueError, AttributeError):
            pass

    log.info("%s — complete — verdict=%s", log_prefix, result["verdict"])
    return result


def _save_training_results(
    strat_dir: Path,
    all_trades: list,
    ps: ParsedStrategy,
    per_stock_metrics: dict,
    passing_syms: list[str],
    trading_days: int,
    train_metrics: Optional[dict] = None,
):
    """Save training phase results."""
    train_dir = strat_dir / "training"
    train_dir.mkdir(parents=True, exist_ok=True)

    if all_trades:
        combined = pl.concat(all_trades)
        _save_parquet(train_dir / "trades.parquet", combined)
        signals = build_signals_from_trades(combined, ps.name)
        if not signals.is_empty():
            _save_parquet(train_dir / "signals.parquet", signals)

        if train_metrics is None:
            passing_trades = [t for t in all_trades
                              if t["symbol"][0] in set(passing_syms)]
            if passing_trades:
                train_combined = pl.concat(passing_trades)
                train_metrics = compute_metrics(train_combined, ps.capital_per_trade, trading_days)
            else:
                train_metrics = compute_metrics(pl.DataFrame(), ps.capital_per_trade, trading_days)

        _save_json(train_dir / "metrics.json", train_metrics)
        _save_json(train_dir / "monthly_pnl.json", train_metrics.get("monthly_pnl", {}))


def _generate_fail_report(
    strat_dir: Path,
    name: str,
    raw_json: dict,
    ps: ParsedStrategy,
    verdict: str,
    reason: str,
    optimization: Optional[dict] = None,
    cv_folds: Optional[list] = None,
    stock_filter: Optional[dict] = None,
    per_stock: Optional[dict] = None,
):
    """Generate a report even for failed strategies."""
    report_md = generate_report(
        strategy_name=name,
        raw_json=raw_json,
        parsed_strategy=ps,
        optimization=optimization,
        stock_filter=stock_filter,
        train_metrics=None,
        test_metrics=None,
        verdict=verdict,
        verdict_reason=reason,
        cv_fold_results=cv_folds,
        per_stock_results=per_stock,
        assumptions=ps.parse_warnings,
    )
    _save_text(strat_dir / "report.md", report_md)
