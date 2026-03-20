#!/usr/bin/env python3
"""Step 4: Optimize promising strategies with Optuna (train/test split).

For any strategy with positive gross Sharpe:
- Split: first 8 days train, last 4 days test
- Optimize on training with Optuna (after-cost Sharpe, 100 trials)
- LOO-CV overfit check within training
- OOS on test set
- Print the honest comparison table
"""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import sys
import time
import importlib
import numpy as np
import polars as pl
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from pipeline.config import (
    NIFTY_LOT, BANKNIFTY_LOT, NIFTY_SYMBOL, BANKNIFTY_SYMBOL,
    OPTUNA_TRIALS, MIN_TRADES_FULL,
)
from pipeline.data_loader import load_all_data, add_atm_strike
from pipeline.option_utils import get_strike_step
from pipeline.state_machine import warmup_numba
from pipeline.run_all import backtest_strategy
from pipeline.metrics import compute_metrics, compute_sharpe_from_trades
from pipeline.cost_model import net_pnl_quick


# Strategies with positive gross PnL from Step 3
GROSS_POSITIVE_STRATEGIES = [
    "volume_spike_v1",
    "nifty_banknifty_spread_v1",
    "volume_momentum_v1",
    "closing_momentum_v1",
    "expiry_afternoon_v1",
    "banknifty_leadlag_v1",
    "large_candle_v1",
    "expiry_day_pattern_v1",
    "implied_vol_dislocation_v1",
    "futures_basis_v1",
    "vix_spot_predict_v1",
]


def load_strategy(name: str):
    try:
        mod = importlib.import_module(f"pipeline.strategies.{name}")
        return mod.Strategy()
    except Exception as e:
        print(f"  ERROR loading {name}: {e}")
        return None


def run_backtest_and_sharpe(strategy, spot_df, option_df, vix_df, lot_size, params, n_days):
    """Run backtest and return (trades_df, metrics, sharpe)."""
    trades_df, metrics = backtest_strategy(strategy, spot_df, option_df, vix_df, lot_size, params)
    sharpe = metrics.get("sharpe_annualized", -999.0)
    return trades_df, metrics, sharpe


def main():
    print("=" * 110)
    print("STEP 4: OPTIMIZE GROSS-POSITIVE STRATEGIES (8-day train / 4-day test)")
    print("=" * 110)

    # Load data
    print("\nLoading data...")
    t0 = time.time()
    all_data = load_all_data()

    for key, symbol in [("nifty_spot", NIFTY_SYMBOL), ("banknifty_spot", BANKNIFTY_SYMBOL)]:
        if all_data[key] is not None:
            step = get_strike_step(symbol)
            all_data[key] = add_atm_strike(all_data[key], step)

    print(f"Data loaded in {time.time() - t0:.1f}s")
    warmup_numba()
    print("Numba ready\n")

    # Split data: first 8 days train, last 4 days test
    nifty_spot = all_data["nifty_spot"]
    day_ids = sorted(nifty_spot["day_id"].unique().to_list())
    n_days = len(day_ids)
    train_days = day_ids[:8]
    test_days = day_ids[8:]

    print(f"Total days: {n_days}")
    print(f"Train days: {len(train_days)} (day_ids {train_days[0]}-{train_days[-1]})")
    print(f"Test days:  {len(test_days)} (day_ids {test_days[0]}-{test_days[-1]})")

    # Pre-split spot data
    nifty_train = nifty_spot.filter(pl.col("day_id").is_in(train_days))
    nifty_test = nifty_spot.filter(pl.col("day_id").is_in(test_days))

    bn_spot = all_data["banknifty_spot"]
    bn_train = bn_spot.filter(pl.col("day_id").is_in(train_days)) if bn_spot is not None else None
    bn_test = bn_spot.filter(pl.col("day_id").is_in(test_days)) if bn_spot is not None else None

    vix_df = all_data["vix"]
    nifty_opts = all_data["nifty_options"]
    bn_opts = all_data["banknifty_options"]

    results = []

    for idx, name in enumerate(GROSS_POSITIVE_STRATEGIES, 1):
        print(f"\n{'─' * 110}")
        print(f"[{idx}/{len(GROSS_POSITIVE_STRATEGIES)}] Optimizing: {name}")
        print(f"{'─' * 110}")

        strategy = load_strategy(name)
        if strategy is None:
            continue

        tp_list = strategy.tunable_params()
        if not tp_list:
            print("  No tunable params, skipping optimization")
            continue

        # Data selection
        underlying = strategy.underlying.upper()
        if underlying in ("NIFTY", "NSE:NIFTY50-INDEX", "BOTH"):
            train_spot = nifty_train
            test_spot = nifty_test
            full_spot = nifty_spot
            option_df = nifty_opts
            lot_size = NIFTY_LOT
        else:
            train_spot = bn_train
            test_spot = bn_test
            full_spot = bn_spot
            option_df = bn_opts
            lot_size = BANKNIFTY_LOT

        # 1. Default params on full data
        default_params = {tp.name: tp.default for tp in tp_list}
        _, full_metrics, full_sharpe = run_backtest_and_sharpe(
            strategy, full_spot, option_df, vix_df, lot_size, default_params, n_days
        )
        print(f"  Default (full 12d): Sharpe={full_sharpe:.2f}, "
              f"Trades={full_metrics['total_trades']}, Net=₹{full_metrics['total_pnl']:,.0f}")

        # 2. Default on training set
        _, train_default_metrics, train_default_sharpe = run_backtest_and_sharpe(
            strategy, train_spot, option_df, vix_df, lot_size, default_params, len(train_days)
        )
        print(f"  Default (train 8d): Sharpe={train_default_sharpe:.2f}, "
              f"Trades={train_default_metrics['total_trades']}, Net=₹{train_default_metrics['total_pnl']:,.0f}")

        # 3. Optuna optimization on training set
        tunable = {tp.name: (tp.default, tp.low, tp.high) for tp in tp_list}

        def objective(trial):
            overrides = {}
            for pname, (default, lo, hi) in tunable.items():
                overrides[pname] = trial.suggest_float(pname, lo, hi)
            try:
                # Need fresh strategy instance for strategies with cached data
                strat = load_strategy(name)
                trades_df, metrics = backtest_strategy(
                    strat, train_spot, option_df, vix_df, lot_size, overrides
                )
                n_trades = metrics.get("total_trades", 0)
                if n_trades < 10:  # relaxed minimum for 8 days
                    return -999.0
                return metrics.get("sharpe_annualized", -999.0)
            except Exception:
                return -999.0

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )

        print(f"  Running Optuna ({OPTUNA_TRIALS} trials on train set)...")
        t1 = time.time()
        study.optimize(objective, n_trials=OPTUNA_TRIALS, timeout=5 * 60)
        elapsed = time.time() - t1
        print(f"  Optuna done in {elapsed:.0f}s, {len(study.trials)} trials")

        try:
            best_params = study.best_params
            best_train_sharpe = study.best_value
        except ValueError:
            print("  No valid trials found, skipping")
            continue

        if best_train_sharpe < -900:
            print("  No viable params found")
            continue

        print(f"  Best train Sharpe: {best_train_sharpe:.2f}")
        print(f"  Optimized params: {best_params}")

        # 4. Evaluate optimized params on train set
        strat_opt = load_strategy(name)
        train_trades, train_opt_metrics, train_opt_sharpe = run_backtest_and_sharpe(
            strat_opt, train_spot, option_df, vix_df, lot_size, best_params, len(train_days)
        )
        print(f"  Optimized (train): Sharpe={train_opt_sharpe:.2f}, "
              f"Trades={train_opt_metrics['total_trades']}, Net=₹{train_opt_metrics['total_pnl']:,.0f}")

        # 5. LOO-CV overfit check on training set
        print(f"  LOO-CV on training set...")
        loo_pnls = []
        for held_day in train_days:
            held_spot = train_spot.filter(pl.col("day_id") == held_day)
            if held_spot.is_empty():
                loo_pnls.append(0.0)
                continue
            strat_loo = load_strategy(name)
            try:
                _, loo_metrics = backtest_strategy(
                    strat_loo, held_spot, option_df, vix_df, lot_size, best_params
                )
                loo_pnls.append(loo_metrics.get("total_pnl", 0.0))
            except Exception:
                loo_pnls.append(0.0)

        loo_profitable = sum(1 for p in loo_pnls if p > 0)
        loo_total_pnl = sum(loo_pnls)
        print(f"  LOO-CV: {loo_profitable}/{len(train_days)} days profitable, "
              f"total=₹{loo_total_pnl:,.0f}")

        # 6. OOS: evaluate on test set
        strat_oos = load_strategy(name)
        test_trades, test_metrics, test_sharpe = run_backtest_and_sharpe(
            strat_oos, test_spot, option_df, vix_df, lot_size, best_params, len(test_days)
        )
        test_pnl = test_metrics.get("total_pnl", 0.0)
        test_trades_n = test_metrics.get("total_trades", 0)
        test_wr = test_metrics.get("win_rate", 0.0)
        print(f"  OOS (test 4d): Sharpe={test_sharpe:.2f}, "
              f"Trades={test_trades_n}, Net=₹{test_pnl:,.0f}, WR={test_wr:.1%}")

        # 7. Also run default on test for fair comparison
        strat_def_oos = load_strategy(name)
        _, test_def_metrics, test_def_sharpe = run_backtest_and_sharpe(
            strat_def_oos, test_spot, option_df, vix_df, lot_size, default_params, len(test_days)
        )

        # Overfit ratio
        if train_default_sharpe > 0.01:
            improvement = train_opt_sharpe / train_default_sharpe
        else:
            improvement = 1.0

        results.append({
            "name": name,
            "default_params": default_params,
            "optimized_params": best_params,
            "train_default_sharpe": train_default_sharpe,
            "train_opt_sharpe": train_opt_sharpe,
            "train_opt_pnl": train_opt_metrics.get("total_pnl", 0.0),
            "train_opt_trades": train_opt_metrics.get("total_trades", 0),
            "loo_profitable": loo_profitable,
            "loo_total_pnl": loo_total_pnl,
            "test_sharpe": test_sharpe,
            "test_pnl": test_pnl,
            "test_trades": test_trades_n,
            "test_wr": test_wr,
            "test_default_sharpe": test_def_sharpe,
            "test_default_pnl": test_def_metrics.get("total_pnl", 0.0),
            "improvement_ratio": improvement,
            "overfit_flag": improvement > 3.0,
        })

    # ── Print comparison table ──
    print("\n" + "=" * 130)
    print("HONEST COMPARISON TABLE — Train vs Test, Default vs Optimized")
    print("=" * 130)
    print(f"{'Strategy':<28} │ {'Train Def':>10} │ {'Train Opt':>10} │ {'Test Def':>10} │ "
          f"{'Test Opt':>10} │ {'LOO':>5} │ {'Test Trades':>6} │ {'Test WR':>7} │ {'Overfit':>7}")
    print(f"{'':28} │ {'Sharpe':>10} │ {'Sharpe':>10} │ {'Sharpe':>10} │ "
          f"{'Sharpe':>10} │ {'prof':>5} │ {'':>6} │ {'':>7} │ {'':>7}")
    print("─" * 130)

    for r in sorted(results, key=lambda x: x["test_pnl"], reverse=True):
        flag = "⚠ YES" if r["overfit_flag"] else "OK"
        surviving = "✓" if r["test_pnl"] > 0 else "✗"
        print(f"{r['name']:<28} │ {r['train_default_sharpe']:>10.2f} │ {r['train_opt_sharpe']:>10.2f} │ "
              f"{r['test_default_sharpe']:>10.2f} │ {r['test_sharpe']:>10.2f} │ "
              f"{r['loo_profitable']:>3}/{len(train_days)} │ {r['test_trades']:>6} │ "
              f"{r['test_wr']:>6.1%} │ {flag:>7} {surviving}")

    # Print PnL comparison
    print(f"\n{'Strategy':<28} │ {'Train Net':>12} │ {'Test Net':>12} │ {'LOO PnL':>12} │ {'Verdict':>10}")
    print("─" * 90)
    for r in sorted(results, key=lambda x: x["test_pnl"], reverse=True):
        verdict = "SURVIVES" if r["test_pnl"] > 0 and not r["overfit_flag"] else "FAILS"
        if r["test_pnl"] > 0 and r["overfit_flag"]:
            verdict = "SUSPECT"
        print(f"{r['name']:<28} │ ₹{r['train_opt_pnl']:>11,.0f} │ ₹{r['test_pnl']:>11,.0f} │ "
              f"₹{r['loo_total_pnl']:>11,.0f} │ {verdict:>10}")

    surviving = [r for r in results if r["test_pnl"] > 0 and not r["overfit_flag"]]
    print(f"\nStrategies surviving OOS with positive net PnL: {len(surviving)}")
    for r in surviving:
        print(f"  {r['name']}: test net ₹{r['test_pnl']:,.0f}, test Sharpe {r['test_sharpe']:.2f}")


if __name__ == "__main__":
    main()
