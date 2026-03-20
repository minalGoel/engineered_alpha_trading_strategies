#!/usr/bin/env python3
"""Step 3: Backtest all directional strategies + surviving vol/IV strategies.

Runs all strategies on full 12 days with default params.
Uses ₹50L capital, Upstox cost model, spread ₹1.00/side.
"""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import sys
import time
import importlib
import numpy as np
import polars as pl

from pipeline.config import (
    NIFTY_LOT, BANKNIFTY_LOT, NIFTY_SYMBOL, BANKNIFTY_SYMBOL,
)
from pipeline.data_loader import load_all_data, add_atm_strike
from pipeline.option_utils import get_strike_step
from pipeline.state_machine import warmup_numba
from pipeline.run_all import backtest_strategy
from pipeline.metrics import compute_metrics
from pipeline.cost_model import SPREAD_POINTS


# All new directional strategies
DIRECTIONAL_STRATEGIES = [
    "spot_momentum_v1",
    "volume_momentum_v1",
    "spot_acceleration_v1",
    "overextension_revert_v1",
    "vwap_deviation_v1",
    "banknifty_leadlag_v1",
    "vix_spot_predict_v1",
    "volume_spike_v1",
    "large_candle_v1",
    "opening_momentum_v1",
    "closing_momentum_v1",
    "expiry_afternoon_v1",
    "futures_basis_v1",
]

# Existing vol/IV strategies to keep
EXISTING_STRATEGIES = [
    "expiry_day_pattern_v1",
    "implied_vol_dislocation_v1",
    "implied_vs_realized_vol_v1",
    "intraday_vol_pattern_v1",
    "nifty_banknifty_spread_v1",
    "vol_surface_signal_v1",
]


def load_strategy(name: str):
    try:
        mod = importlib.import_module(f"pipeline.strategies.{name}")
        return mod.Strategy()
    except Exception as e:
        print(f"  ERROR loading {name}: {e}")
        return None


def main():
    print("=" * 100)
    print("STEP 3: BACKTEST ALL STRATEGIES (DEFAULT PARAMS)")
    print("=" * 100)

    # Load data
    print("\nLoading data...")
    t0 = time.time()
    all_data = load_all_data()

    for key, symbol in [("nifty_spot", NIFTY_SYMBOL), ("banknifty_spot", BANKNIFTY_SYMBOL)]:
        if all_data[key] is not None:
            step = get_strike_step(symbol)
            all_data[key] = add_atm_strike(all_data[key], step)

    print(f"Data loaded in {time.time() - t0:.1f}s")

    # Warmup Numba
    print("Warming up Numba...")
    warmup_numba()
    print("Numba ready\n")

    all_strategies = DIRECTIONAL_STRATEGIES + EXISTING_STRATEGIES
    results = []

    for idx, name in enumerate(all_strategies, 1):
        category = "DIRECTIONAL" if name in DIRECTIONAL_STRATEGIES else "VOL/IV"
        print(f"[{idx}/{len(all_strategies)}] {name} ({category})")

        strategy = load_strategy(name)
        if strategy is None:
            results.append({"name": name, "category": category, "error": True})
            continue

        # Determine data
        underlying = strategy.underlying.upper()
        if underlying in ("NIFTY", "NSE:NIFTY50-INDEX", "BOTH"):
            spot_df = all_data["nifty_spot"]
            option_df = all_data["nifty_options"]
            lot_size = NIFTY_LOT
        else:
            spot_df = all_data["banknifty_spot"]
            option_df = all_data["banknifty_options"]
            lot_size = BANKNIFTY_LOT

        vix_df = all_data["vix"]

        if spot_df is None or option_df is None:
            print(f"  No data for {underlying}")
            results.append({"name": name, "category": category, "error": True})
            continue

        try:
            t1 = time.time()
            trades_df, metrics = backtest_strategy(
                strategy, spot_df, option_df, vix_df, lot_size,
            )
            elapsed = time.time() - t1

            n_trades = metrics.get("total_trades", 0)
            net_pnl = metrics.get("total_pnl", 0.0)
            sharpe = metrics.get("sharpe_annualized", 0.0)
            win_rate = metrics.get("win_rate", 0.0)
            avg_pnl = metrics.get("avg_trade_pnl", 0.0)
            pf = metrics.get("profit_factor", 0.0)

            # Compute gross PnL (before costs) for comparison
            gross_pnl = 0.0
            if trades_df is not None and not trades_df.is_empty():
                gross_pnl = float(trades_df["pnl"].sum())

            # Gross Sharpe (approximate from gross PnL)
            gross_avg = gross_pnl / max(n_trades, 1)

            print(f"  Trades: {n_trades}, Gross: ₹{gross_pnl:,.0f}, Net: ₹{net_pnl:,.0f}, "
                  f"Sharpe: {sharpe:.2f}, WR: {win_rate:.1%}, PnL/trade: ₹{avg_pnl:.0f} ({elapsed:.1f}s)")

            results.append({
                "name": name,
                "category": category,
                "trades": n_trades,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "sharpe": sharpe,
                "win_rate": win_rate,
                "avg_pnl_per_trade": avg_pnl,
                "profit_factor": pf,
                "error": False,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({"name": name, "category": category, "error": True})

    # Print summary table
    print("\n" + "=" * 100)
    print("RESULTS TABLE — All Strategies (Default Params)")
    print("=" * 100)
    print(f"{'Strategy':<30} {'Cat':<5} {'Trades':>7} {'Gross PnL':>12} {'Net PnL':>12} "
          f"{'Sharpe':>8} {'WR':>6} {'PnL/Trade':>10}")
    print("─" * 100)

    for r in sorted(results, key=lambda x: x.get("net_pnl", -1e9), reverse=True):
        if r.get("error"):
            print(f"{r['name']:<30} {'ERR':<5}")
            continue
        flag = ""
        if r["gross_pnl"] > 0:
            flag = " ★"  # gross positive
        if r["net_pnl"] > 0:
            flag = " ★★"  # net positive
        print(f"{r['name']:<30} {r['category']:<5} {r['trades']:>7} "
              f"₹{r['gross_pnl']:>11,.0f} ₹{r['net_pnl']:>11,.0f} "
              f"{r['sharpe']:>8.2f} {r['win_rate']:>5.1%} "
              f"₹{r['avg_pnl_per_trade']:>9,.0f}{flag}")

    # Summary
    gross_positive = [r for r in results if not r.get("error") and r.get("gross_pnl", 0) > 0]
    net_positive = [r for r in results if not r.get("error") and r.get("net_pnl", 0) > 0]

    print(f"\n{'─' * 50}")
    print(f"Total strategies: {len(all_strategies)}")
    print(f"Gross-positive:   {len(gross_positive)} ★")
    print(f"Net-positive:     {len(net_positive)} ★★")
    print(f"\nGross-positive strategies (candidates for optimization):")
    for r in sorted(gross_positive, key=lambda x: x.get("gross_pnl", 0), reverse=True):
        print(f"  {r['name']}: gross ₹{r['gross_pnl']:,.0f}, net ₹{r['net_pnl']:,.0f}, Sharpe {r['sharpe']:.2f}")

    return results


if __name__ == "__main__":
    results = main()
