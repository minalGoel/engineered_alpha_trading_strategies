#!/usr/bin/env python3
"""Capital allocation analysis: 3 strategies × 4 spread scenarios with ₹50L capital."""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import importlib
import numpy as np
import polars as pl
import math
from pathlib import Path

from pipeline.config import NIFTY_LOT, NIFTY_SYMBOL, BANKNIFTY_SYMBOL
from pipeline.data_loader import load_all_data, add_atm_strike
from pipeline.option_utils import get_strike_step
from pipeline.state_machine import warmup_numba
from pipeline.run_all import backtest_strategy
from pipeline.cost_model import (
    compute_trade_costs, net_pnl_quick, print_cost_breakdown,
    BROKERAGE_PER_ORDER, STT_RATE, EXCHANGE_TXN_RATE, SEBI_FEE_RATE,
    STAMP_DUTY_RATE, IPFT_RATE, GST_RATE,
)

STRATEGIES = [
    "nifty_banknifty_spread_v1",
    "expiry_day_pattern_v1",
    "implied_vol_dislocation_v1",
]

TOTAL_CAPITAL = 50_00_000  # ₹50 lakh
CAPITAL_PER_STRATEGY = TOTAL_CAPITAL // 3  # ₹16,66,666

SPREAD_SCENARIOS = [
    ("Conservative", 1.50),
    ("Realistic", 1.00),
    ("Limit orders", 0.75),
    ("Aggressive", 0.50),
]


def load_strategy(name):
    mod = importlib.import_module(f"pipeline.strategies.{name}")
    return mod.Strategy()


def compute_sharpe(daily_returns, total_days):
    """Annualized Sharpe from daily return array, zero-filled to total_days."""
    if len(daily_returns) < total_days:
        daily_returns = np.concatenate([daily_returns, np.zeros(total_days - len(daily_returns))])
    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
    if std_r > 0:
        return float(mean_r / std_r * math.sqrt(252))
    return 0.0


def find_breakeven_spread(trades_df, lot_size, lots_per_trade, total_days):
    """Binary search for the spread/side where net PnL = 0."""
    lo, hi = 0.0, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        total_net = 0.0
        for row_idx in range(len(trades_df)):
            ep = trades_df["entry_premium"][row_idx]
            xp = trades_df["exit_premium"][row_idx]
            lots = lots_per_trade[row_idx]
            c = compute_trade_costs(ep, xp, lot_size, lots, spread_per_side=mid)
            total_net += c["net_pnl"]
        if total_net > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    print("=" * 110)
    print("COST MODEL VERIFICATION")
    print("=" * 110)

    print(f"\nSample 1: NIFTY ATM CE, entry ₹300, exit ₹303, 1 lot ({NIFTY_LOT} units)")
    print_cost_breakdown(300, 303, NIFTY_LOT, 1)

    print(f"\nSample 2: NIFTY ATM CE, entry ₹300, exit ₹303, 7 lots ({NIFTY_LOT * 7} units)")
    print_cost_breakdown(300, 303, NIFTY_LOT, 7)

    print("\n  Note: Brokerage stays ₹40 regardless of lots.")
    print("  Spread/STT/exchange/stamp scale with quantity.")

    # ──────────────────────────────────────────────────────────────────
    # Load data and run strategies
    # ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("CAPITAL ALLOCATION ANALYSIS — ₹50L across 3 strategies")
    print("=" * 110)

    print("\nLoading data...")
    all_data = load_all_data()
    for key, symbol in [("nifty_spot", NIFTY_SYMBOL), ("banknifty_spot", BANKNIFTY_SYMBOL)]:
        if all_data[key] is not None:
            step = get_strike_step(symbol)
            all_data[key] = add_atm_strike(all_data[key], step)

    warmup_numba()

    spot_df = all_data["nifty_spot"]
    option_df = all_data["nifty_options"]
    vix_df = all_data["vix"]
    lot_size = NIFTY_LOT
    total_days = all_data["trading_days"]

    all_dates = sorted(spot_df["session_date"].unique().to_list())
    session_dates_list = spot_df["session_date"].to_list()

    # Get trades for each strategy
    strategy_trades = {}
    for name in STRATEGIES:
        strategy = load_strategy(name)
        trades_df, _ = backtest_strategy(strategy, spot_df, option_df, vix_df, lot_size)
        if trades_df is not None and not trades_df.is_empty():
            strategy_trades[name] = trades_df
            print(f"  {name}: {len(trades_df)} trades")
        else:
            print(f"  {name}: 0 trades")

    print(f"\n  Capital per strategy: ₹{CAPITAL_PER_STRATEGY:,.0f}")
    print(f"  Lot size: {lot_size}")

    # ──────────────────────────────────────────────────────────────────
    # Run 4 spread scenarios
    # ──────────────────────────────────────────────────────────────────
    for scenario_name, spread_val in SPREAD_SCENARIOS:
        print("\n" + "=" * 110)
        print(f"SCENARIO: {scenario_name} — Spread = ₹{spread_val:.2f}/side")
        print("=" * 110)

        portfolio_daily_pnl = {str(d): 0.0 for d in all_dates}
        portfolio_total_gross = 0.0
        portfolio_total_net = 0.0
        portfolio_total_cost = 0.0
        portfolio_total_trades = 0
        portfolio_cost_breakdown = {
            "spread": 0, "stt": 0, "brokerage": 0, "exchange_txn": 0,
            "sebi_fee": 0, "stamp_duty": 0, "ipft": 0, "gst": 0,
        }

        strategy_results = []

        for name in STRATEGIES:
            if name not in strategy_trades:
                strategy_results.append(None)
                continue

            trades_df = strategy_trades[name]
            n_trades = len(trades_df)
            entry_p = trades_df["entry_premium"].to_numpy()
            exit_p = trades_df["exit_premium"].to_numpy()
            entry_bars = trades_df["entry_bar"].to_numpy()

            # Compute lots per trade based on actual entry premium
            lots_per_trade = []
            for ep in entry_p:
                margin_per_lot = ep * lot_size  # premium × lot_size
                max_lots = int(CAPITAL_PER_STRATEGY / margin_per_lot) if margin_per_lot > 0 else 0
                max_lots = max(max_lots, 1)  # at least 1 lot
                lots_per_trade.append(max_lots)

            lots_arr = np.array(lots_per_trade)

            # Compute costs per trade
            total_gross = 0.0
            total_net = 0.0
            total_cost = 0.0
            cb = {k: 0.0 for k in portfolio_cost_breakdown}
            net_winners = 0
            daily_pnl = {str(d): 0.0 for d in all_dates}

            for i in range(n_trades):
                ep, xp = entry_p[i], exit_p[i]
                lots = lots_per_trade[i]
                c = compute_trade_costs(ep, xp, lot_size, lots, spread_per_side=spread_val)

                total_gross += c["gross_pnl"]
                total_net += c["net_pnl"]
                total_cost += c["total_cost"]

                cb["spread"] += c["spread_cost"]
                cb["stt"] += c["stt"]
                cb["brokerage"] += c["brokerage"]
                cb["exchange_txn"] += c["exchange_txn"]
                cb["sebi_fee"] += c["sebi_fee"]
                cb["stamp_duty"] += c["stamp_duty"]
                cb["ipft"] += c["ipft"]
                cb["gst"] += c["gst"]

                if c["net_pnl"] > 0:
                    net_winners += 1

                date_key = str(session_dates_list[int(entry_bars[i])])
                daily_pnl[date_key] += c["net_pnl"]

            # Sharpe
            daily_arr = np.array([daily_pnl[str(d)] for d in all_dates])
            capital_proxy = CAPITAL_PER_STRATEGY
            daily_returns = daily_arr / capital_proxy
            net_sharpe = compute_sharpe(daily_returns, total_days)

            # Breakeven spread
            be_spread = find_breakeven_spread(trades_df, lot_size, lots_per_trade, total_days)

            # Accumulate into portfolio
            for d in all_dates:
                portfolio_daily_pnl[str(d)] += daily_pnl[str(d)]
            portfolio_total_gross += total_gross
            portfolio_total_net += total_net
            portfolio_total_cost += total_cost
            portfolio_total_trades += n_trades
            for k in portfolio_cost_breakdown:
                portfolio_cost_breakdown[k] += cb[k]

            strategy_results.append({
                "name": name,
                "n_trades": n_trades,
                "avg_lots": float(np.mean(lots_arr)),
                "total_gross": total_gross,
                "total_net": total_net,
                "total_cost": total_cost,
                "cb": cb,
                "net_sharpe": net_sharpe,
                "net_per_trade": total_net / n_trades if n_trades > 0 else 0,
                "win_rate_net": net_winners / n_trades if n_trades > 0 else 0,
                "be_spread": be_spread,
            })

        # ── Print per-strategy results ──
        for sr in strategy_results:
            if sr is None:
                continue
            print(f"\n  ┌─ {sr['name']}")
            print(f"  │  Trades: {sr['n_trades']}   Avg lots/trade: {sr['avg_lots']:.1f}")
            print(f"  │  Gross PnL:       ₹{sr['total_gross']:>+12,.2f}")
            print(f"  │  ── Cost breakdown ──")
            print(f"  │    Spread:         ₹{sr['cb']['spread']:>12,.2f}")
            print(f"  │    STT:            ₹{sr['cb']['stt']:>12,.2f}")
            print(f"  │    Brokerage:      ₹{sr['cb']['brokerage']:>12,.2f}")
            print(f"  │    Exchange txn:   ₹{sr['cb']['exchange_txn']:>12,.2f}")
            print(f"  │    SEBI fee:       ₹{sr['cb']['sebi_fee']:>12,.2f}")
            print(f"  │    Stamp duty:     ₹{sr['cb']['stamp_duty']:>12,.2f}")
            print(f"  │    IPFT:           ₹{sr['cb']['ipft']:>12,.2f}")
            print(f"  │    GST:            ₹{sr['cb']['gst']:>12,.2f}")
            print(f"  │  Total costs:      ₹{sr['total_cost']:>12,.2f}")
            print(f"  │  Net PnL:          ₹{sr['total_net']:>+12,.2f}")
            print(f"  │  Net PnL/trade:    ₹{sr['net_per_trade']:>+12,.2f}")
            print(f"  │  Win rate (net):   {sr['win_rate_net']:.1%}")
            print(f"  │  Net Sharpe:       {sr['net_sharpe']:>+.4f}")
            print(f"  │  Breakeven spread: ₹{sr['be_spread']:.2f}/side")
            print(f"  └─")

        # ── Portfolio combined ──
        port_daily_arr = np.array([portfolio_daily_pnl[str(d)] for d in all_dates])
        port_returns = port_daily_arr / TOTAL_CAPITAL
        port_sharpe = compute_sharpe(port_returns, total_days)

        cum_pnl = np.cumsum(port_daily_arr)
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        best_day_idx = int(np.argmax(port_daily_arr))
        worst_day_idx = int(np.argmin(port_daily_arr))

        print(f"\n  ╔══ PORTFOLIO COMBINED (spread={spread_val:.2f}/side)")
        print(f"  ║  Total trades:      {portfolio_total_trades}")
        print(f"  ║  Total gross PnL:   ₹{portfolio_total_gross:>+12,.2f}")
        print(f"  ║  ── Cost breakdown ──")
        for k, v in portfolio_cost_breakdown.items():
            label = k.replace("_", " ").title()
            print(f"  ║    {label:18s}₹{v:>12,.2f}")
        print(f"  ║  Total costs:       ₹{portfolio_total_cost:>12,.2f}")
        print(f"  ║  Total net PnL:     ₹{portfolio_total_net:>+12,.2f}")
        print(f"  ║  Portfolio Sharpe:   {port_sharpe:>+.4f}")
        print(f"  ║  Max daily drawdown: ₹{max_dd:>12,.2f}")
        print(f"  ║  Best day:  {all_dates[best_day_idx]} → ₹{port_daily_arr[best_day_idx]:>+,.2f}")
        print(f"  ║  Worst day: {all_dates[worst_day_idx]} → ₹{port_daily_arr[worst_day_idx]:>+,.2f}")
        print(f"  ║  Return on capital:  {portfolio_total_net/TOTAL_CAPITAL*100:>+.2f}% over 12 days")
        print(f"  ╚══")

        # Daily PnL table
        print(f"\n  Daily portfolio PnL:")
        print(f"  {'Date':>12}  {'PnL':>12}")
        print(f"  {'─'*12}  {'─'*12}")
        for i, d in enumerate(all_dates):
            print(f"  {str(d):>12}  ₹{port_daily_arr[i]:>+11,.2f}")

    print("\n" + "=" * 110)
    print("END")
    print("=" * 110)


if __name__ == "__main__":
    main()
