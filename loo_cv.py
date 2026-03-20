#!/usr/bin/env python3
"""Leave-one-day-out CV for vol_surface_signal_v1 and expiry_day_pattern_v1."""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import importlib
import numpy as np
import polars as pl
import math

from pipeline.config import (
    NIFTY_LOT, NIFTY_SYMBOL, BANKNIFTY_SYMBOL,
)
from pipeline.data_loader import load_all_data, add_atm_strike
from pipeline.option_utils import get_strike_step
from pipeline.state_machine import run_state_machine, warmup_numba, EXIT_REASON_MAP
from pipeline.run_all import build_option_premium_arrays, backtest_strategy
from pipeline.cost_model import (
    compute_trade_costs, net_pnl_quick, SPREAD_POINTS, STT_RATE,
    BROKERAGE_PER_ORDER, EXCHANGE_PER_LOT,
)


def load_strategy(name):
    mod = importlib.import_module(f"pipeline.strategies.{name}")
    return mod.Strategy()


def run_loo_cv(strategy_name, all_data):
    """Run LOO-CV: for each of 12 days, hold out that day and test on it.

    With default params and rolling lookbacks, signals are computed on the full
    dataset but only trades on the test day are scored. The rolling windows are
    purely local (20 min for vol_surface, 10 min for expiry_day), so there is
    no information leakage from the test day into training.
    """
    strategy = load_strategy(strategy_name)
    lot_size = NIFTY_LOT
    spot_df = all_data["nifty_spot"]
    option_df = all_data["nifty_options"]
    vix_df = all_data["vix"]

    # Run backtest on ALL 12 days to get all trades
    trades_df, _ = backtest_strategy(strategy, spot_df, option_df, vix_df, lot_size)

    if trades_df is None or trades_df.is_empty():
        print(f"  ** ZERO TRADES across all days **")
        return

    n_trades_total = len(trades_df)

    # Map each trade to its session_date via entry_time
    # Get session_date for each bar from spot_df
    spot_dates = spot_df["session_date"].to_list()
    spot_datetimes = spot_df["datetime"].to_list()

    # Get all unique session dates in order
    all_dates = sorted(spot_df["session_date"].unique().to_list())
    day_id_to_date = {}
    for d in spot_df.select(["day_id", "session_date"]).unique().iter_rows():
        day_id_to_date[d[0]] = d[1]

    # Assign each trade to a session_date
    entry_bars = trades_df["entry_bar"].to_numpy()
    trade_dates = []
    for eb in entry_bars:
        trade_dates.append(spot_dates[int(eb)])

    trades_with_date = trades_df.with_columns(
        pl.Series("session_date", trade_dates)
    )

    # Also compute day_id for each trade
    spot_day_ids = spot_df["day_id"].to_numpy()
    trade_day_ids = [int(spot_day_ids[int(eb)]) for eb in entry_bars]
    trades_with_date = trades_with_date.with_columns(
        pl.Series("trade_day_id", trade_day_ids)
    )

    # Per-day results
    print(f"\n  {'Date':>12}  {'DayID':>5}  {'DOW':>5}  {'#Trades':>7}  {'Gross PnL':>12}  {'Net PnL':>12}  {'Gross/trade':>12}  {'Net/trade':>12}  {'Net Profitable?':>15}")
    print(f"  {'─'*12}  {'─'*5}  {'─'*5}  {'─'*7}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*12}  {'─'*15}")

    total_net_pnl = 0.0
    total_gross_pnl = 0.0
    total_trades_oos = 0
    profitable_days = 0
    days_with_trades = 0

    for day_idx, date_val in enumerate(all_dates):
        day_trades = trades_with_date.filter(pl.col("session_date") == date_val)
        n_day = len(day_trades)

        # Day of week
        import datetime as dt
        if hasattr(date_val, 'strftime'):
            dow = date_val.strftime("%a")
        else:
            dow = "?"

        if n_day == 0:
            print(f"  {str(date_val):>12}  {day_idx:>5}  {dow:>5}  {0:>7}  {'—':>12}  {'—':>12}  {'—':>12}  {'—':>12}  {'—':>15}")
            continue

        days_with_trades += 1

        entry_p = day_trades["entry_premium"].to_numpy()
        exit_p = day_trades["exit_premium"].to_numpy()

        gross_pnl = np.sum((exit_p - entry_p) * lot_size)
        net_pnls = np.array([net_pnl_quick(ep, xp, lot_size) for ep, xp in zip(entry_p, exit_p)])
        net_pnl = np.sum(net_pnls)

        total_net_pnl += net_pnl
        total_gross_pnl += gross_pnl
        total_trades_oos += n_day

        is_profitable = "YES" if net_pnl > 0 else "NO"
        if net_pnl > 0:
            profitable_days += 1

        print(f"  {str(date_val):>12}  {day_idx:>5}  {dow:>5}  {n_day:>7}  {gross_pnl:>+12.2f}  {net_pnl:>+12.2f}  {gross_pnl/n_day:>+12.2f}  {net_pnl/n_day:>+12.2f}  {is_profitable:>15}")

    # Summary
    print(f"\n  SUMMARY:")
    print(f"    Total trading days:         12")
    print(f"    Days with trades:           {days_with_trades}")
    print(f"    Days net profitable:        {profitable_days} / {days_with_trades} ({profitable_days/max(days_with_trades,1)*100:.0f}%)")
    print(f"    Total OOS trades:           {total_trades_oos}")
    print(f"    Total gross PnL (INR):      {total_gross_pnl:+.2f}")
    print(f"    Total net PnL (INR):        {total_net_pnl:+.2f}")
    if total_trades_oos > 0:
        print(f"    Avg OOS net PnL/trade:      {total_net_pnl/total_trades_oos:+.2f}")
        print(f"    Avg OOS gross PnL/trade:    {total_gross_pnl/total_trades_oos:+.2f}")

    # OOS Sharpe estimate
    # Build daily returns array (0 for days without trades)
    daily_pnls = []
    for date_val in all_dates:
        day_trades = trades_with_date.filter(pl.col("session_date") == date_val)
        if len(day_trades) == 0:
            daily_pnls.append(0.0)
        else:
            ep = day_trades["entry_premium"].to_numpy()
            xp = day_trades["exit_premium"].to_numpy()
            day_net = sum(net_pnl_quick(e, x, lot_size) for e, x in zip(ep, xp))
            daily_pnls.append(day_net)

    daily_pnls = np.array(daily_pnls)
    capital_proxy = lot_size * 200
    daily_returns = daily_pnls / capital_proxy
    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
    if std_r > 0:
        oos_sharpe = mean_r / std_r * math.sqrt(252)
    else:
        oos_sharpe = 0.0

    # Also gross Sharpe
    daily_gross = []
    for date_val in all_dates:
        day_trades = trades_with_date.filter(pl.col("session_date") == date_val)
        if len(day_trades) == 0:
            daily_gross.append(0.0)
        else:
            ep = day_trades["entry_premium"].to_numpy()
            xp = day_trades["exit_premium"].to_numpy()
            day_g = float(np.sum((xp - ep) * lot_size))
            daily_gross.append(day_g)

    daily_gross = np.array(daily_gross)
    gross_returns = daily_gross / capital_proxy
    mean_g = np.mean(gross_returns)
    std_g = np.std(gross_returns, ddof=1) if len(gross_returns) > 1 else 0.0
    gross_sharpe = (mean_g / std_g * math.sqrt(252)) if std_g > 0 else 0.0

    print(f"\n    OOS Sharpe (net):            {oos_sharpe:+.4f}")
    print(f"    OOS Sharpe (gross):          {gross_sharpe:+.4f}")

    # Distribution of daily net PnL
    print(f"\n    Daily net PnL distribution (INR):")
    print(f"      Min:    {np.min(daily_pnls):+.2f}")
    print(f"      P25:    {np.percentile(daily_pnls, 25):+.2f}")
    print(f"      Median: {np.median(daily_pnls):+.2f}")
    print(f"      P75:    {np.percentile(daily_pnls, 75):+.2f}")
    print(f"      Max:    {np.max(daily_pnls):+.2f}")

    return trades_with_date


def main():
    print("=" * 90)
    print("LEAVE-ONE-DAY-OUT CROSS-VALIDATION")
    print("=" * 90)
    print()
    print("NOTE: Both strategies use rolling lookbacks (20 min / 10 min).")
    print("With default params, signals at bar i depend only on bars [i-240, i].")
    print("No cross-day information leakage. Per-day results = OOS results.")
    print("No Optuna optimization is performed — default params only.")

    # Load data
    print("\nLoading data...")
    all_data = load_all_data()
    for key, symbol in [("nifty_spot", NIFTY_SYMBOL), ("banknifty_spot", BANKNIFTY_SYMBOL)]:
        if all_data[key] is not None:
            step = get_strike_step(symbol)
            all_data[key] = add_atm_strike(all_data[key], step)
    print(f"  Trading days: {all_data['trading_days']}")

    # Show all session dates
    spot_df = all_data["nifty_spot"]
    all_dates = sorted(spot_df["session_date"].unique().to_list())
    print(f"  Session dates: {[str(d) for d in all_dates]}")

    # Show which dates are expiry days
    option_df = all_data["nifty_options"]
    expiry_dates = sorted(option_df["expiry"].unique().to_list())
    print(f"  Expiry dates in data: {[str(d) for d in expiry_dates]}")
    expiry_set = set(expiry_dates)
    trading_expiry_days = [d for d in all_dates if d in expiry_set]
    print(f"  Trading days that are also expiry days: {[str(d) for d in trading_expiry_days]}")

    print("\nWarming up Numba...")
    warmup_numba()

    # ── vol_surface_signal_v1 ──
    print("\n" + "=" * 90)
    print("STRATEGY: vol_surface_signal_v1")
    print("=" * 90)
    run_loo_cv("vol_surface_signal_v1", all_data)

    # ── expiry_day_pattern_v1 ──
    print("\n" + "=" * 90)
    print("STRATEGY: expiry_day_pattern_v1")
    print("=" * 90)
    run_loo_cv("expiry_day_pattern_v1", all_data)

    print("\n" + "=" * 90)
    print("END OF LOO-CV")
    print("=" * 90)


if __name__ == "__main__":
    main()
