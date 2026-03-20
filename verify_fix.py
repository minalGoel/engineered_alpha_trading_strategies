#!/usr/bin/env python3
"""Verify strike-lock fix: run vol_surface_signal_v1 and expiry_day_pattern_v1,
print detailed per-trade table for manual verification."""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import importlib
import numpy as np
import polars as pl
from pathlib import Path

from pipeline.config import NIFTY_LOT, NIFTY_SYMBOL, BANKNIFTY_SYMBOL
from pipeline.data_loader import load_all_data, add_atm_strike
from pipeline.option_utils import get_strike_step
from pipeline.state_machine import warmup_numba
from pipeline.run_all import backtest_strategy
from pipeline.cost_model import compute_trade_costs

STRATEGIES = ["vol_surface_signal_v1", "expiry_day_pattern_v1"]
OUT_DIR = Path("trades_per_strategy")


def load_strategy(name):
    mod = importlib.import_module(f"pipeline.strategies.{name}")
    return mod.Strategy()


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    all_data = load_all_data()
    for key, symbol in [("nifty_spot", NIFTY_SYMBOL), ("banknifty_spot", BANKNIFTY_SYMBOL)]:
        if all_data[key] is not None:
            step = get_strike_step(symbol)
            all_data[key] = add_atm_strike(all_data[key], step)
    print(f"  Trading days: {all_data['trading_days']}")

    print("Warming up Numba (recompiling with new signature)...")
    warmup_numba()
    print("  Done.\n")

    for name in STRATEGIES:
        print("=" * 110)
        print(f"  STRATEGY: {name}  (STRIKE-LOCKED)")
        print("=" * 110)

        strategy = load_strategy(name)
        spot_df = all_data["nifty_spot"]
        option_df = all_data["nifty_options"]
        vix_df = all_data["vix"]
        lot_size = NIFTY_LOT

        trades_df, metrics = backtest_strategy(strategy, spot_df, option_df, vix_df, lot_size)

        if trades_df is None or trades_df.is_empty():
            print("  ** ZERO TRADES **\n")
            continue

        n_trades = len(trades_df)

        # Print header
        print(f"\n  {'#':>3}  {'Entry Time':>20}  {'Exit Time':>20}  {'Side':>4}  "
              f"{'EntStrike':>9}  {'ATM@Ent':>8}  {'ATM@Exit':>8}  "
              f"{'Spot@Ent':>9}  {'Spot@Exit':>9}  "
              f"{'EntPrem':>8}  {'ExitPrem':>8}  {'Bars':>4}  {'Secs':>4}  "
              f"{'Reason':>7}  {'GrossPnL':>10}  {'Costs':>8}  {'NetPnL':>10}  {'Miss?':>5}")
        print(f"  {'─'*3}  {'─'*20}  {'─'*20}  {'─'*4}  "
              f"{'─'*9}  {'─'*8}  {'─'*8}  "
              f"{'─'*9}  {'─'*9}  "
              f"{'─'*8}  {'─'*8}  {'─'*4}  {'─'*4}  "
              f"{'─'*7}  {'─'*10}  {'─'*8}  {'─'*10}  {'─'*5}")

        total_gross = 0.0
        total_net = 0.0
        total_cost = 0.0
        missing_count = 0

        for row_idx in range(n_trades):
            row = trades_df.row(row_idx)
            # Unpack — column order from trades_df construction:
            # entry_bar, exit_bar, side, entry_premium, exit_premium,
            # exit_reason, pnl, entry_time, exit_time, holding_bars,
            # entry_strike, entry_strike_idx, atm_at_entry, atm_at_exit,
            # spot_at_entry, spot_at_exit, premium_missing_at_exit
            (entry_bar, exit_bar, side_int, entry_prem, exit_prem,
             exit_reason, pnl_raw, entry_time, exit_time, holding_bars,
             entry_strike, entry_strike_idx, atm_at_entry, atm_at_exit,
             spot_at_entry, spot_at_exit, premium_missing) = row

            side_str = "CE" if side_int == 1 else "PE"
            hold_sec = holding_bars * 5

            costs = compute_trade_costs(entry_prem, exit_prem, lot_size)
            gross_pnl = costs["gross_pnl"]
            total_cost_inr = costs["total_cost"]
            net_pnl = costs["net_pnl"]

            total_gross += gross_pnl
            total_net += net_pnl
            total_cost += total_cost_inr
            if premium_missing:
                missing_count += 1

            miss_flag = "YES" if premium_missing else ""

            # Flag strike mismatch (ATM shifted during trade)
            strike_note = ""
            if entry_strike != atm_at_exit:
                strike_note = " *"

            entry_t_str = str(entry_time)[11:19]
            exit_t_str = str(exit_time)[11:19]
            date_str = str(entry_time)[:10]

            print(f"  {row_idx+1:>3}  {date_str} {entry_t_str:>8}  {date_str} {exit_t_str:>8}  {side_str:>4}  "
                  f"{entry_strike:>9.0f}  {atm_at_entry:>8.0f}  {atm_at_exit:>8.0f}  "
                  f"{spot_at_entry:>9.2f}  {spot_at_exit:>9.2f}  "
                  f"{entry_prem:>8.2f}  {exit_prem:>8.2f}  {holding_bars:>4}  {hold_sec:>4}  "
                  f"{exit_reason:>7}  {gross_pnl:>+10.2f}  {total_cost_inr:>8.2f}  {net_pnl:>+10.2f}  {miss_flag:>5}{strike_note}")

        # Summary
        net_winners = 0
        gross_winners = 0
        for row_idx in range(n_trades):
            ep = trades_df["entry_premium"][row_idx]
            xp = trades_df["exit_premium"][row_idx]
            if (xp - ep) * lot_size > 0:
                gross_winners += 1
            c = compute_trade_costs(ep, xp, lot_size)
            if c["net_pnl"] > 0:
                net_winners += 1

        print(f"\n  SUMMARY:")
        print(f"    Total trades:           {n_trades}")
        print(f"    Total gross PnL (INR):  {total_gross:+.2f}")
        print(f"    Total costs (INR):      {total_cost:.2f}")
        print(f"    Total net PnL (INR):    {total_net:+.2f}")
        print(f"    Avg gross PnL/trade:    {total_gross/n_trades:+.2f}")
        print(f"    Avg net PnL/trade:      {total_net/n_trades:+.2f}")
        print(f"    Win rate (gross):       {gross_winners}/{n_trades} ({gross_winners/n_trades*100:.1f}%)")
        print(f"    Win rate (net):         {net_winners}/{n_trades} ({net_winners/n_trades*100:.1f}%)")
        print(f"    Premium missing trades: {missing_count}")
        if missing_count > 0:
            print(f"    *** {missing_count} trades used last-known premium at exit ***")

        # Export CSV
        entry_p = trades_df["entry_premium"].to_numpy()
        exit_p = trades_df["exit_premium"].to_numpy()
        side_labels = ["CE" if s == 1 else "PE" for s in trades_df["side"].to_list()]

        spread_costs = []
        stt_costs = []
        brokerage_costs = []
        exchange_costs = []
        net_pnls = []
        total_costs_list = []
        for ep, xp in zip(entry_p, exit_p):
            c = compute_trade_costs(ep, xp, lot_size)
            spread_costs.append(c["spread_cost"])
            stt_costs.append(c["stt"])
            brokerage_costs.append(c["brokerage"])
            exchange_costs.append(c["exchange"])
            total_costs_list.append(c["total_cost"])
            net_pnls.append(c["net_pnl"])

        out = pl.DataFrame({
            "trade_no": list(range(1, n_trades + 1)),
            "session_date": [str(trades_df["entry_time"][i])[:10] for i in range(n_trades)],
            "entry_time": trades_df["entry_time"].to_list(),
            "exit_time": trades_df["exit_time"].to_list(),
            "side": side_labels,
            "entry_strike": trades_df["entry_strike"].to_list(),
            "atm_at_entry": trades_df["atm_at_entry"].to_list(),
            "atm_at_exit": trades_df["atm_at_exit"].to_list(),
            "spot_at_entry": trades_df["spot_at_entry"].to_list(),
            "spot_at_exit": trades_df["spot_at_exit"].to_list(),
            "entry_premium": entry_p.tolist(),
            "exit_premium": exit_p.tolist(),
            "holding_bars": trades_df["holding_bars"].to_list(),
            "holding_seconds": (trades_df["holding_bars"].to_numpy() * 5).tolist(),
            "exit_reason": trades_df["exit_reason"].to_list(),
            "gross_pnl_pts": (exit_p - entry_p).tolist(),
            "gross_pnl_inr": ((exit_p - entry_p) * lot_size).tolist(),
            "spread_cost_inr": spread_costs,
            "stt_inr": stt_costs,
            "brokerage_inr": brokerage_costs,
            "exchange_inr": exchange_costs,
            "total_cost_inr": total_costs_list,
            "net_pnl_inr": net_pnls,
            "premium_missing_at_exit": trades_df["premium_missing_at_exit"].to_list(),
        })

        csv_path = OUT_DIR / f"{name}.csv"
        out.write_csv(csv_path)
        print(f"\n    CSV exported → {csv_path}")
        print()


if __name__ == "__main__":
    main()
