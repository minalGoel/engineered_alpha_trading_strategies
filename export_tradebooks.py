#!/usr/bin/env python3
"""Export full tradebook CSV for each of the 6 strategies."""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import importlib
import numpy as np
import polars as pl
from pathlib import Path

from pipeline.config import NIFTY_LOT, BANKNIFTY_LOT, NIFTY_SYMBOL, BANKNIFTY_SYMBOL
from pipeline.data_loader import load_all_data, add_atm_strike
from pipeline.option_utils import get_strike_step
from pipeline.state_machine import warmup_numba
from pipeline.run_all import backtest_strategy
from pipeline.cost_model import compute_trade_costs, net_pnl_quick

STRATEGIES = [
    "implied_vol_dislocation_v1",
    "implied_vs_realized_vol_v1",
    "intraday_vol_pattern_v1",
    "vol_surface_signal_v1",
    "nifty_banknifty_spread_v1",
    "expiry_day_pattern_v1",
]

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

    warmup_numba()

    for name in STRATEGIES:
        print(f"\n  {name}...")
        strategy = load_strategy(name)
        underlying = strategy.underlying.upper()
        if underlying in ("NIFTY", "NSE:NIFTY50-INDEX"):
            spot_df = all_data["nifty_spot"]
            option_df = all_data["nifty_options"]
            lot_size = NIFTY_LOT
        elif underlying in ("BANKNIFTY", "NSE:NIFTYBANK-INDEX"):
            spot_df = all_data["banknifty_spot"]
            option_df = all_data["banknifty_options"]
            lot_size = BANKNIFTY_LOT
        else:
            spot_df = all_data["nifty_spot"]
            option_df = all_data["nifty_options"]
            lot_size = NIFTY_LOT

        vix_df = all_data["vix"]
        trades_df, metrics = backtest_strategy(strategy, spot_df, option_df, vix_df, lot_size)

        if trades_df is None or trades_df.is_empty():
            print(f"    0 trades — skipping")
            continue

        # Enrich with cost breakdown and session_date
        spot_dates = spot_df["session_date"].to_list()
        spot_atm = spot_df["atm_strike"].to_numpy()
        spot_close_arr = spot_df["close"].to_numpy()
        entry_bars = trades_df["entry_bar"].to_numpy()
        exit_bars = trades_df["exit_bar"].to_numpy()
        entry_p = trades_df["entry_premium"].to_numpy()
        exit_p = trades_df["exit_premium"].to_numpy()

        session_dates = [spot_dates[int(b)] for b in entry_bars]
        atm_strikes = [float(spot_atm[int(b)]) for b in entry_bars]
        spot_at_entry = [float(spot_close_arr[int(b)]) for b in entry_bars]
        spot_at_exit = [float(spot_close_arr[int(b)]) for b in exit_bars]

        gross_pnl_pts = exit_p - entry_p
        gross_pnl_inr = gross_pnl_pts * lot_size
        hold_seconds = trades_df["holding_bars"].to_numpy() * 5

        spread_costs = []
        stt_costs = []
        brokerage_costs = []
        exchange_costs = []
        net_pnls = []
        total_costs = []
        for ep, xp in zip(entry_p, exit_p):
            c = compute_trade_costs(ep, xp, lot_size)
            spread_costs.append(c["spread_cost"])
            stt_costs.append(c["stt"])
            brokerage_costs.append(c["brokerage"])
            exchange_costs.append(c["exchange"])
            total_costs.append(c["total_cost"])
            net_pnls.append(c["net_pnl"])

        side_labels = ["CE" if s == 1 else "PE" for s in trades_df["side"].to_list()]

        out = pl.DataFrame({
            "trade_no": list(range(1, len(trades_df) + 1)),
            "session_date": session_dates,
            "entry_time": trades_df["entry_time"].to_list(),
            "exit_time": trades_df["exit_time"].to_list(),
            "side": side_labels,
            "atm_strike": atm_strikes,
            "spot_at_entry": spot_at_entry,
            "spot_at_exit": spot_at_exit,
            "entry_premium": entry_p.tolist(),
            "exit_premium": exit_p.tolist(),
            "holding_bars": trades_df["holding_bars"].to_list(),
            "holding_seconds": hold_seconds.tolist(),
            "exit_reason": trades_df["exit_reason"].to_list(),
            "gross_pnl_pts": gross_pnl_pts.tolist(),
            "gross_pnl_inr": gross_pnl_inr.tolist(),
            "spread_cost_inr": spread_costs,
            "stt_inr": stt_costs,
            "brokerage_inr": brokerage_costs,
            "exchange_inr": exchange_costs,
            "total_cost_inr": total_costs,
            "net_pnl_inr": net_pnls,
        })

        csv_path = OUT_DIR / f"{name}.csv"
        out.write_csv(csv_path)
        print(f"    {len(out)} trades → {csv_path}")

    print(f"\nDone. All tradebooks in {OUT_DIR}/")


if __name__ == "__main__":
    main()
