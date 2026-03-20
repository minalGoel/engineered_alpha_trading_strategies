#!/usr/bin/env python3
"""Diagnose the CE premium vs spot inconsistency."""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import numpy as np
import polars as pl

from pipeline.config import NIFTY_LOT, NIFTY_SYMBOL
from pipeline.data_loader import load_all_data, add_atm_strike
from pipeline.option_utils import get_strike_step
from pipeline.run_all import build_option_premium_arrays

print("Loading data...")
all_data = load_all_data()
spot_df = all_data["nifty_spot"]
step = get_strike_step(NIFTY_SYMBOL)
spot_df = add_atm_strike(spot_df, step)
option_df = all_data["nifty_options"]

# Build the premium arrays as the pipeline does
option_ce_close, option_pe_close, is_expiry = build_option_premium_arrays(
    spot_df, option_df, "NIFTY"
)

# Look at the bars around trade 1 entry (09:47:00 on 2026-03-04)
spot_times = spot_df["datetime"].to_list()
spot_close = spot_df["close"].to_numpy()
atm_strike = spot_df["atm_strike"].to_numpy()

print("\nBars around 09:47:00 on 2026-03-04:")
print(f"{'Bar':>5}  {'Time':>25}  {'Spot':>10}  {'ATM Strike':>12}  {'CE Prem':>10}  {'PE Prem':>10}")
print("-" * 80)

for i in range(len(spot_df)):
    t = spot_times[i]
    if str(t).startswith("2026-03-04") and "09:46" <= str(t)[11:16] <= "09:48":
        print(f"{i:>5}  {str(t):>25}  {spot_close[i]:>10.2f}  {atm_strike[i]:>12.0f}  {option_ce_close[i]:>10.2f}  {option_pe_close[i]:>10.2f}")

print("\n" + "=" * 80)
print("DIAGNOSIS: What strike is each premium coming from?")
print("=" * 80)

# The entry bar and exit bar from the trade
# Trade 1: entry_time=09:47:00, exit_time=09:47:10, holding_bars=2
# Find those bars
entry_idx = None
exit_idx = None
for i in range(len(spot_df)):
    t = str(spot_times[i])
    if t == "2026-03-04 09:47:00":
        entry_idx = i
    if t == "2026-03-04 09:47:10":
        exit_idx = i

if entry_idx and exit_idx:
    print(f"\nEntry bar {entry_idx}:")
    print(f"  Spot close:     {spot_close[entry_idx]:.2f}")
    print(f"  ATM strike:     {atm_strike[entry_idx]:.0f}")
    print(f"  CE premium:     {option_ce_close[entry_idx]:.2f}")
    print(f"  → This is the close of CE at strike {atm_strike[entry_idx]:.0f}")

    print(f"\nExit bar {exit_idx}:")
    print(f"  Spot close:     {spot_close[exit_idx]:.2f}")
    print(f"  ATM strike:     {atm_strike[exit_idx]:.0f}")
    print(f"  CE premium:     {option_ce_close[exit_idx]:.2f}")
    print(f"  → This is the close of CE at strike {atm_strike[exit_idx]:.0f}")

    if atm_strike[entry_idx] != atm_strike[exit_idx]:
        print(f"\n  *** BUG CONFIRMED ***")
        print(f"  ATM strike shifted from {atm_strike[entry_idx]:.0f} to {atm_strike[exit_idx]:.0f}")
        print(f"  Entry premium is from {atm_strike[entry_idx]:.0f} CE")
        print(f"  Exit premium is from {atm_strike[exit_idx]:.0f} CE")
        print(f"  PnL is computed across TWO DIFFERENT CONTRACTS!")
        print(f"  The 24350 CE is deeper ITM → naturally higher premium than 24400 CE")
    else:
        print(f"\n  ATM strike did NOT change. Issue is elsewhere.")

    # Verify: what is the 24400 CE premium at exit time?
    print(f"\nVerification — actual 24400 CE premium at exit time:")
    exit_date = spot_df["session_date"].to_list()[exit_idx]
    exit_time = spot_times[exit_idx]
    ce_24400_at_exit = option_df.filter(
        (pl.col("session_date") == exit_date) &
        (pl.col("strike") == 24400.0) &
        (pl.col("option_type") == "CE") &
        (pl.col("datetime") <= exit_time)
    ).sort("datetime").tail(1)
    if not ce_24400_at_exit.is_empty():
        real_exit_prem = ce_24400_at_exit["close"].to_list()[0]
        print(f"  24400 CE close at {exit_time}: {real_exit_prem:.2f}")
        real_pnl = (real_exit_prem - option_ce_close[entry_idx]) * 75
        print(f"  Correct gross PnL: ({real_exit_prem:.2f} - {option_ce_close[entry_idx]:.2f}) × 75 = ₹{real_pnl:.2f}")
        print(f"  Pipeline gross PnL: ({option_ce_close[exit_idx]:.2f} - {option_ce_close[entry_idx]:.2f}) × 75 = ₹{(option_ce_close[exit_idx]-option_ce_close[entry_idx])*75:.2f}")

# Count how many bar-to-bar ATM strike changes happen
changes = np.sum(np.diff(atm_strike) != 0)
print(f"\n\nTotal ATM strike changes across all bars: {changes} / {len(atm_strike)-1}")
print(f"Percentage of bars where ATM strike shifts: {changes/(len(atm_strike)-1)*100:.1f}%")
