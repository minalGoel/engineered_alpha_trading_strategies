#!/usr/bin/env python3
"""Full rebuild: run all 6 strategies with corrected pipeline, verify against raw parquet.

Step 5: For every trade, cross-check entry and exit premiums against raw option parquet.
Step 6: Report honest aggregate numbers.
"""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import importlib
import numpy as np
import polars as pl
import math
from pathlib import Path

from pipeline.config import (
    NIFTY_LOT, BANKNIFTY_LOT, NIFTY_SYMBOL, BANKNIFTY_SYMBOL,
    OPTION_FILE,
)
from pipeline.data_loader import load_all_data, add_atm_strike
from pipeline.option_utils import get_strike_step
from pipeline.state_machine import warmup_numba
from pipeline.run_all import backtest_strategy
from pipeline.cost_model import compute_trade_costs, net_pnl_quick, STT_RATE, BROKERAGE_PER_ORDER, EXCHANGE_PER_LOT

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


def load_raw_option_parquet():
    """Load raw option parquet with UTC→IST conversion for cross-checking."""
    from pipeline.data_loader import _utc_to_ist
    df = pl.read_parquet(OPTION_FILE)
    df = _utc_to_ist(df)
    return df


def cross_check_trade(raw_opts, session_date, entry_strike, opt_type, expiry,
                      entry_time, exit_time, entry_prem_pipeline, exit_prem_pipeline):
    """Cross-check a single trade against raw option parquet.

    Returns (raw_entry_prem, raw_exit_prem, match_entry, match_exit).
    """
    # Filter to the exact strike, type, expiry, date
    chain = raw_opts.filter(
        (pl.col("session_date") == session_date) &
        (pl.col("strike") == entry_strike) &
        (pl.col("option_type") == opt_type) &
        (pl.col("expiry") == expiry)
    ).sort("datetime")

    if chain.is_empty():
        return None, None, False, False

    # Find entry premium: last close at or before entry_time
    at_entry = chain.filter(pl.col("datetime") <= entry_time)
    if at_entry.is_empty():
        raw_entry = None
    else:
        raw_entry = at_entry["close"].to_list()[-1]

    # Find exit premium: last close at or before exit_time
    at_exit = chain.filter(pl.col("datetime") <= exit_time)
    if at_exit.is_empty():
        raw_exit = None
    else:
        raw_exit = at_exit["close"].to_list()[-1]

    tol = 0.01
    match_entry = raw_entry is not None and abs(raw_entry - entry_prem_pipeline) < tol
    match_exit = raw_exit is not None and abs(raw_exit - exit_prem_pipeline) < tol

    return raw_entry, raw_exit, match_entry, match_exit


def compute_sharpe(daily_pnls_arr, total_days, lot_size):
    """Sharpe from daily PnL array."""
    capital_proxy = lot_size * 200
    daily_returns = daily_pnls_arr / capital_proxy
    if len(daily_returns) < total_days:
        daily_returns = np.concatenate([daily_returns, np.zeros(total_days - len(daily_returns))])
    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
    if std_r > 0:
        return float(mean_r / std_r * math.sqrt(252))
    return 0.0


def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 110)
    print("FULL REBUILD + VERIFICATION — All 6 Strategies, Corrected Pipeline")
    print("=" * 110)

    # Load data
    print("\nLoading pipeline data...")
    all_data = load_all_data()
    for key, symbol in [("nifty_spot", NIFTY_SYMBOL), ("banknifty_spot", BANKNIFTY_SYMBOL)]:
        if all_data[key] is not None:
            step = get_strike_step(symbol)
            all_data[key] = add_atm_strike(all_data[key], step)

    print("Loading raw option parquet for cross-checks...")
    raw_opts = load_raw_option_parquet()

    print("Warming up Numba...")
    warmup_numba()
    print(f"  Trading days: {all_data['trading_days']}\n")

    # Get nearest expiry per session date (for cross-check lookups)
    def get_nearest_expiry(session_date):
        day = raw_opts.filter(pl.col("session_date") == session_date)
        if day.is_empty():
            return None
        return day["expiry"].min()

    # ══════════════════════════════════════════════════════════════════
    # Run all 6 strategies
    # ══════════════════════════════════════════════════════════════════
    summary_rows = []

    for name in STRATEGIES:
        print("=" * 110)
        print(f"  STRATEGY: {name}")
        print("=" * 110)

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

        if spot_df is None or option_df is None:
            print("  ** NO DATA **\n")
            summary_rows.append((name, 0, 0, 0, 0, 0, 0, 0, 0))
            continue

        trades_df, metrics = backtest_strategy(strategy, spot_df, option_df, vix_df, lot_size)
        n_trades = len(trades_df) if trades_df is not None else 0

        if n_trades == 0:
            print("  ** ZERO TRADES **\n")
            summary_rows.append((name, 0, 0, 0, 0, 0, 0, 0, 0))
            continue

        # ── Per-trade table ──
        print(f"\n  {'#':>3}  {'Date':>10}  {'Entry':>8}  {'Exit':>8}  {'S':>2}  "
              f"{'Strike':>7}  {'Spot@E':>8}  {'Spot@X':>8}  "
              f"{'EntPr':>7}  {'ExitPr':>7}  {'Bars':>4}  "
              f"{'Why':>6}  {'GrossPnL':>9}  {'NetPnL':>9}  "
              f"{'RawEnt':>7}  {'RawExit':>7}  {'Check':>5}")
        print(f"  {'─'*3}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*2}  "
              f"{'─'*7}  {'─'*8}  {'─'*8}  "
              f"{'─'*7}  {'─'*7}  {'─'*4}  "
              f"{'─'*6}  {'─'*9}  {'─'*9}  "
              f"{'─'*7}  {'─'*7}  {'─'*5}")

        total_gross = 0.0
        total_net = 0.0
        total_cost = 0.0
        checks_passed = 0
        checks_failed = 0
        missing_count = 0
        daily_net_pnl = {}

        session_dates_list = spot_df["session_date"].to_list()

        for row_idx in range(n_trades):
            row = trades_df.row(row_idx)
            (entry_bar, exit_bar, side_int, entry_prem, exit_prem,
             exit_reason, pnl_raw, entry_time, exit_time, holding_bars,
             entry_strike, entry_strike_idx, atm_at_entry, atm_at_exit,
             spot_at_entry, spot_at_exit, premium_missing) = row

            side_str = "CE" if side_int == 1 else "PE"
            session_date = session_dates_list[int(entry_bar)]

            costs = compute_trade_costs(entry_prem, exit_prem, lot_size)
            gross_pnl = costs["gross_pnl"]
            net_pnl = costs["net_pnl"]
            total_gross += gross_pnl
            total_net += net_pnl
            total_cost += costs["total_cost"]

            # Accumulate daily net PnL
            date_key = str(session_date)
            daily_net_pnl[date_key] = daily_net_pnl.get(date_key, 0.0) + net_pnl

            if premium_missing:
                missing_count += 1

            # Cross-check against raw parquet
            expiry = get_nearest_expiry(session_date)
            raw_entry, raw_exit, match_e, match_x = cross_check_trade(
                raw_opts, session_date, entry_strike,
                side_str, expiry,
                entry_time, exit_time,
                entry_prem, exit_prem,
            )

            both_match = match_e and match_x
            if both_match:
                checks_passed += 1
                check_str = "OK"
            else:
                checks_failed += 1
                check_str = "FAIL"

            raw_e_str = f"{raw_entry:.2f}" if raw_entry is not None else "N/A"
            raw_x_str = f"{raw_exit:.2f}" if raw_exit is not None else "N/A"

            date_str = str(session_date)
            entry_t = str(entry_time)[11:19]
            exit_t = str(exit_time)[11:19]

            print(f"  {row_idx+1:>3}  {date_str:>10}  {entry_t:>8}  {exit_t:>8}  {side_str:>2}  "
                  f"{entry_strike:>7.0f}  {spot_at_entry:>8.2f}  {spot_at_exit:>8.2f}  "
                  f"{entry_prem:>7.2f}  {exit_prem:>7.2f}  {holding_bars:>4}  "
                  f"{exit_reason:>6}  {gross_pnl:>+9.2f}  {net_pnl:>+9.2f}  "
                  f"{raw_e_str:>7}  {raw_x_str:>7}  {check_str:>5}")

        # ── Aggregate metrics ──
        gross_winners = 0
        net_winners = 0
        entry_p = trades_df["entry_premium"].to_numpy()
        exit_p = trades_df["exit_premium"].to_numpy()
        for ep, xp in zip(entry_p, exit_p):
            if (xp - ep) * lot_size > 0:
                gross_winners += 1
            c = compute_trade_costs(ep, xp, lot_size)
            if c["net_pnl"] > 0:
                net_winners += 1

        # Sharpe
        all_dates = sorted(spot_df["session_date"].unique().to_list())
        daily_arr = np.array([daily_net_pnl.get(str(d), 0.0) for d in all_dates])
        net_sharpe = compute_sharpe(daily_arr, len(all_dates), lot_size)

        # Gross sharpe
        daily_gross_pnl = {}
        for row_idx in range(n_trades):
            row = trades_df.row(row_idx)
            entry_bar = row[0]
            sd = str(session_dates_list[int(entry_bar)])
            ep, xp = row[3], row[4]
            daily_gross_pnl[sd] = daily_gross_pnl.get(sd, 0.0) + (xp - ep) * lot_size
        daily_gross_arr = np.array([daily_gross_pnl.get(str(d), 0.0) for d in all_dates])
        gross_sharpe = compute_sharpe(daily_gross_arr, len(all_dates), lot_size)

        avg_hold = np.mean(trades_df["holding_bars"].to_numpy()) * 5

        print(f"\n  VERIFICATION: {checks_passed} OK, {checks_failed} FAIL out of {n_trades} trades")
        if checks_failed > 0:
            print(f"  *** WARNING: {checks_failed} trades FAILED raw parquet cross-check ***")
        if missing_count > 0:
            print(f"  *** {missing_count} trades used last-known premium (data gap) ***")

        print(f"\n  AGGREGATE METRICS:")
        print(f"    Total trades:           {n_trades}")
        print(f"    Total gross PnL (INR):  {total_gross:+.2f}")
        print(f"    Total costs (INR):      {total_cost:.2f}")
        print(f"    Total net PnL (INR):    {total_net:+.2f}")
        print(f"    Avg gross PnL/trade:    {total_gross/n_trades:+.2f}")
        print(f"    Avg net PnL/trade:      {total_net/n_trades:+.2f}")
        print(f"    Win rate (gross):       {gross_winners}/{n_trades} ({gross_winners/n_trades*100:.1f}%)")
        print(f"    Win rate (net):         {net_winners}/{n_trades} ({net_winners/n_trades*100:.1f}%)")
        print(f"    Sharpe (gross):         {gross_sharpe:+.4f}")
        print(f"    Sharpe (net):           {net_sharpe:+.4f}")
        print(f"    Avg hold time:          {avg_hold:.0f}s ({avg_hold/5:.0f} bars)")

        summary_rows.append((
            name, n_trades, total_gross, total_net, total_cost,
            gross_winners / n_trades if n_trades > 0 else 0,
            net_winners / n_trades if n_trades > 0 else 0,
            gross_sharpe, net_sharpe,
        ))

        # ── Export CSV ──
        side_labels = ["CE" if s == 1 else "PE" for s in trades_df["side"].to_list()]
        spread_costs, stt_costs, brokerage_costs, exchange_costs, net_pnls, total_costs_list = [], [], [], [], [], []
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
            "session_date": [str(session_dates_list[int(trades_df["entry_bar"][i])]) for i in range(n_trades)],
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
            "premium_missing": trades_df["premium_missing_at_exit"].to_list(),
        })
        csv_path = OUT_DIR / f"{name}.csv"
        out.write_csv(csv_path)
        print(f"\n    CSV → {csv_path}")
        print()

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Summary table
    # ══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 110)
    print("STEP 6: HONEST NUMBERS — All 6 Strategies, Corrected Pipeline")
    print("=" * 110)
    print(f"\n  {'Strategy':>35}  {'Trades':>6}  {'GrossPnL':>10}  {'NetPnL':>10}  {'Costs':>8}  "
          f"{'WR(g)':>6}  {'WR(n)':>6}  {'Sharpe(g)':>10}  {'Sharpe(n)':>10}")
    print(f"  {'─'*35}  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}  "
          f"{'─'*6}  {'─'*6}  {'─'*10}  {'─'*10}")

    for (nm, nt, gp, np_, tc, wrg, wrn, sg, sn) in summary_rows:
        if nt == 0:
            print(f"  {nm:>35}  {nt:>6}  {'—':>10}  {'—':>10}  {'—':>8}  "
                  f"{'—':>6}  {'—':>6}  {'—':>10}  {'—':>10}")
        else:
            print(f"  {nm:>35}  {nt:>6}  {gp:>+10.0f}  {np_:>+10.0f}  {tc:>8.0f}  "
                  f"{wrg:>5.1%}  {wrn:>5.1%}  {sg:>+10.4f}  {sn:>+10.4f}")

    print(f"\n  Note: All results use default params, no optimization, all 12 days in-sample.")
    print(f"  Premium grid: strike locked at entry for entire trade duration.")
    print(f"  Every trade cross-checked against raw option_candles.parquet.")

    print("\n" + "=" * 110)
    print("END")
    print("=" * 110)


if __name__ == "__main__":
    main()
