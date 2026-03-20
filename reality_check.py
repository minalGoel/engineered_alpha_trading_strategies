#!/usr/bin/env python3
"""Reality check: comprehensive analysis of all 6 strategies across 12 days."""
from __future__ import annotations
import os
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import sys
import importlib
import numpy as np
import polars as pl
import math

from pipeline.config import (
    NIFTY_LOT, BANKNIFTY_LOT, NIFTY_SYMBOL, BANKNIFTY_SYMBOL,
    TOTAL_TRADING_DAYS,
)
from pipeline.data_loader import load_all_data, add_atm_strike
from pipeline.option_utils import get_strike_step
from pipeline.state_machine import run_state_machine, warmup_numba, EXIT_REASON_MAP
from pipeline.run_all import build_option_premium_arrays, backtest_strategy
from pipeline.cost_model import (
    compute_trade_costs, net_pnl_quick, SPREAD_POINTS, STT_RATE,
    BROKERAGE_PER_ORDER, EXCHANGE_PER_LOT,
)

STRATEGIES = [
    "implied_vol_dislocation_v1",
    "implied_vs_realized_vol_v1",
    "intraday_vol_pattern_v1",
    "vol_surface_signal_v1",
    "nifty_banknifty_spread_v1",
    "expiry_day_pattern_v1",
]


def load_strategy(name):
    mod = importlib.import_module(f"pipeline.strategies.{name}")
    return mod.Strategy()


def compute_sharpe_custom(trades_df, lot_size, total_days, use_gross=False):
    """Compute Sharpe from trades. If use_gross=True, use gross PnL (no costs)."""
    if trades_df is None or trades_df.is_empty() or len(trades_df) < 2:
        return 0.0

    entry_p = trades_df["entry_premium"].to_numpy()
    exit_p = trades_df["exit_premium"].to_numpy()

    if use_gross:
        pnl_arr = (exit_p - entry_p) * lot_size
    else:
        pnl_arr = np.array([net_pnl_quick(ep, xp, lot_size) for ep, xp in zip(entry_p, exit_p)])

    trades_with_pnl = trades_df.with_columns(pl.Series("_pnl", pnl_arr))
    trades_with_date = trades_with_pnl.with_columns(
        pl.col("exit_time").cast(pl.Date).alias("trade_date")
    )
    daily_pnl = trades_with_date.group_by("trade_date").agg(
        pl.col("_pnl").sum().alias("daily_pnl")
    ).sort("trade_date")

    capital_proxy = lot_size * 200
    daily_returns = daily_pnl["daily_pnl"].to_numpy() / max(capital_proxy, 1)
    days_with = len(daily_returns)
    zero_fill = max(0, total_days - days_with)
    if zero_fill > 0:
        daily_returns = np.concatenate([daily_returns, np.zeros(zero_fill)])

    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
    if std_r > 0:
        return float(mean_r / std_r * math.sqrt(252))
    return 0.0


def compute_sharpe_with_spread(trades_df, lot_size, total_days, spread_per_side):
    """Compute Sharpe with custom spread cost."""
    if trades_df is None or trades_df.is_empty() or len(trades_df) < 2:
        return 0.0

    entry_p = trades_df["entry_premium"].to_numpy()
    exit_p = trades_df["exit_premium"].to_numpy()
    qty = lot_size

    pnl_arr = np.array([
        (xp - ep) * qty - (spread_per_side * 2 * qty + xp * STT_RATE * qty + BROKERAGE_PER_ORDER * 2 + EXCHANGE_PER_LOT * 2)
        for ep, xp in zip(entry_p, exit_p)
    ])

    trades_with_pnl = trades_df.with_columns(pl.Series("_pnl", pnl_arr))
    trades_with_date = trades_with_pnl.with_columns(
        pl.col("exit_time").cast(pl.Date).alias("trade_date")
    )
    daily_pnl = trades_with_date.group_by("trade_date").agg(
        pl.col("_pnl").sum().alias("daily_pnl")
    ).sort("trade_date")

    capital_proxy = lot_size * 200
    daily_returns = daily_pnl["daily_pnl"].to_numpy() / max(capital_proxy, 1)
    days_with = len(daily_returns)
    zero_fill = max(0, total_days - days_with)
    if zero_fill > 0:
        daily_returns = np.concatenate([daily_returns, np.zeros(zero_fill)])

    mean_r = np.mean(daily_returns)
    std_r = np.std(daily_returns, ddof=1) if len(daily_returns) > 1 else 0.0
    if std_r > 0:
        return float(mean_r / std_r * math.sqrt(252))
    return 0.0


def main():
    print("=" * 80)
    print("REALITY CHECK: 6 Strategies × 12 Days — Full Pipeline")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    all_data = load_all_data()
    for key, symbol in [("nifty_spot", NIFTY_SYMBOL), ("banknifty_spot", BANKNIFTY_SYMBOL)]:
        if all_data[key] is not None:
            step = get_strike_step(symbol)
            all_data[key] = add_atm_strike(all_data[key], step)
    print(f"  Trading days: {all_data['trading_days']}")

    # Warmup numba
    print("Warming up Numba...")
    warmup_numba()

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 1 & 2: Run each strategy, print detailed metrics
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 1 & 2: Per-Strategy Backtest Results (default params, all 12 days)")
    print("=" * 80)

    all_trades = {}  # store for later sections
    best_sharpe_net = -999
    best_strategy = None

    for name in STRATEGIES:
        print(f"\n{'─' * 70}")
        print(f"  STRATEGY: {name}")
        print(f"{'─' * 70}")

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
            print("  ** NO DATA **")
            continue

        trades_df, metrics = backtest_strategy(strategy, spot_df, option_df, vix_df, lot_size)
        n_trades = metrics.get("total_trades", 0)

        if n_trades == 0:
            print("  ** ZERO TRADES **")
            continue

        all_trades[name] = (trades_df, lot_size)

        # Gross vs Net analysis
        entry_p = trades_df["entry_premium"].to_numpy()
        exit_p = trades_df["exit_premium"].to_numpy()
        gross_pnl_points = exit_p - entry_p  # per trade, in option points
        gross_pnl_inr = gross_pnl_points * lot_size

        # Compute full cost breakdown per trade
        spread_costs = []
        stt_costs = []
        brokerage_costs = []
        exchange_costs = []
        net_pnls = []
        for ep, xp in zip(entry_p, exit_p):
            costs = compute_trade_costs(ep, xp, lot_size)
            spread_costs.append(costs["spread_cost"])
            stt_costs.append(costs["stt"])
            brokerage_costs.append(costs["brokerage"])
            exchange_costs.append(costs["exchange"])
            net_pnls.append(costs["net_pnl"])

        spread_costs = np.array(spread_costs)
        stt_costs = np.array(stt_costs)
        brokerage_costs = np.array(brokerage_costs)
        exchange_costs = np.array(exchange_costs)
        net_pnls = np.array(net_pnls)
        total_costs = spread_costs + stt_costs + brokerage_costs + exchange_costs

        # Win rates
        gross_wins = np.sum(gross_pnl_inr > 0) / n_trades
        net_wins = np.sum(net_pnls > 0) / n_trades

        # Sharpe gross vs net
        total_days = all_data["trading_days"]
        sharpe_gross = compute_sharpe_custom(trades_df, lot_size, total_days, use_gross=True)
        sharpe_net = compute_sharpe_custom(trades_df, lot_size, total_days, use_gross=False)

        # Holding time
        holding_bars = trades_df["holding_bars"].to_numpy()
        avg_hold_sec = np.mean(holding_bars) * 5

        print(f"  Underlying:     {underlying}")
        print(f"  Lot size:       {lot_size}")
        print(f"  Total trades:   {n_trades}")
        print()
        print(f"  Gross PnL/trade (option pts):  {np.mean(gross_pnl_points):+.4f}")
        print(f"  Net PnL/trade   (option pts):  {np.mean(net_pnls / lot_size):+.4f}")
        print(f"  Gross PnL/trade (INR):         {np.mean(gross_pnl_inr):+.2f}")
        print(f"  Net PnL/trade   (INR):         {np.mean(net_pnls):+.2f}")
        print()
        print(f"  Total cost/trade breakdown (INR avg):")
        print(f"    Spread:      {np.mean(spread_costs):8.2f}  ({np.mean(spread_costs)/np.mean(total_costs)*100:.1f}%)")
        print(f"    STT:         {np.mean(stt_costs):8.2f}  ({np.mean(stt_costs)/np.mean(total_costs)*100:.1f}%)")
        print(f"    Brokerage:   {np.mean(brokerage_costs):8.2f}  ({np.mean(brokerage_costs)/np.mean(total_costs)*100:.1f}%)")
        print(f"    Exchange:    {np.mean(exchange_costs):8.2f}  ({np.mean(exchange_costs)/np.mean(total_costs)*100:.1f}%)")
        print(f"    TOTAL:       {np.mean(total_costs):8.2f}")
        print()
        print(f"  Avg hold time:  {avg_hold_sec:.1f} seconds  ({np.mean(holding_bars):.1f} bars)")
        print()
        print(f"  Win rate (gross): {gross_wins:.4f}  ({gross_wins*100:.1f}%)")
        print(f"  Win rate (net):   {net_wins:.4f}  ({net_wins*100:.1f}%)")
        print()
        print(f"  Sharpe (gross):   {sharpe_gross:+.4f}")
        print(f"  Sharpe (net):     {sharpe_net:+.4f}")
        print()
        print(f"  Total gross PnL (INR): {np.sum(gross_pnl_inr):+.2f}")
        print(f"  Total net PnL   (INR): {np.sum(net_pnls):+.2f}")
        print(f"  Total costs     (INR): {np.sum(total_costs):.2f}")

        if sharpe_net > best_sharpe_net:
            best_sharpe_net = sharpe_net
            best_strategy = name

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 3: Actual bid-ask spread analysis
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 3: Actual Bid-Ask Spread in Option Data")
    print("=" * 80)

    option_df = all_data["nifty_options"]
    print(f"\n  Option candles columns: {option_df.columns}")

    has_bid = "bid" in option_df.columns or "bid_price" in option_df.columns
    has_ask = "ask" in option_df.columns or "ask_price" in option_df.columns

    if has_bid and has_ask:
        bid_col = "bid" if "bid" in option_df.columns else "bid_price"
        ask_col = "ask" if "ask" in option_df.columns else "ask_price"
        print(f"\n  Found bid/ask columns: {bid_col}, {ask_col}")
    else:
        print("\n  ** NO BID/ASK COLUMNS in option_candles.parquet **")
        print("  Available columns: ", option_df.columns)
        print("  The ₹1.5/side spread assumption is a GUESS — no market microstructure data.")
        print()
        print("  Proxy analysis using high-low as spread proxy for ATM CE options:")

        # Filter to ATM NIFTY CE options
        spot_df = all_data["nifty_spot"]
        # Get rough ATM strike range
        avg_spot = spot_df["close"].mean()
        step = 50
        atm_approx = round(avg_spot / step) * step

        atm_opts = option_df.filter(
            (pl.col("option_type") == "CE") &
            (pl.col("strike") >= atm_approx - 100) &
            (pl.col("strike") <= atm_approx + 100)
        )

        if not atm_opts.is_empty():
            hl_spread = (atm_opts["high"] - atm_opts["low"])
            # Filter to bars where there was actual movement (non-zero spread)
            valid_spread = hl_spread.filter(hl_spread > 0)
            print(f"\n  ATM CE options (strike {atm_approx-100} to {atm_approx+100}):")
            print(f"  Bars analyzed: {len(hl_spread)} total, {len(valid_spread)} with H≠L")
            print(f"  High-Low spread (proxy for intra-bar range, NOT bid-ask):")
            print(f"    Median:          {valid_spread.median():.2f} pts")
            print(f"    25th percentile: {valid_spread.quantile(0.25):.2f} pts")
            print(f"    75th percentile: {valid_spread.quantile(0.75):.2f} pts")
            print(f"    Mean:            {valid_spread.mean():.2f} pts")
            print()
            print(f"  CAVEAT: High-Low is NOT the same as bid-ask spread.")
            print(f"  It's the range of traded prices in 5 seconds.")
            print(f"  True bid-ask spread at any instant is likely TIGHTER than H-L.")
            print(f"  But H-L gives a rough upper bound on what you'd lose to spread.")

            # Also check: how many bars have zero volume (no trading)
            zero_vol = atm_opts.filter(pl.col("volume") == 0)
            print(f"\n  Zero-volume bars: {len(zero_vol)} / {len(atm_opts)} ({len(zero_vol)/len(atm_opts)*100:.1f}%)")
            print(f"  (Zero volume = no real price, using stale quote)")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 4: Sensitivity analysis on spread cost
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print(f"SECTION 4: Spread Cost Sensitivity — Best Strategy: {best_strategy}")
    print("=" * 80)

    if best_strategy and best_strategy in all_trades:
        trades_df, lot_size = all_trades[best_strategy]
        total_days = all_data["trading_days"]

        spread_values = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
        print(f"\n  {'Spread/side':>12}  {'Sharpe':>10}  {'Net PnL/trade':>15}  {'Total Net PnL':>15}  {'Win Rate':>10}")
        print(f"  {'─'*12}  {'─'*10}  {'─'*15}  {'─'*15}  {'─'*10}")

        for sp in spread_values:
            sharpe = compute_sharpe_with_spread(trades_df, lot_size, total_days, sp)
            entry_p = trades_df["entry_premium"].to_numpy()
            exit_p = trades_df["exit_premium"].to_numpy()
            qty = lot_size
            pnls = np.array([
                (xp - ep) * qty - (sp * 2 * qty + xp * STT_RATE * qty + BROKERAGE_PER_ORDER * 2 + EXCHANGE_PER_LOT * 2)
                for ep, xp in zip(entry_p, exit_p)
            ])
            win_rate = np.sum(pnls > 0) / len(pnls)
            print(f"  {sp:>10.2f}pts  {sharpe:>+10.4f}  {np.mean(pnls):>+15.2f}  {np.sum(pnls):>+15.2f}  {win_rate:>9.1%}")

        # Also show gross (zero cost) for reference
        gross_sharpe = compute_sharpe_custom(trades_df, lot_size, total_days, use_gross=True)
        gross_pnls = (trades_df["exit_premium"].to_numpy() - trades_df["entry_premium"].to_numpy()) * lot_size
        gross_wr = np.sum(gross_pnls > 0) / len(gross_pnls)
        print(f"\n  {'GROSS(0cost)':>12}  {gross_sharpe:>+10.4f}  {np.mean(gross_pnls):>+15.2f}  {np.sum(gross_pnls):>+15.2f}  {gross_wr:>9.1%}")
    else:
        print("  No strategy had trades to analyze.")

    # ═══════════════════════════════════════════════════════════════════
    # SECTION 5: Market microstructure context
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("SECTION 5: Market Microstructure Context")
    print("=" * 80)

    spot_df = all_data["nifty_spot"]
    option_df = all_data["nifty_options"]

    # 5a: Average 5-second move in NIFTY spot
    spot_close = spot_df["close"].to_numpy().astype(np.float64)
    spot_changes = np.abs(np.diff(spot_close))
    spot_changes_signed = np.diff(spot_close)

    print(f"\n  NIFTY Spot — 5-second bar-to-bar moves:")
    print(f"    Mean |move|:    {np.mean(spot_changes):.4f} pts")
    print(f"    Median |move|:  {np.median(spot_changes):.4f} pts")
    print(f"    Std of moves:   {np.std(spot_changes_signed):.4f} pts")
    print(f"    P95 |move|:     {np.percentile(spot_changes, 95):.4f} pts")
    print(f"    P99 |move|:     {np.percentile(spot_changes, 99):.4f} pts")
    print(f"    Total bars:     {len(spot_close)}")

    # 5b: Average 5-second move in ATM NIFTY CE option
    avg_spot = spot_df["close"].mean()
    step = 50
    atm_approx = round(avg_spot / step) * step

    atm_ce = option_df.filter(
        (pl.col("option_type") == "CE") &
        (pl.col("strike") >= atm_approx - 50) &
        (pl.col("strike") <= atm_approx + 50)
    ).sort(["strike", "datetime"])

    if not atm_ce.is_empty():
        # Process per strike per day
        opt_moves = []
        for strike_val in atm_ce["strike"].unique().to_list():
            for day_val in atm_ce["day_id"].unique().to_list():
                chunk = atm_ce.filter(
                    (pl.col("strike") == strike_val) &
                    (pl.col("day_id") == day_val)
                ).sort("datetime")
                if len(chunk) > 1:
                    closes = chunk["close"].to_numpy()
                    moves = np.abs(np.diff(closes))
                    opt_moves.extend(moves.tolist())

        opt_moves = np.array(opt_moves)
        opt_moves_nonzero = opt_moves[opt_moves > 0]

        print(f"\n  ATM NIFTY CE Option — 5-second bar-to-bar moves:")
        print(f"    Mean |move|:    {np.mean(opt_moves):.4f} pts")
        print(f"    Median |move|:  {np.median(opt_moves):.4f} pts")
        print(f"    Mean (non-zero): {np.mean(opt_moves_nonzero):.4f} pts")
        print(f"    Std of moves:   {np.std(opt_moves):.4f} pts")
        print(f"    P95 |move|:     {np.percentile(opt_moves, 95):.4f} pts")
        print(f"    P99 |move|:     {np.percentile(opt_moves, 99):.4f} pts")
        print(f"    Zero-move bars: {np.sum(opt_moves == 0)}/{len(opt_moves)} ({np.sum(opt_moves == 0)/len(opt_moves)*100:.1f}%)")
        print(f"    Total bars:     {len(opt_moves)}")

        # 5c: Feasibility calculation
        avg_opt_move = np.mean(opt_moves)
        breakeven_cost_pts = SPREAD_POINTS * 2 + 0  # just spread for simplicity

        # With NIFTY lot=75, the full cost per trade
        avg_entry = atm_ce["close"].mean()
        sample_cost = compute_trade_costs(avg_entry, avg_entry, NIFTY_LOT)
        total_cost_pts = sample_cost["total_cost"] / NIFTY_LOT  # convert INR back to points

        print(f"\n  Feasibility Check:")
        print(f"    Average option move per 5-sec bar: {avg_opt_move:.4f} pts")
        print(f"    Average entry premium (ATM CE):    {avg_entry:.2f} pts")
        print(f"    Total round-trip cost (points):    {total_cost_pts:.4f} pts")
        print(f"    Bars of directional move needed:   {total_cost_pts / max(avg_opt_move, 0.001):.1f} bars")
        print(f"    = {total_cost_pts / max(avg_opt_move, 0.001) * 5:.0f} seconds of uninterrupted directional movement")
        print()

        # Also: how many consecutive same-direction moves happen?
        print(f"  Consecutive same-direction move analysis (ATM CE, all strikes/days):")
        all_run_lengths = []
        for strike_val in atm_ce["strike"].unique().to_list():
            for day_val in atm_ce["day_id"].unique().to_list():
                chunk = atm_ce.filter(
                    (pl.col("strike") == strike_val) &
                    (pl.col("day_id") == day_val)
                ).sort("datetime")
                if len(chunk) > 2:
                    closes = chunk["close"].to_numpy()
                    diffs = np.diff(closes)
                    # Count consecutive positive moves
                    run_len = 0
                    for d in diffs:
                        if d > 0:
                            run_len += 1
                        else:
                            if run_len > 0:
                                all_run_lengths.append(run_len)
                            run_len = 0
                    if run_len > 0:
                        all_run_lengths.append(run_len)

        if all_run_lengths:
            runs = np.array(all_run_lengths)
            print(f"    Mean consecutive up-move run:     {np.mean(runs):.2f} bars")
            print(f"    Median:                           {np.median(runs):.1f} bars")
            print(f"    P75:                              {np.percentile(runs, 75):.1f} bars")
            print(f"    P90:                              {np.percentile(runs, 90):.1f} bars")
            print(f"    P95:                              {np.percentile(runs, 95):.1f} bars")
            print(f"    Max:                              {np.max(runs)} bars")
            pct_above_needed = np.sum(runs >= total_cost_pts / max(avg_opt_move, 0.001)) / len(runs) * 100
            print(f"    Runs >= breakeven ({total_cost_pts/max(avg_opt_move,0.001):.0f} bars): {pct_above_needed:.1f}% of all runs")

    print("\n" + "=" * 80)
    print("END OF REALITY CHECK")
    print("=" * 80)


if __name__ == "__main__":
    main()
