#!/usr/bin/env python3
"""Step 1: Understand the opportunity in NIFTY spot data.

Computes return distributions at 30s and 60s horizons,
theoretical max PnL, breakeven accuracy, and viability metrics.

Key insight: we can't scalp EVERY 30s window. Costs (~2.9 option pts)
exceed median moves (~2 option pts). The edge must come from SELECTIVE
trading — only entering when conditions predict a LARGER-than-average move.
"""
import numpy as np
import polars as pl
from pipeline.data_loader import load_spot_data
from pipeline.config import NIFTY_SYMBOL, NIFTY_LOT
from pipeline.cost_model import (
    compute_trade_costs, net_pnl_quick, min_points_to_breakeven,
)


def main():
    print("=" * 80)
    print("STEP 1: SPOT DIRECTION OPPORTUNITY ANALYSIS")
    print("=" * 80)

    spot = load_spot_data(NIFTY_SYMBOL)
    if spot is None:
        raise RuntimeError("Cannot load NIFTY spot data")

    spot = spot.sort("datetime")
    close = spot["close"].to_numpy().astype(np.float64)
    time_min = spot["time_minutes"].to_numpy().astype(np.int32)
    day_id = spot["day_id"].to_numpy().astype(np.int32)
    n = len(close)

    n_days = spot["day_id"].n_unique()
    print(f"\nData: {n:,} bars, {n_days} trading days")
    print(f"Date range: {spot['session_date'].min()} to {spot['session_date'].max()}")
    print(f"Spot range: {close.min():.0f} - {close.max():.0f}")

    # Session filter: 09:18 to 15:25
    session_mask = (time_min >= 558) & (time_min < 925)
    lot_size = NIFTY_LOT  # 65
    spread_per_side = 1.0  # ₹1.00/side as specified
    delta = 0.5
    atm_premium = 150.0  # typical ATM weekly premium

    # ── 1. Distribution of 6-bar (30-second) returns ──
    print("\n" + "─" * 70)
    print("1. DISTRIBUTION OF 6-BAR (30-SECOND) SPOT RETURNS")
    print("─" * 70)

    returns_30s = []
    for i in range(6, n):
        if session_mask[i] and session_mask[i - 6] and day_id[i] == day_id[i - 6]:
            returns_30s.append(close[i] - close[i - 6])
    returns_30s = np.array(returns_30s)
    _print_return_stats(returns_30s, "30-second")

    # ── 2. Distribution of 12-bar (60-second) returns ──
    print("\n" + "─" * 70)
    print("2. DISTRIBUTION OF 12-BAR (60-SECOND) SPOT RETURNS")
    print("─" * 70)

    returns_60s = []
    for i in range(12, n):
        if session_mask[i] and session_mask[i - 12] and day_id[i] == day_id[i - 12]:
            returns_60s.append(close[i] - close[i - 12])
    returns_60s = np.array(returns_60s)
    _print_return_stats(returns_60s, "60-second")

    # Also compute 24-bar (120-second) returns for wider hold times
    print("\n" + "─" * 70)
    print("2b. DISTRIBUTION OF 24-BAR (120-SECOND) SPOT RETURNS")
    print("─" * 70)

    returns_120s = []
    for i in range(24, n):
        if session_mask[i] and session_mask[i - 24] and day_id[i] == day_id[i - 24]:
            returns_120s.append(close[i] - close[i - 24])
    returns_120s = np.array(returns_120s)
    _print_return_stats(returns_120s, "120-second")

    # ── 3. Cost analysis & breakeven ──
    print("\n" + "─" * 70)
    print("3. COST ANALYSIS & BREAKEVEN THRESHOLDS")
    print("─" * 70)

    min_pts = min_points_to_breakeven(lot_size, atm_premium, 1, spread_per_side)
    min_spot_move = min_pts / delta
    print(f"  Min option premium move to breakeven: {min_pts:.2f} pts")
    print(f"  Min spot move at delta=0.5: {min_spot_move:.2f} pts")
    print(f"  (Spread: ₹{spread_per_side}/side, Lot: {lot_size})")

    # What fraction of moves are large enough?
    abs_30s = np.abs(returns_30s)
    abs_60s = np.abs(returns_60s)
    abs_120s = np.abs(returns_120s)

    print(f"\n  Fraction of moves exceeding breakeven spot threshold ({min_spot_move:.1f} pts):")
    print(f"    30s windows:  {np.mean(abs_30s > min_spot_move) * 100:.1f}%")
    print(f"    60s windows:  {np.mean(abs_60s > min_spot_move) * 100:.1f}%")
    print(f"    120s windows: {np.mean(abs_120s > min_spot_move) * 100:.1f}%")

    # ── 4. REALISTIC selective trading scenario ──
    print("\n" + "─" * 70)
    print("4. REALISTIC SELECTIVE TRADING (target > breakeven moves)")
    print("─" * 70)

    # Strategy: only trade when we predict a move > 8 spot points (4 option pts)
    # With stop at 3 option pts, target at 5 option pts
    for horizon_name, returns, horizon_bars in [
        ("30s (6 bars)", returns_30s, 6),
        ("60s (12 bars)", returns_60s, 12),
        ("120s (24 bars)", returns_120s, 24),
    ]:
        print(f"\n  --- Horizon: {horizon_name} ---")

        # If we trade moves > 8 spot pts = 4 option pts
        big_moves = returns[np.abs(returns) > 8]
        pct_big = len(big_moves) / len(returns) * 100

        # Among big moves, what's the avg size?
        if len(big_moves) > 0:
            winners = big_moves[big_moves > 0]
            losers = big_moves[big_moves < 0]
            avg_win_spot = np.mean(np.abs(winners)) if len(winners) > 0 else 0
            avg_loss_spot = np.mean(np.abs(losers)) if len(losers) > 0 else 0

            # With stop=3 option pts, target=5 option pts
            stop_opt = 3.0
            target_opt = 5.0

            # Winner: gain target_opt points
            w_costs = compute_trade_costs(atm_premium, atm_premium + target_opt,
                                           lot_size, 1, spread_per_side)
            net_win = target_opt * lot_size - w_costs["total_cost"]

            # Loser: lose stop_opt points
            l_costs = compute_trade_costs(atm_premium, atm_premium - stop_opt,
                                           lot_size, 1, spread_per_side)
            net_loss = -(stop_opt * lot_size) - l_costs["total_cost"]

            # Breakeven accuracy for this stop/target
            be_acc = (-net_loss) / (net_win - net_loss)

            print(f"    Moves > 8 spot pts: {pct_big:.1f}% of windows")
            print(f"    Avg |winner| spot: {avg_win_spot:.1f} pts, Avg |loser| spot: {avg_loss_spot:.1f} pts")
            print(f"    Target: {target_opt} opt pts → net win: ₹{net_win:.0f}")
            print(f"    Stop: {stop_opt} opt pts → net loss: ₹{net_loss:.0f}")
            print(f"    Breakeven accuracy: {be_acc * 100:.1f}%")

            # PnL at various accuracies
            for acc in [0.50, 0.52, 0.55, 0.58, 0.60]:
                epnl = acc * net_win + (1 - acc) * net_loss
                print(f"    {acc*100:.0f}% accuracy → E[PnL/trade] = ₹{epnl:.0f}")
        else:
            print(f"    No big moves found")

    # ── 5. Asymmetric stop/target analysis ──
    print("\n" + "─" * 70)
    print("5. STOP/TARGET COMBINATIONS — BREAKEVEN ACCURACY")
    print("─" * 70)

    print(f"  {'Stop':>6} {'Target':>8} {'Net Win':>10} {'Net Loss':>10} {'BE Acc':>8} {'55% E[PnL]':>12}")
    for stop in [3.0, 4.0, 5.0]:
        for target in [5.0, 7.0, 8.0, 10.0]:
            w_c = compute_trade_costs(atm_premium, atm_premium + target,
                                       lot_size, 1, spread_per_side)
            l_c = compute_trade_costs(atm_premium, atm_premium - stop,
                                       lot_size, 1, spread_per_side)
            nw = target * lot_size - w_c["total_cost"]
            nl = -(stop * lot_size) - l_c["total_cost"]
            be = (-nl) / (nw - nl) if (nw - nl) != 0 else 1.0
            epnl_55 = 0.55 * nw + 0.45 * nl
            print(f"  {stop:>5.0f}p {target:>7.0f}p {nw:>10.0f} {nl:>10.0f} {be*100:>7.1f}% {epnl_55:>11.0f}")

    # ── 6. Perfect prediction ceiling (selective) ──
    print("\n" + "─" * 70)
    print("6. PERFECT PREDICTION CEILING (selective, big moves only)")
    print("─" * 70)

    # If we perfectly predict direction AND only trade when |move| > 10 spot pts
    # Target = actual move × delta, stop never hit
    for threshold in [8, 10, 15]:
        big = returns_60s[np.abs(returns_60s) > threshold]
        if len(big) == 0:
            continue
        avg_move_opt = np.mean(np.abs(big)) * delta
        costs = compute_trade_costs(atm_premium, atm_premium + avg_move_opt,
                                     lot_size, 1, spread_per_side)
        net_per = avg_move_opt * lot_size - costs["total_cost"]
        # How many such non-overlapping opportunities per day?
        # Rough: fraction × bars_per_day / horizon_bars
        frac = len(big) / len(returns_60s)
        bars_per_day = np.sum(session_mask) / n_days
        opps_per_day = frac * bars_per_day / 12  # non-overlapping
        # But realistically, max 20-50 trades/day
        realistic_trades = min(opps_per_day, 50)
        daily_pnl = net_per * realistic_trades

        print(f"  |spot move| > {threshold} pts (60s horizon):")
        print(f"    Frequency: {frac*100:.1f}% of windows")
        print(f"    Avg |move| in option pts: {avg_move_opt:.1f}")
        print(f"    Net per trade: ₹{net_per:.0f}")
        print(f"    Opportunities/day: ~{opps_per_day:.0f} (capped at {realistic_trades:.0f})")
        print(f"    Perfect daily PnL: ₹{daily_pnl:,.0f}")

    # ── Summary ──
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print(f"""
  KEY FINDINGS:
  1. Median 30s spot move = 4 pts → 2 option pts. Too small for scalping every window.
  2. Breakeven cost ≈ {min_pts:.1f} option pts = {min_spot_move:.1f} spot pts per trade.
  3. 40% of 30s windows, 57% of 60s windows have |move| > {min_spot_move:.0f} pts.
  4. With stop=3/target=5 option pts, breakeven accuracy is ~40-45%.
  5. At 55% accuracy with selective entry, strategies ARE profitable.

  STRATEGY DESIGN IMPLICATIONS:
  - Don't trade every bar. Be SELECTIVE — only when expecting moves > 8-10 spot pts.
  - Use 60-120s hold times to capture larger moves (60s P75=+5.9, P90=+12.5 pts).
  - Stop: 3-5 option pts. Target: 5-10 option pts. The asymmetry creates edge.
  - Aim for 10-30 trades/day per strategy, not 100+.
  - The opportunity EXISTS if we can predict direction > 45-50% on our selected entries.
""")


def _print_return_stats(returns, label):
    abs_returns = np.abs(returns)
    print(f"  Sample size: {len(returns):,} overlapping windows")
    print(f"  Mean:   {np.mean(returns):+.2f} pts")
    print(f"  Median: {np.median(returns):+.2f} pts")
    print(f"  Std:    {np.std(returns):.2f} pts")
    print(f"  P10:    {np.percentile(returns, 10):+.2f} pts")
    print(f"  P25:    {np.percentile(returns, 25):+.2f} pts")
    print(f"  P75:    {np.percentile(returns, 75):+.2f} pts")
    print(f"  P90:    {np.percentile(returns, 90):+.2f} pts")
    print(f"  Min:    {np.min(returns):+.2f} pts")
    print(f"  Max:    {np.max(returns):+.2f} pts")
    print(f"\n  |move| > 5 pts:  {np.sum(abs_returns > 5) / len(returns) * 100:.1f}%")
    print(f"  |move| > 10 pts: {np.sum(abs_returns > 10) / len(returns) * 100:.1f}%")
    print(f"  |move| > 15 pts: {np.sum(abs_returns > 15) / len(returns) * 100:.1f}%")
    print(f"  |move| > 20 pts: {np.sum(abs_returns > 20) / len(returns) * 100:.1f}%")


if __name__ == "__main__":
    main()
