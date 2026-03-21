"""ORB Failure Trade — NIFTY 5-second index options.

When NIFTY breaks the 15-minute ORB boundary (09:15-09:30) then closes back
inside the range within 60 seconds, trapped breakout participants cut losses
and market-maker hedge reversals drive a fast 15-25 spot point reversal.
Enter ATM options on the failure bar; hold up to 120 seconds.
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "orb_failure_trade_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — entries only after ORB forms
    session_end_minutes = 870     # 14:30 IST — ORB failures later are weak
    max_trades_per_day = 3
    max_lookback = 240            # 20-min warmup covers full ORB formation window

    def tunable_params(self) -> list[TunableParam]:
        return [
            # How quickly the failure must occur (bars since breakout bar)
            TunableParam("failure_bars", 12.0, 6.0, 24.0),
            # RSI(12) threshold for failed-down entry (buy CE): momentum turning up
            TunableParam("rsi_low", 45.0, 35.0, 55.0),
            # RSI(12) threshold for failed-up entry (buy PE): momentum turning down
            TunableParam("rsi_high", 55.0, 45.0, 65.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Extract arrays (forward-fill NaN in Polars before numpy) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # --- Parameters ---
        failure_bars = int(params.get("failure_bars", 12))
        rsi_low = params.get("rsi_low", 45.0)
        rsi_high = params.get("rsi_high", 55.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # --- VIX filter (skip if VIX < 13 or > 22) ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- RSI(12) — 1-minute RSI at 5-second resolution ---
        rsi = _compute_rsi(close, 12)
        # Replace leading NaN with neutral value 50 (no signal)
        nan_mask = np.isnan(rsi)
        rsi[nan_mask] = 50.0

        # --- Compute per-day ORB high/low (09:15-09:30, time_min [555, 570)) ---
        orb_high = np.full(n, np.nan)
        orb_low = np.full(n, np.nan)

        unique_days = np.unique(day_id)
        for d in unique_days:
            day_mask = day_id == d
            day_idx = np.where(day_mask)[0]
            orb_mask = day_mask & (time_min >= 555) & (time_min < 570)
            orb_idx = np.where(orb_mask)[0]
            if len(orb_idx) == 0:
                continue
            orb_h = float(np.max(high[orb_idx]))
            orb_l = float(np.min(low[orb_idx]))
            orb_high[day_idx] = orb_h
            orb_low[day_idx] = orb_l

        # Forward-fill ORB values across the session
        for i in range(1, n):
            if np.isnan(orb_high[i]):
                orb_high[i] = orb_high[i - 1]
            if np.isnan(orb_low[i]):
                orb_low[i] = orb_low[i - 1]

        # --- Signal arrays ---
        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        # --- State machine: detect breakout → failure → entry ---
        # breakout_type: 0=none, 1=upward (close > orb_high), -1=downward (close < orb_low)
        current_day = -1
        breakout_type = 0
        bars_since_breakout = 0
        trades_today = 0

        for i in range(n):
            d = int(day_id[i])
            t = int(time_min[i])

            # Day boundary: reset state
            if d != current_day:
                current_day = d
                breakout_type = 0
                bars_since_breakout = 0
                trades_today = 0

            # Only look for entries in 09:30-14:30 window
            if t < 570 or t >= 870:
                continue

            # Trade cap
            if trades_today >= self.max_trades_per_day:
                continue

            # Need valid ORB levels
            if np.isnan(orb_high[i]) or np.isnan(orb_low[i]):
                continue

            # VIX filter
            v = vix_close[i]
            if v < 13.0 or v > 22.0:
                # Reset breakout state when VIX is outside range (skip period)
                breakout_type = 0
                bars_since_breakout = 0
                continue

            orb_h = orb_high[i]
            orb_l = orb_low[i]

            if breakout_type == 0:
                # Detect a fresh breakout
                if close[i] > orb_h:
                    breakout_type = 1
                    bars_since_breakout = 0
                elif close[i] < orb_l:
                    breakout_type = -1
                    bars_since_breakout = 0
            else:
                bars_since_breakout += 1

                # Failure window expired — reset
                if bars_since_breakout > failure_bars:
                    breakout_type = 0
                    bars_since_breakout = 0
                    continue

                if breakout_type == 1:
                    # Upward breakout: failure = price returns below orb_high → buy PE
                    if close[i] < orb_h and rsi[i] < rsi_high:
                        buy_pe[i] = True
                        trades_today += 1
                        breakout_type = 0
                        bars_since_breakout = 0

                elif breakout_type == -1:
                    # Downward breakout: failure = price returns above orb_low → buy CE
                    if close[i] > orb_l and rsi[i] > rsi_low:
                        buy_ce[i] = True
                        trades_today += 1
                        breakout_type = 0
                        bars_since_breakout = 0

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI. Returns array with NaN for the first `period` bars."""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]

    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Seed with simple average over first `period` bars
    avg_gain = float(np.mean(gains[1: period + 1]))
    avg_loss = float(np.mean(losses[1: period + 1]))

    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi
