"""Opening Range Breakout (15-minute) — NIFTY 5-second options strategy.

Mechanism:
    NIFTY's first 15 minutes (09:15-09:30) form the overnight information absorption
    range. When price breaks above (or below) this range with a volume surge after
    09:30, it signals institutional commitment — a second wave of directional flow
    from momentum algos and VWAP buyers. 2-bar (10s) persistence filters out
    single-bar spikes common at the NIFTY open.

Converted from: trading_strategies/unique_strategies_all/Strategy_340.json
Original: 15-min ORB on top-20 FNO gappers, 1-min bars, hold 30-120 min.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean with warmup bars returning 1.0 to avoid division issues."""
    result = np.ones(len(arr), dtype=np.float64)
    for i in range(window, len(arr)):
        m = np.mean(arr[i - window : i])
        result[i] = m if m > 0.0 else 1.0
    return result


class Strategy(BaseStrategy):
    name = "opening_range_breakout_15"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — entry only after range fully forms
    session_end_minutes = 660     # 11:00 IST — ORB thesis weakens after mid-morning
    max_trades_per_day = 5
    max_lookback = 180            # 15 min warmup (180 × 5s = 900s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("volume_ratio_threshold", 1.5, 1.0, 3.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Extract spot arrays (forward-fill nulls before to_numpy) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .fill_null(0)
            .cast(pl.Float64)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        vol_ratio_thresh = params.get("volume_ratio_threshold", 1.5)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # --- Compute ORB high/low per day (09:15-09:30 = time_minutes 555-569) ---
        # Fixed time window — 180 bars at 5s = exactly 15 calendar minutes.
        orb_high = np.full(n, np.nan)
        orb_low = np.full(n, np.nan)

        for d in np.unique(day_id):
            orb_mask = (day_id == d) & (time_min >= 555) & (time_min < 570)
            if orb_mask.any():
                dh = float(np.nanmax(high[orb_mask]))
                dl = float(np.nanmin(low[orb_mask]))
                day_mask = day_id == d
                orb_high[day_mask] = dh
                orb_low[day_mask] = dl

        valid_orb = ~np.isnan(orb_high) & ~np.isnan(orb_low)

        # --- Volume ratio: current bar vs 60-bar rolling mean (5-minute baseline) ---
        # 60 bars = 5 minutes; measures surge relative to recent activity, not whole morning.
        vol_mean_60 = _rolling_mean(volume, 60)
        vol_ratio = np.where(vol_mean_60 > 0.0, volume / vol_mean_60, 0.0)

        # --- 2-bar (10-second) persistence: prev bar must also have broken out ---
        close_prev = np.empty(n, dtype=np.float64)
        close_prev[0] = close[0]
        close_prev[1:] = close[:-1]

        # --- Session filter: 09:30-11:00 IST only ---
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # Bullish: close above ORB high for 2 consecutive bars + volume surge
        buy_ce = (
            in_session
            & valid_orb
            & (close > orb_high)
            & (close_prev > orb_high)
            & (vol_ratio >= vol_ratio_thresh)
        )

        # Bearish: close below ORB low for 2 consecutive bars + volume surge
        buy_pe = (
            in_session
            & valid_orb
            & (close < orb_low)
            & (close_prev < orb_low)
            & (vol_ratio >= vol_ratio_thresh)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
