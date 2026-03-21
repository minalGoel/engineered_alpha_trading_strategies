"""closing_auction_anticipation_v1 — NIFTY closing-window momentum

Entry window: 15:00–15:20 IST only.
Mechanism: At 15:00, index fund NAV rebalancing and institutional TWAP closers
push NIFTY in the direction of the prior 30-minute trend. We enter at the first
bar of 15:00 when the 30-min trend, VWAP alignment, and volume acceleration
all confirm, and exit within 2 minutes capturing the initial closing flow burst.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "closing_auction_anticipation_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 — strategy is inactive until 15:00 but data needed for VWAP
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 3
    max_lookback = 360            # 30 min warmup (360 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("trend_threshold", 0.0015, 0.0005, 0.004),
            TunableParam("volume_ratio_threshold", 1.3, 1.0, 2.5),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        trend_threshold = params.get("trend_threshold", 0.0015)
        vol_ratio_threshold = params.get("volume_ratio_threshold", 1.3)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── 30-min return (360 × 5s = 30 min) ──────────────────────────────
        # Same time window as original — FIXED at 30 minutes, not 12x-scaled.
        return_360 = np.zeros(n, dtype=np.float64)
        for i in range(360, n):
            base = close[i - 360]
            if base > 0:
                return_360[i] = (close[i] - base) / base

        # ── Session VWAP (cumulative, resets each day) ──────────────────────
        vwap = np.zeros(n, dtype=np.float64)
        cum_vol = 0.0
        cum_pv = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_vol = 0.0
                cum_pv = 0.0
                prev_day = day_id[i]
            v = volume[i]
            cum_vol += v
            cum_pv += close[i] * v
            vwap[i] = cum_pv / cum_vol if cum_vol > 0 else close[i]

        # ── Volume ratio: last 30s (6 bars) vs last 5 min (60 bars) ────────
        vol_ratio = np.ones(n, dtype=np.float64)
        for i in range(60, n):
            long_avg = np.mean(volume[i - 60:i])
            if long_avg > 0:
                short_avg = np.mean(volume[max(i - 6, 0):i])
                vol_ratio[i] = short_avg / long_avg

        # ── Entry window: 15:00–15:20 IST ───────────────────────────────────
        entry_window = (time_min >= 900) & (time_min < 920)

        # ── Signal generation ────────────────────────────────────────────────
        above_vwap = close > vwap
        below_vwap = close < vwap
        vol_ok = vol_ratio > vol_ratio_threshold

        buy_ce = (
            entry_window
            & (return_360 > trend_threshold)
            & above_vwap
            & vol_ok
        )
        buy_pe = (
            entry_window
            & (return_360 < -trend_threshold)
            & below_vwap
            & vol_ok
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts, dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
