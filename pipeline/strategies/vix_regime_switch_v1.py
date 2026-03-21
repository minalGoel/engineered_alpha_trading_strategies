"""
vix_regime_switch_v1 — VIX Regime Switch Strategy for NIFTY 5-second options.

Thesis: When India VIX is elevated above its 60-minute session rolling mean (fear regime)
and VIX begins declining (put-unwinding by hedgers), NIFTY bounces 30-50 points in the
next 30-90 seconds. Conversely, when VIX is depressed and starts rising (fresh hedging
demand), NIFTY falls. VIX re-prices faster than spot, giving a 5-10 second lead.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vix_regime_switch_v1"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — skip first 15 min for VIX stabilisation
    session_end_minutes = 920      # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 720             # 60-min warmup for rolling VIX z-score

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_zscore_high", 1.5, 1.0, 2.5),   # elevated VIX threshold
            TunableParam("vix_zscore_low", 1.0, 0.5, 2.0),    # depressed VIX threshold
            TunableParam("vix_slope_thresh", 0.03, 0.01, 0.10),  # VIX turning speed
            TunableParam("stop_pts", 3.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 14.0),
        ]

    def compute(self, spot_df: pl.DataFrame, option_df: pl.DataFrame,
                vix_df: pl.DataFrame, params: dict) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        vix_zscore_high = params.get("vix_zscore_high", 1.5)
        vix_zscore_low = params.get("vix_zscore_low", 1.0)
        vix_slope_thresh = params.get("vix_slope_thresh", 0.03)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Align VIX to spot timestamps (backward join) ─────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select([
                    "datetime",
                    pl.col("close").alias("vix_close")
                ]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Rolling 720-bar (60-min) VIX z-score ─────────────────────────────
        # Captures intraday session VIX baseline; not contaminated by prior day levels.
        window = 720
        vix_mean = np.full(n, 15.0)
        vix_std = np.ones(n)
        for i in range(window, n):
            w = vix_close[i - window:i]
            m = np.mean(w)
            s = np.std(w)
            vix_mean[i] = m
            vix_std[i] = s if s > 0.01 else 0.01

        vix_zscore = (vix_close - vix_mean) / vix_std

        # ── VIX slope over 12 bars (60s) ──────────────────────────────────────
        # Minimum reliable slope estimate at 5s resolution; shorter is too noisy.
        vix_slope = np.zeros(n)
        for i in range(12, n):
            base = vix_close[i - 12]
            if base > 0.1:
                vix_slope[i] = (vix_close[i] - base) / base

        # ── NIFTY 30-second return (6 bars) for directional guard ─────────────
        ret_6 = np.zeros(n)
        for i in range(6, n):
            base = close[i - 6]
            if base > 0:
                ret_6[i] = (close[i] - base) / base

        # ── Session and warmup masks ──────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed_up = np.arange(n) >= window

        # ── Signal generation ─────────────────────────────────────────────────
        # buy_ce: VIX elevated + turning down → hedgers unwinding puts → NIFTY bounces
        buy_ce = (
            in_session
            & warmed_up
            & (vix_zscore > vix_zscore_high)
            & (vix_slope < -vix_slope_thresh)
            & (ret_6 > -0.002)     # guard: NIFTY not already in freefall (genuine crash)
        )

        # buy_pe: VIX depressed + starting to rise → hedging demand → NIFTY declines
        buy_pe = (
            in_session
            & warmed_up
            & (vix_zscore < -vix_zscore_low)
            & (vix_slope > vix_slope_thresh)
            & (ret_6 < 0.002)      # guard: NIFTY not already in melt-up
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
