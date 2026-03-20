"""VWAP Z-Score Bounce — vwap_zscore_bounce_v1

Converted from: trading_strategies/unique_strategies_all/Strategy_140.json
Original: VWAP z-score mean reversion on NIFTY 100 stocks, 1-min bars, 10-45 min holds.

Mechanism:
When NIFTY's VWAP z-score crosses ±2.0 sigma, TWAP/VWAP-benchmarked institutional
algorithms start getting significantly worse fills vs. their benchmark. This creates
mechanical reversal: at z < -2.0, TWAP buy orders accelerate; at z > +2.0, sell orders
accelerate. At 5s resolution we enter the moment the z-score turns back (zscore_delta3),
capturing the first 12-15 spot points of reversion.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_zscore_bounce_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — 15 min after open for VWAP stabilization
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 360            # 30 min warmup (360 bars × 5s) for rolling std

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 2.0, 1.5, 3.0),
            TunableParam("vix_max", 24.0, 18.0, 30.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Extract arrays (forward-fill before numpy conversion) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # --- Parameters ---
        zscore_threshold = float(params.get("zscore_threshold", 2.0))
        vix_max = float(params.get("vix_max", 24.0))
        stop_pts = float(params.get("stop_pts", 4.0))
        target_pts = float(params.get("target_pts", 6.0))

        # --- VIX (aligned to spot bars, forward-fill) ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy().astype(float)

        # --- Session VWAP (cumulative, resets per day) ---
        vwap = np.zeros(n)
        cum_pv = 0.0
        cum_vol = 0.0
        cur_day = -1
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                cum_pv = 0.0
                cum_vol = 0.0
            v = volume[i] if volume[i] > 0 else 1.0
            cum_pv += close[i] * v
            cum_vol += v
            vwap[i] = cum_pv / cum_vol

        # --- VWAP deviation ---
        vwap_dev = close - vwap

        # --- Rolling 30-min std of vwap_dev (capped at day start, min 10 bars) ---
        # Same 30-minute window as original's STDEV(vwap_dev, 30 on 1-min bars).
        # Floor at 0.5 pts to avoid near-zero division in low-vol pre-open data.
        vwap_dev_std = np.ones(n)
        day_start_idx = 0
        cur_day = -1
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                day_start_idx = i
            lo = max(day_start_idx, i - 359)  # 360-bar window, inclusive of i
            if i - lo >= 9:                    # need at least 10 samples
                s = float(np.std(vwap_dev[lo : i + 1]))
                vwap_dev_std[i] = max(s, 0.5)

        # --- VWAP z-score ---
        vwap_zscore = vwap_dev / vwap_dev_std

        # --- Z-score 15-second rate of change (3 bars × 5s) ---
        # Confirms the z-score has turned from its extreme before we enter.
        zscore_delta3 = np.zeros(n)
        zscore_delta3[3:] = vwap_zscore[3:] - vwap_zscore[:-3]

        # --- Session filter ---
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # --- VIX filter ---
        low_vix = vix_close < vix_max

        # --- Entry signals ---
        # buy_ce: z-score below -threshold AND turning upward (TWAP buy flow reversal)
        buy_ce = (
            in_session
            & low_vix
            & (vwap_zscore < -zscore_threshold)
            & (zscore_delta3 > 0.0)
        )

        # buy_pe: z-score above +threshold AND turning downward (TWAP sell flow reversal)
        buy_pe = (
            in_session
            & low_vix
            & (vwap_zscore > zscore_threshold)
            & (zscore_delta3 < 0.0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,              # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
