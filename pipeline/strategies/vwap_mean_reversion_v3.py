"""vwap_mean_reversion_v3 — Extreme VWAP Deviation Reversion with Volume Confirmation

Targets rare capitulation events on NIFTY where the index deviates >2.7 sigma from
session VWAP accompanied by a volume surge. At this extreme, VWAP-benchmarked institutions,
gamma-hedging dealers, and stat-arb desks all converge as buyers simultaneously, producing
a sharp 20-40 spot point snap-back within 30-90 seconds.

Differentiated from v18 (z<-1.7, dev_momentum) and v19 (moderate 20-35pt, microstructure basing)
by: deeper threshold, explicit volume-surge confirmation, and multi-buyer-convergence mechanism.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_v3"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — 15 min warmup after open
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 360            # 30 min warmup (360 × 5s) for stddev to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 2.7, 2.0, 3.5),
            TunableParam("rel_vol_threshold", 1.7, 1.2, 2.5),
            TunableParam("stop_pts", 7.0, 4.0, 12.0),
            TunableParam("target_pts", 12.0, 7.0, 20.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract columns (forward-fill in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        zscore_thresh = params.get("zscore_threshold", 2.7)
        rel_vol_thresh = params.get("rel_vol_threshold", 1.7)
        stop_pts = params.get("stop_pts", 7.0)
        target_pts = params.get("target_pts", 12.0)

        # ── VWAP: cumulative typical_price * volume, reset per day ──
        typical_price = (high + low + close) / 3.0
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            vol_i = max(volume[i], 1.0)
            cum_tp_vol += typical_price[i] * vol_i
            cum_vol += vol_i
            vwap[i] = cum_tp_vol / cum_vol

        # ── Rolling 30-min stddev (360 bars) of close for z-score denominator ──
        LOOKBACK_STD = 360
        vwap_std = np.zeros(n)
        for i in range(LOOKBACK_STD, n):
            vwap_std[i] = np.std(close[i - LOOKBACK_STD:i])

        # ── VWAP z-score ──
        vwap_zscore = np.zeros(n)
        valid_std = vwap_std > 0
        vwap_zscore[valid_std] = (close[valid_std] - vwap[valid_std]) / vwap_std[valid_std]

        # ── Relative volume: current bar vs. 2-min (24 bar) rolling mean ──
        LOOKBACK_VOL = 24
        rel_vol = np.ones(n)
        for i in range(LOOKBACK_VOL, n):
            avg_vol = np.mean(volume[i - LOOKBACK_VOL:i])
            if avg_vol > 0:
                rel_vol[i] = volume[i] / avg_vol

        # ── Session and warmup filters ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed_up = np.zeros(n, dtype=bool)
        warmed_up[LOOKBACK_STD:] = True

        active = in_session & warmed_up

        # ── Signals ──
        # Bullish: extreme oversold with panic volume → reversion up → buy CE
        buy_ce = active & (vwap_zscore < -zscore_thresh) & (rel_vol > rel_vol_thresh)

        # Bearish: extreme overbought with panic volume → reversion down → buy PE
        buy_pe = active & (vwap_zscore > zscore_thresh) & (rel_vol > rel_vol_thresh)

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
