"""
quantile_regression_v1 — Composite distribution-shift signal on NIFTY.

Inspired by quantile regression (Koenker & Bassett 1978): instead of estimating the mean
return, estimate whether the FULL conditional return distribution is shifted in one direction.
Original applied QR on individual stocks; here we apply the same three predictors (VWAP
deviation, volume ratio, 1-min return) to NIFTY index itself, approximating the QR_P10 > 0
condition via rolling composite percentile ranking.

Trade buy_ce when all three predictors simultaneously hit extreme bullish territory
(composite score in top pct_threshold% of recent 20-min distribution).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "quantile_regression_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — 15 min for VWAP and indicator warmup
    session_end_minutes = 925     # 15:25 IST
    max_lookback = 240            # 20 min warmup (240 × 5s = 1200s)
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("composite_pct_threshold", 85.0, 75.0, 95.0),
            TunableParam("vwap_dist_threshold",      5.0,  2.0, 15.0),
            TunableParam("vol_ratio_threshold",      1.3,  1.1,  2.0),
            TunableParam("stop_pts",                 4.0,  2.0,  8.0),
            TunableParam("target_pts",               6.0,  4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays, forward-filling NaN ──────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").fill_null(0.0).to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ───────────────────────────────────────────────────────
        pct_threshold   = float(params.get("composite_pct_threshold", 85.0))
        vwap_dist_thresh = float(params.get("vwap_dist_threshold", 5.0))
        vol_ratio_thresh = float(params.get("vol_ratio_threshold", 1.3))
        stop_pts        = float(params.get("stop_pts", 4.0))
        target_pts      = float(params.get("target_pts", 6.0))

        # ── Session VWAP (cumulative from each day's open) ───────────────────
        vwap = np.zeros(n)
        cum_pv  = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_pv  = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            cum_pv  += close[i] * volume[i]
            cum_vol += volume[i]
            vwap[i] = cum_pv / cum_vol if cum_vol > 0.0 else close[i]

        # VWAP distance in bps
        vwap_dist = np.zeros(n)
        nz = vwap > 0.0
        vwap_dist[nz] = (close[nz] - vwap[nz]) / vwap[nz] * 10000.0

        # ── Volume ratio: current bar vs 5-min (60-bar) rolling mean ─────────
        vol_period = 60
        vol_ratio = np.ones(n)
        for i in range(vol_period, n):
            avg_vol = np.mean(volume[i - vol_period:i])
            if avg_vol > 0.0:
                vol_ratio[i] = volume[i] / avg_vol

        # ── 1-minute (12-bar) return in bps ──────────────────────────────────
        ret_12 = np.zeros(n)
        for i in range(12, n):
            if close[i - 12] > 0.0:
                ret_12[i] = (close[i] - close[i - 12]) / close[i - 12] * 10000.0

        # ── Composite z-score: 0.5 * z(vwap_dist) + 0.5 * z(ret_12) ─────────
        norm_period = 120  # 10 min normalization window
        comp_score = np.zeros(n)
        for i in range(norm_period, n):
            vd_std = np.std(vwap_dist[i - norm_period:i])
            vd_std = vd_std if vd_std > 0.1 else 0.1
            r_std  = np.std(ret_12[i - norm_period:i])
            r_std  = r_std  if r_std  > 0.1 else 0.1
            comp_score[i] = (
                (vwap_dist[i] / vd_std) * 0.5
                + (ret_12[i]  / r_std)  * 0.5
            )

        # ── Rolling percentile thresholds over 20-min window (240 bars) ──────
        pct_period = 240
        score_high = np.zeros(n)   # bullish extreme (upper tail)
        score_low  = np.zeros(n)   # bearish extreme (lower tail)
        for i in range(pct_period, n):
            window     = comp_score[i - pct_period:i]
            score_high[i] = np.percentile(window, pct_threshold)
            score_low[i]  = np.percentile(window, 100.0 - pct_threshold)

        # ── Session / warmup mask ─────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed_up  = np.arange(n) >= pct_period

        # ── Entry signals ─────────────────────────────────────────────────────
        # Buy CE: composite score in top tier AND VWAP above AND volume surge AND momentum up
        buy_ce = (
            in_session
            & warmed_up
            & (comp_score > score_high)
            & (vwap_dist  > vwap_dist_thresh)
            & (vol_ratio  > vol_ratio_thresh)
            & (ret_12     > 0.0)
        )

        # Buy PE: composite score in bottom tier AND VWAP below AND volume surge AND momentum down
        buy_pe = (
            in_session
            & warmed_up
            & (comp_score < score_low)
            & (vwap_dist  < -vwap_dist_thresh)
            & (vol_ratio  > vol_ratio_thresh)
            & (ret_12     < 0.0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,              # 60 seconds — tight hold for high-confidence signals
            max_trades_per_day=self.max_trades_per_day,
        )
