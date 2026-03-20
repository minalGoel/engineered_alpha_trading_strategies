"""roc_momentum_v1 — Rate-of-Change Z-Score Momentum on NIFTY

Mechanism:
    Institutional TWAP/VWAP algorithms executing large directional orders compress
    their flow into 1-2 minute windows, creating 1-minute ROC readings that rank in
    the top/bottom decile of NIFTY's own intraday ROC distribution. These multi-
    participant flow bursts (FII, prop, index arb) take 30-90 additional seconds to
    absorb. We detect them by z-scoring the current 1-min ROC against a 20-minute
    rolling baseline (adapts to session regime), entering only when NIFTY is on the
    correct side of session VWAP with elevated volume.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Z-score of arr[i] relative to arr[i-window:i] for each i."""
    n = len(arr)
    z = np.zeros(n)
    for i in range(window, n):
        w = arr[i - window:i]
        std = np.std(w)
        if std > 1e-10:
            z[i] = (arr[i] - np.mean(w)) / std
    return z


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean of arr[i-window:i] (excludes current bar)."""
    n = len(arr)
    out = np.zeros(n)
    for i in range(window, n):
        out[i] = np.mean(arr[i - window:i])
    return out


class Strategy(BaseStrategy):
    name = "roc_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min of price discovery
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 260            # 240 (z-score window) + 12 (roc) + 8 buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 1.8, 1.0, 3.0),
            TunableParam("min_roc_pct", 0.05, 0.02, 0.15),
            TunableParam("volume_ratio_min", 1.5, 1.0, 3.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # --- Extract arrays (forward-fill nulls before numpy conversion) ---
        close = (
            spot_df.select(pl.col("close").fill_nan(None).fill_null(strategy="forward"))
            .to_series()
            .to_numpy()
        )
        volume = (
            spot_df.select(pl.col("volume").fill_null(0))
            .to_series()
            .to_numpy()
            .astype(float)
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        zscore_thresh = params.get("zscore_threshold", 1.8)
        min_roc_pct = params.get("min_roc_pct", 0.05)
        vol_ratio_min = params.get("volume_ratio_min", 1.5)

        # --- 1-minute ROC (12 bars × 5s = 60s) ---
        roc_12 = np.zeros(n)
        for i in range(12, n):
            base = close[i - 12]
            if base > 0.0:
                roc_12[i] = (close[i] - base) / base * 100.0

        # --- Rolling 240-bar z-score of roc_12 (20-min normalization window) ---
        roc_zscore = _rolling_zscore(roc_12, 240)

        # --- Session VWAP per day (cumulative from first bar of each day) ---
        vwap = np.zeros(n)
        cum_cv = 0.0
        cum_vol = 0.0
        cur_day = -1
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                cum_cv = 0.0
                cum_vol = 0.0
            cum_cv += close[i] * volume[i]
            cum_vol += volume[i]
            vwap[i] = cum_cv / cum_vol if cum_vol > 0.0 else close[i]

        # --- Volume ratio: current vs 60-bar (5-min) rolling mean ---
        vol_sma_60 = _rolling_mean(volume, 60)
        vol_ratio = np.where(vol_sma_60 > 0.0, volume / vol_sma_60, 0.0)

        # --- Session filter ---
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # --- Entry signals ---
        # Bullish: extreme positive ROC burst above VWAP with volume surge
        buy_ce = (
            in_session
            & (roc_zscore > zscore_thresh)
            & (roc_12 > min_roc_pct)
            & (vol_ratio > vol_ratio_min)
            & (close > vwap)
        )

        # Bearish: extreme negative ROC burst below VWAP with volume surge
        buy_pe = (
            in_session
            & (roc_zscore < -zscore_thresh)
            & (roc_12 < -min_roc_pct)
            & (vol_ratio > vol_ratio_min)
            & (close < vwap)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),
            target_points=np.full(n, 6.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,        # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
