"""
deviation_from_ema_v1 — EMA Rubber-Band Snap-Back on NIFTY (5-second bars)

Mechanism:
  When NIFTY deviates sharply from its 5-minute EMA (z-score > ±1.5 of recent
  deviations), VWAP-benchmarked algorithms and market makers absorb the overshoot.
  We enter on the first bar where the deviation z-score peaks and begins narrowing,
  capturing the 15-90 second snap-back toward the EMA.

Converted from: trading_strategies/unique_strategies_all/Strategy_153.json
Original: EMA(50) deviation on 1-min stock bars, hold 10-30 min.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average via standard recursive formula."""
    alpha = 2.0 / (period + 1)
    out = np.empty(len(arr), dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rolling_zscore(dev: np.ndarray, window: int, min_std: float = 0.5) -> np.ndarray:
    """
    Rolling z-score of dev[] over a trailing window.
    min_std prevents division by near-zero in flat sessions.
    """
    n = len(dev)
    out = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        w = dev[i - window:i]
        mu = np.mean(w)
        sigma = np.std(w)
        if sigma > min_std:
            out[i] = (dev[i] - mu) / sigma
        # else stays 0.0 (neutral — no signal in flat regimes)
    return out


class Strategy(BaseStrategy):
    name = "deviation_from_ema_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — 15-min warmup for EMA(60) + rolling std(120)
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 240            # 20-min warmup (120 bars EMA warm + 120 bars rolling std)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 1.5, 1.0, 2.5),
            TunableParam("stop_pts",         4.0, 2.0, 8.0),
            TunableParam("target_pts",       6.0, 3.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Parameters ──────────────────────────────────────────────────────────
        zscore_thresh = params.get("zscore_threshold", 1.5)
        stop_pts      = params.get("stop_pts",         4.0)
        target_pts    = params.get("target_pts",       6.0)

        # ── Spot close (forward-fill NaN before numpy) ───────────────────────────
        close    = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── EMA(60) = 5-minute rubber-band anchor ────────────────────────────────
        ema60 = _ema(close, 60)

        # ── Deviation from EMA ───────────────────────────────────────────────────
        dev = close - ema60

        # ── Rolling z-score of deviation (120-bar = 10-min window) ───────────────
        # Replaces ATR normalization: more robust on 5s index bars where ATR is noisy.
        dev_zscore = _rolling_zscore(dev, window=120, min_std=0.5)

        # ── Snap-back confirmation: deviation z-score peaked and is now narrowing ─
        # Use 2-bar lag to filter single-bar noise at 5s resolution.
        # Long snap: zscore was very negative, now less negative (moving toward 0)
        # Short snap: zscore was very positive, now less positive (moving toward 0)
        snapping_up   = np.zeros(n, dtype=bool)
        snapping_down = np.zeros(n, dtype=bool)
        if n > 2:
            snapping_up[2:]   = dev_zscore[2:] > dev_zscore[:-2]
            snapping_down[2:] = dev_zscore[2:] < dev_zscore[:-2]

        # ── VIX regime filter ────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        low_vix    = vix_close < 25.0
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry signals ────────────────────────────────────────────────────────
        # Bullish: NIFTY stretched well below 5-min EMA, snap-back just started → buy CE
        buy_ce = (
            in_session
            & low_vix
            & (dev_zscore < -zscore_thresh)
            & snapping_up
        )

        # Bearish: NIFTY stretched well above 5-min EMA, fade just started → buy PE
        buy_pe = (
            in_session
            & low_vix
            & (dev_zscore > zscore_thresh)
            & snapping_down
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
