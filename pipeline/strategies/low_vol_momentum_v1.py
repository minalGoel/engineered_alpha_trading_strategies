"""low_vol_momentum_v1 — Low-Volatility Regime Momentum on NIFTY

When NIFTY's 5-minute realized vol is below 80% of its 20-minute baseline,
institutional TWAP/VWAP algorithms dominate the order book. EMA alignment
across 1-min and 5-min windows then reflects sustained directional absorption,
not noise. Enter in the EMA direction, capture 30-90 second continuation.

Adapted from cross-sectional low-vol anomaly (Strategy_215.json):
the stock-universe vol ranking is replaced by a NIFTY time-series vol ratio.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (in-place loop, no pandas dependency)."""
    out = np.empty(len(arr))
    k = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation — returns 0.0 for bars before warmup."""
    out = np.zeros(len(arr))
    for i in range(window, len(arr)):
        out[i] = np.std(arr[i - window:i])
    return out


class Strategy(BaseStrategy):
    name = "low_vol_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min of session noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 240            # 20 min warmup for rvol_240 baseline

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_ratio_threshold", 0.80, 0.50, 0.95),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        vol_ratio_threshold = params.get("vol_ratio_threshold", 0.80)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # Log returns (5-second bars)
        log_ret = np.zeros(n)
        safe_prev = np.where(close[:-1] > 0, close[:-1], 1.0)
        log_ret[1:] = np.log(close[1:] / safe_prev)

        # Short-term realized vol: 60 bars = 5 minutes
        rvol_60 = _rolling_std(log_ret, 60)

        # Baseline realized vol: 240 bars = 20 minutes
        rvol_240 = _rolling_std(log_ret, 240)

        # Vol ratio: current / baseline (clamp division by zero to 1.0)
        vol_ratio = np.where(rvol_240 > 1e-10, rvol_60 / rvol_240, 1.0)

        # EMA alignment: 1-min fast (12 bars) vs 5-min slow (60 bars)
        ema_12 = _ema(close, 12)
        ema_60 = _ema(close, 60)

        # Single-bar return: directional confirmation
        ret_1 = np.zeros(n)
        safe_prev2 = np.where(close[:-1] > 0, close[:-1], 1.0)
        ret_1[1:] = (close[1:] - close[:-1]) / safe_prev2

        # Session and warmup filters
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed_up = np.arange(n) >= 240  # need full rvol_240 window

        # Low-vol regime: current 5-min vol is below threshold of 20-min baseline
        low_vol_regime = vol_ratio < vol_ratio_threshold

        # Bullish: low-vol regime + EMA up + confirming bar up
        buy_ce = in_session & warmed_up & low_vol_regime & (ema_12 > ema_60) & (ret_1 > 0)

        # Bearish: low-vol regime + EMA down + confirming bar down
        buy_pe = in_session & warmed_up & low_vol_regime & (ema_12 < ema_60) & (ret_1 < 0)

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
