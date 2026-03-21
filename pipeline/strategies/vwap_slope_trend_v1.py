"""vwap_slope_trend_v1 — NIFTY VWAP slope institutional flow strategy.

Thesis: TWAP/VWAP execution algorithms benchmarked against session VWAP systematically
lift offers at progressively higher prices when executing large directional orders.
The 3-minute VWAP slope captures this institutional flow; when the 1-minute slope
exceeds the 3-minute slope (acceleration), the executing algorithm has remaining
order flow that will continue pushing NIFTY in the same direction for the next 60-90s.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _linreg_slope(arr: np.ndarray, period: int) -> np.ndarray:
    """Linear regression slope over a rolling window (in-place safe)."""
    n = len(arr)
    result = np.zeros(n)
    if period < 2 or n < period:
        return result
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = float(np.sum((x - x_mean) ** 2))
    if x_var == 0.0:
        return result
    for i in range(period - 1, n):
        y = arr[i - period + 1 : i + 1]
        y_mean = float(y.mean())
        result[i] = float(np.sum((x - x_mean) * (y - y_mean))) / x_var
    return result


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(period - 1, n):
        result[i] = float(np.mean(arr[i - period + 1 : i + 1]))
    return result


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (no day-boundary reset — contamination negligible at 36 bars)."""
    n = len(arr)
    result = np.zeros(n)
    if n == 0:
        return result
    alpha = 2.0 / (period + 1)
    result[0] = arr[0]
    for i in range(1, n):
        result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
    return result


def _compute_vwap(
    close: np.ndarray, volume: np.ndarray, day_id: np.ndarray
) -> np.ndarray:
    """Session VWAP resetting at each trading day start."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(n):
        v = float(volume[i]) if volume[i] > 0 else 1.0
        if i == 0 or day_id[i] != day_id[i - 1]:
            cum_pv = close[i] * v
            cum_v = v
        else:
            cum_pv += close[i] * v
            cum_v += v
        vwap[i] = cum_pv / cum_v
    return vwap


class Strategy(BaseStrategy):
    """NIFTY VWAP slope trend — buy CE/PE when institutional flow is accelerating."""

    name = "vwap_slope_trend_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — 5 min warmup for slope stability
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 120            # 10 min = 120 bars at 5s
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("slope_threshold", 0.08, 0.02, 0.40),
            TunableParam("stop_pts", 5.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Extract arrays with NaN handling
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        slope_threshold = float(params.get("slope_threshold", 0.08))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # --- Session VWAP (cumulative, resets each day) ---
        vwap = _compute_vwap(close, volume, day_id)

        # --- 3-minute VWAP slope: primary institutional flow direction signal ---
        # 36 bars × 5s = 3 minutes. Detects the currently-executing institutional order.
        vwap_slope_36 = _linreg_slope(vwap, 36)

        # --- 1-minute VWAP slope: acceleration detector ---
        # When slope_12 > slope_36 (bullish), institutional buying is ramping up,
        # not plateauing — the algorithm has remaining order flow.
        vwap_slope_12 = _linreg_slope(vwap, 12)

        # --- 90-second SMA of 3-min slope: ensures slope is building, not topping ---
        slope_ma_18 = _sma(vwap_slope_36, 18)

        # --- 3-minute EMA of close: short-term price trend confirmation ---
        ema_36 = _ema(close, 36)

        # --- Filters ---
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        # Need at least 36 bars for slope_36 to be valid
        warmed = np.arange(n) >= 36

        # --- Bullish entry: NIFTY trending up with accelerating institutional buying ---
        # 1. 3-min VWAP slope positive and meaningful
        # 2. 1-min slope > 3-min slope → TWAP algorithm increasing urgency (acceleration)
        # 3. Price above VWAP → strength relative to institutional benchmark
        # 4. Price above EMA_36 → confirmed short-term uptrend
        # 5. 3-min slope above its 90s MA → slope building, not decaying
        buy_ce = (
            in_session
            & warmed
            & (vwap_slope_36 > slope_threshold)
            & (vwap_slope_12 > vwap_slope_36)
            & (close > vwap)
            & (close > ema_36)
            & (vwap_slope_36 > slope_ma_18)
        )

        # --- Bearish entry: mirror conditions for institutional selling ---
        buy_pe = (
            in_session
            & warmed
            & (vwap_slope_36 < -slope_threshold)
            & (vwap_slope_12 < vwap_slope_36)
            & (close < vwap)
            & (close < ema_36)
            & (vwap_slope_36 < slope_ma_18)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
