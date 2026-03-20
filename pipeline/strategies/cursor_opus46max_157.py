"""Smart Beta Intraday v1 — cursor_opus46max_157

Thesis: Low-vol stocks that break out of compressed range (Keltner Channel)
sustain moves because breakouts are information-driven. OBV slope confirms
volume support.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_ema(arr, period):
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = arr[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_157"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("keltner_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("range_ratio_thresh", default=1.5, low=1.2, high=2.5),
            TunableParam("target_pct", default=0.006, low=0.003, high=0.008),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_stop_pct", default=0.002, low=0.001, high=0.003),
            TunableParam("trailing_activate_pct", default=0.0035, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        keltner_m = params.get("keltner_mult", 1.5)
        range_ratio_t = params.get("range_ratio_thresh", 1.5)
        target_pct = params.get("target_pct", 0.006)
        stop_pct = params.get("stop_loss_pct", 0.003)
        trail_pct = params.get("trailing_stop_pct", 0.002)
        trail_act = params.get("trailing_activate_pct", 0.0035)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ATR(14) and EMA(20) for Keltner
        atr = _compute_atr(high, low, close, 14)
        ema20 = _compute_ema(close, 20)
        keltner_upper = ema20 + keltner_m * atr
        keltner_lower = ema20 - keltner_m * atr

        # Range ratio: (high-low) / SMA(high-low, 20)
        bar_range = high - low
        avg_range = df.select(
            (pl.col("high") - pl.col("low")).rolling_mean(20).alias("_ar")
        )["_ar"].to_numpy().astype(np.float64)
        avg_range = np.nan_to_num(avg_range, nan=1e10)
        avg_range = np.clip(avg_range, 1e-10, None)
        range_ratio = bar_range / avg_range

        # OBV and its slope
        obv = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]

        # OBV slope via simple linear regression over 20 bars
        obv_slope = np.zeros(n, dtype=np.float64)
        for i in range(19, n):
            window = obv[i - 19:i + 1]
            x = np.arange(20, dtype=np.float64)
            x_mean = x.mean()
            y_mean = window.mean()
            num = np.sum((x - x_mean) * (window - y_mean))
            den = np.sum((x - x_mean) ** 2)
            if den > 0:
                obv_slope[i] = num / den

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 2.0 * avg_vol

        time_ok = (time_mins >= 570) & (time_mins <= 910)

        # Breakout confirmation: next bar still outside channel
        close_above_kelt = close > keltner_upper
        close_below_kelt = close < keltner_lower

        prev_above = np.zeros(n, dtype=np.bool_)
        prev_below = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            prev_above[i] = close_above_kelt[i - 1]
            prev_below[i] = close_below_kelt[i - 1]

        # Long: breakout above Keltner, range expansion, OBV positive slope
        long_entry = close_above_kelt & prev_above & (range_ratio > range_ratio_t) & \
                     (obv_slope > 0) & vol_ok & time_ok

        # Short: breakdown below Keltner
        short_entry = close_below_kelt & prev_below & (range_ratio > range_ratio_t) & \
                      (obv_slope < 0) & vol_ok & time_ok

        # Signal exit: close re-enters Keltner channel
        sig_exit_long = close < keltner_upper
        sig_exit_short = close > keltner_lower

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=120,
        )
