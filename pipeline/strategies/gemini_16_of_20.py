# AUDIT FIX: Added session window enforcement — entries were firing outside session_start/session_end
"""Multi EMA Ribbon Squeeze — gemini_16_of_20

Thesis: When the EMA(9), EMA(21), and EMA(50) converge into a tight squeeze,
a breakout above/below all EMAs with volume confirmation signals a new trend.
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


def _compute_ema(data, period):
    n = len(data)
    ema = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return np.nan_to_num(ema, nan=data[0] if n > 0 else 0.0)
    ema[period - 1] = np.mean(data[:period])
    mult = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = data[i] * mult + ema[i - 1] * (1.0 - mult)
    for i in range(period - 1):
        ema[i] = ema[period - 1]
    return ema


class Strategy(BaseStrategy):
    name = "multi_ema_ribbon_squeeze"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 920     # 15:20
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("squeeze_ratio", default=1.001, low=1.0002, high=1.003),
            TunableParam("target_atr_mult", default=3.0, low=1.5, high=5.0),
            TunableParam("vol_mult", default=1.0, low=0.5, high=2.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        squeeze_ratio = params.get("squeeze_ratio", 1.001)
        target_atr = params.get("target_atr_mult", 3.0)
        vol_mult_thresh = params.get("vol_mult", 1.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)

        atr20 = _compute_atr(high, low, close, 20)

        # EMAs
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)
        ema50 = _compute_ema(close, 50)

        # Squeeze: max(emas)/min(emas) < squeeze_ratio
        ema_max = np.maximum(np.maximum(ema9, ema21), ema50)
        ema_min = np.minimum(np.minimum(ema9, ema21), ema50)
        safe_min = np.where(ema_min > 1e-10, ema_min, 1e-10)
        squeeze = (ema_max / safe_min) < squeeze_ratio

        # Volume SMA(20)
        vol_sma20 = np.full(n, 1.0, dtype=np.float64)
        for i in range(19, n):
            vol_sma20[i] = np.mean(volume[i - 19:i + 1])
        vol_sma20 = np.where(vol_sma20 > 1e-10, vol_sma20, 1.0)
        vol_ok = volume > (vol_mult_thresh * vol_sma20)

        # Session window filter
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Long: squeeze + close > max(all EMAs) + volume > SMA(20)
        long_entry = squeeze & (close > ema_max) & vol_ok & in_session

        # Short: squeeze + close < min(all EMAs) + volume > SMA(20)
        short_entry = squeeze & (close < ema_min) & vol_ok & in_session

        # Signal exit: close crosses EMA(50) — stop-like
        signal_exit_long = close < ema50
        signal_exit_short = close > ema50

        # Trailing: close crosses EMA(21)
        # Use EMA(21) distance for trailing pct
        median_close = np.median(close) if n > 0 else 1.0
        median_ema21_dist = np.median(np.abs(close - ema21)) if n > 0 else 0.005 * median_close
        trailing_pct = median_ema21_dist / median_close if median_close > 1e-10 else 0.005

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr20,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_atr_mult=target_atr,
            trailing_stop_pct=trailing_pct,
            time_stop_bars=150,
        )
