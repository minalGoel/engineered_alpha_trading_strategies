"""MACD Trend Continuation — cursor_gemini31pro_strategy_071

Thesis: When the MACD line crosses the signal line in the direction of the
primary trend (close vs EMA-50), it flags an acceleration of institutional
buying/selling. MACD crossover below zero (for longs) or above zero (for shorts)
indicates early momentum. Volume surge confirmation required.
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


def _ema(arr, period):
    """Compute EMA using standard multiplier."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


class Strategy(BaseStrategy):
    name = "gemini31pro_macd_trend_continuation_v71"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_low", default=12.0, low=8.0, high=18.0),
            TunableParam("vix_high", default=22.0, low=18.0, high=30.0),
            TunableParam("stop_atr_mult", default=1.5, low=0.8, high=3.0),
            TunableParam("target_atr_mult", default=21.0, low=10.0, high=30.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_lo = params.get("vix_low", 12.0)
        vix_hi = params.get("vix_high", 22.0)
        stop_atr = params.get("stop_atr_mult", 1.5)
        tgt_atr = params.get("target_atr_mult", 21.0)
        be_pct = params.get("breakeven_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr14 = _compute_atr(high, low, close, 14)

        # EMA(50) for trend direction
        ema50 = _ema(close, 50)

        # MACD: EMA(12) - EMA(26), signal = EMA(macd, 9)
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd_line = ema12 - ema26
        macd_signal = _ema(macd_line, 9)

        # MACD crossover/crossunder detection
        prev_macd = np.roll(macd_line, 1)
        prev_macd[0] = macd_line[0]
        prev_signal = np.roll(macd_signal, 1)
        prev_signal[0] = macd_signal[0]

        cross_up = (prev_macd <= prev_signal) & (macd_line > macd_signal)
        cross_down = (prev_macd >= prev_signal) & (macd_line < macd_signal)

        # Volume SMA(20)
        vol_sma20 = np.full(n, 1.0, dtype=np.float64)
        for i in range(19, n):
            vol_sma20[i] = np.mean(volume[i - 19:i + 1])
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        vol_ok = volume > vol_sma20

        # Filters
        vix_ok = (vix >= vix_lo) & (vix <= vix_hi)
        time_ok = (time_mins >= 570) & (time_mins <= 870)  # 09:30-14:30

        # Entry: long when close > EMA50, MACD crosses above signal, MACD < 0
        long_entry = (close > ema50) & cross_up & (macd_line < 0) & vol_ok & vix_ok & time_ok
        # Entry: short when close < EMA50, MACD crosses below signal, MACD > 0
        short_entry = (close < ema50) & cross_down & (macd_line > 0) & vol_ok & vix_ok & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_atr_mult=tgt_atr,
            breakeven_pct=be_pct,
            time_stop_bars=60,
        )
