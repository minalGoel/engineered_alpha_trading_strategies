"""Momentum Exhaustion VIX Reversal — gemini_4_of_20

Thesis: When RSI(2) hits extreme levels (< 5 or > 95) with volume spike and
elevated VIX, expect a mean-reversion snap. Target EMA(20) touch.
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


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "momentum_exhaustion_vix_reversal"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 920     # 15:20
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_long_thresh", default=5.0, low=2.0, high=15.0),
            TunableParam("rsi_short_thresh", default=95.0, low=85.0, high=98.0),
            TunableParam("vol_mult", default=3.0, low=1.5, high=5.0),
            TunableParam("vix_min", default=15.0, low=10.0, high=25.0),
            TunableParam("stop_atr_mult", default=1.0, low=0.5, high=2.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rsi_long = params.get("rsi_long_thresh", 5.0)
        rsi_short = params.get("rsi_short_thresh", 95.0)
        vol_mult = params.get("vol_mult", 3.0)
        vix_min = params.get("vix_min", 15.0)
        stop_atr = params.get("stop_atr_mult", 1.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # RSI(2)
        rsi2 = _compute_rsi(close, 2)

        # Volume SMA(50)
        vol_sma50 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - 49)
            vol_sma50[i] = np.mean(volume[start:i + 1])
        vol_sma50 = np.where(vol_sma50 > 1e-10, vol_sma50, 1.0)
        vol_spike = volume > vol_mult * vol_sma50

        # VIX rising: current > 5-bar ago
        vix_rising = np.zeros(n, dtype=np.bool_)
        for i in range(5, n):
            vix_rising[i] = vix[i] > vix[i - 5]

        # EMA(20) for target
        ema20 = np.zeros(n, dtype=np.float64)
        alpha = 2.0 / 21.0
        ema20[0] = close[0]
        for i in range(1, n):
            ema20[i] = alpha * close[i] + (1 - alpha) * ema20[i - 1]

        # ATR(14)
        atr14 = _compute_atr(high, low, close, 14)

        # Time filter: no last 30 min (after 14:50 = 890)
        no_late = time_mins <= 890

        # Entries
        long_entry = ((rsi2 < rsi_long) & vol_spike & vix_rising &
                       (vix > vix_min) & no_late)
        short_entry = ((rsi2 > rsi_short) & vol_spike &
                        (vix > vix_min) & no_late)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=ema20,
            use_target_indicator=True,
            stop_loss_atr_mult=stop_atr,
            time_stop_bars=15,
        )
