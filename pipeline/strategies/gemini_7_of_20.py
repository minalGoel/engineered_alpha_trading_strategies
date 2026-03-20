# AUDIT FIX: Added session window filter to long_entry and short_entry
"""Bollinger Squeeze Breakout — gemini_7_of_20

Thesis: When Bollinger Band width compresses below 0.002 (squeeze), a breakout
with volume confirms directional expansion. Target via ATR, stop at SMA(20).
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


class Strategy(BaseStrategy):
    name = "bollinger_squeeze_breakout"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 885     # 14:45
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bb_width_thresh", default=0.002, low=0.0005, high=0.005),
            TunableParam("vol_mult", default=1.5, low=1.0, high=3.0),
            TunableParam("target_atr_mult", default=2.5, low=1.0, high=4.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        bb_width = params.get("bb_width_thresh", 0.002)
        vol_mult = params.get("vol_mult", 1.5)
        tgt_atr = params.get("target_atr_mult", 2.5)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)

        # SMA(20) and Bollinger Bands
        sma20 = np.zeros(n, dtype=np.float64)
        std20 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - 19)
            window = close[start:i + 1]
            sma20[i] = np.mean(window)
            std20[i] = np.std(window) if len(window) > 1 else 0.0

        safe_sma = np.where(sma20 > 1e-10, sma20, 1e-10)
        upper_bb = sma20 + 2.0 * std20
        lower_bb = sma20 - 2.0 * std20
        bb_w = (upper_bb - lower_bb) / safe_sma

        # Squeeze condition
        squeeze = bb_w < bb_width

        # Volume: current > vol_mult * previous bar
        vol_ok = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            prev_v = max(volume[i - 1], 1e-10)
            vol_ok[i] = volume[i] > vol_mult * prev_v

        # ATR(20)
        atr20 = _compute_atr(high, low, close, 20)

        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Entries
        long_entry = squeeze & (close > upper_bb) & vol_ok & in_session
        short_entry = squeeze & (close < lower_bb) & vol_ok & in_session

        # Signal exit: close crosses SMA(20) against direction
        signal_exit_long = close < sma20
        signal_exit_short = close > sma20

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr20,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_atr_mult=tgt_atr,
            time_stop_bars=150,
        )
