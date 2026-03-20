"""Narrow Range (NR7) Breakout — Opus_15

Thesis: A bar whose range is the narrowest of the past 7 bars signals
compression.  When ATR is also declining, the subsequent breakout
(with volume confirmation) tends to produce clean directional moves.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple EMA."""
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    k = 2.0 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "opus_narrow_range_breakout_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=1.2, low=1.0, high=2.0),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("target_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 1.2)
        stop_pct = params.get("stop_loss_pct", 0.002)
        target_pct = params.get("target_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        bar_range = high - low

        # ── NR7: narrowest range in 7 bars ──
        is_nr7 = np.zeros(n, dtype=np.bool_)
        for i in range(6, n):
            window = bar_range[i - 6: i + 1]
            if bar_range[i] <= np.min(window) + 1e-12:
                is_nr7[i] = True

        # NR7 high/low (the NR7 bar's range)
        nr7_high = np.zeros(n, dtype=np.float64)
        nr7_low = np.zeros(n, dtype=np.float64)
        last_nr7_h = 0.0
        last_nr7_l = 0.0
        for i in range(n):
            if is_nr7[i]:
                last_nr7_h = high[i]
                last_nr7_l = low[i]
            nr7_high[i] = last_nr7_h
            nr7_low[i] = last_nr7_l

        # ── ATR declining ──
        atr14 = _compute_atr(high, low, close, 14)
        atr_declining = np.zeros(n, dtype=np.bool_)
        for i in range(3, n):
            if atr14[i] < atr14[i - 3]:
                atr_declining[i] = True

        # ── EMA(20) trend filter ──
        ema20 = _compute_ema(close, 20)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Entry: bar after NR7, breakout of its high/low ──
        # We look for breakout on subsequent bars
        breakout_long = np.zeros(n, dtype=np.bool_)
        breakout_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if nr7_high[i - 1] > 0:
                if close[i] > nr7_high[i - 1]:
                    breakout_long[i] = True
                if close[i] < nr7_low[i - 1]:
                    breakout_short[i] = True

        long_entry = breakout_long & atr_declining & (close > ema20) & vol_ok
        short_entry = breakout_short & atr_declining & (close < ema20) & vol_ok

        # ── Signal exit: close back inside NR7 range ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if nr7_high[i] > 0 and nr7_low[i] > 0:
                if close[i] < nr7_high[i] and close[i] > nr7_low[i]:
                    sig_exit_long[i] = True
                    sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=45,
        )
