# AUDIT FIX: Added time_ok session filter to long_entry and short_entry (were firing outside session window)
"""Psychological Level Mean Reversion — gemini_9_of_20

Thesis: Price tends to bounce off round psychological numbers (multiples of 50
and 100). Combine with RSI extremes for mean-reversion entries.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


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
    name = "psychological_level_mean_reversion"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 920     # 15:20
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("round_proximity", default=0.002, low=0.0005, high=0.005),
            TunableParam("rsi_long_thresh", default=30.0, low=15.0, high=40.0),
            TunableParam("rsi_short_thresh", default=70.0, low=60.0, high=85.0),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        round_prox = params.get("round_proximity", 0.002)
        rsi_long = params.get("rsi_long_thresh", 30.0)
        rsi_short = params.get("rsi_short_thresh", 70.0)
        tgt_pct = params.get("target_pct", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # RSI(14)
        rsi14 = _compute_rsi(close, 14)

        # Near round number: distance to nearest multiple of 50
        # mod 100 gives distance to nearest 100; mod 50 gives distance to nearest 50
        dist_50 = np.mod(close, 50.0)
        # Distance to nearest 50-multiple: min(dist_50, 50 - dist_50)
        dist_to_round = np.minimum(dist_50, 50.0 - dist_50)
        safe_close = np.where(close > 1e-10, close, 1e-10)
        near_round = (dist_to_round / safe_close) < round_prox

        atr14 = _compute_atr(high, low, close, 14)

        # Session filter
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Entries
        long_entry = near_round & (rsi14 < rsi_long) & time_ok
        short_entry = near_round & (rsi14 > rsi_short) & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=20,
        )
