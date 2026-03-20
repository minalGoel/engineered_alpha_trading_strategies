"""RSI Connors Pullback v42 — Gemini31Pro Strategy 042

Thesis: Short-term extreme oversold (RSI-2) within a longer-term uptrend
(close > SMA-200) indicates temporary liquidity exhaustion.
Long when RSI(2) < 14 & close > SMA(200);
short when RSI(2) > 10 & close < SMA(200).
Signal exit: close > SMA(close,5) for longs, close < SMA(close,5) for shorts.
No VIX filter. No volume filter. Time stop 30 bars.
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
    """Wilder's smoothed RSI."""
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
    name = "gemini31pro_rsi_connors_pullback_v42"
    is_long_only = False
    session_start = 570
    session_end = 915
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_long_thresh", default=14.0, low=5.0, high=25.0),
            TunableParam("rsi_short_thresh", default=10.0, low=60.0, high=95.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rsi_long_thresh = params.get("rsi_long_thresh", 14.0)
        rsi_short_thresh = params.get("rsi_short_thresh", 10.0)
        be_pct = params.get("breakeven_pct", 0.005)

        n = len(df)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── RSI(2) ──
        rsi_2 = _compute_rsi(close, 2)

        # ── SMA(200) ──
        sma_200 = df["close"].rolling_mean(200).to_numpy().astype(np.float64)
        sma_200 = np.nan_to_num(sma_200, nan=0.0)

        # ── SMA(5) for signal exit ──
        sma_5 = df["close"].rolling_mean(5).to_numpy().astype(np.float64)
        sma_5 = np.nan_to_num(sma_5, nan=0.0)

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (close > sma_200) & (rsi_2 < rsi_long_thresh) & time_ok
        short_entry = (close < sma_200) & (rsi_2 > rsi_short_thresh) & time_ok

        # ── Signal exits: close crosses SMA(5) ──
        signal_exit_long = close > sma_5
        signal_exit_short = close < sma_5

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=1.5,
            breakeven_pct=be_pct,
            time_stop_bars=30,
        )
