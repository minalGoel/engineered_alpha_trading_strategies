"""Intraday Vol Pattern v1 — cursor_opus46max_079

Thesis: Intraday volatility follows a U-shape. Mean-reversion during lunch
(low vol), momentum during open/close (high vol), and vol-expansion during
afternoon transition. Strategy adapts by session period.
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
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = arr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss > 0:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss > 0:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i + 1] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_079"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 929     # 15:29
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bb_period", default=20.0, low=15.0, high=30.0),
            TunableParam("bb_mult", default=2.0, low=1.5, high=2.5),
            TunableParam("vol_surprise_low", default=0.8, low=0.5, high=1.0),
            TunableParam("vol_surprise_high", default=1.2, low=1.0, high=1.5),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_atr_mult", default=1.5, low=1.0, high=2.5),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        bb_period = int(params.get("bb_period", 20.0))
        bb_mult = params.get("bb_mult", 2.0)
        vol_surprise_low = params.get("vol_surprise_low", 0.8)
        vol_surprise_high = params.get("vol_surprise_high", 1.2)
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_atr_mult = params.get("target_atr_mult", 1.5)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        time_min = df["time_minutes"].to_numpy()

        # ── Bollinger Bands ──
        sma = np.zeros(n, dtype=np.float64)
        std = np.zeros(n, dtype=np.float64)
        for i in range(bb_period, n):
            sma[i] = np.mean(close[i - bb_period:i])
            std[i] = np.std(close[i - bb_period:i])
        bb_upper = sma + bb_mult * std
        bb_lower = sma - bb_mult * std

        # ── EMA(5) and EMA(13) ──
        ema5 = _ema(close, 5)
        ema13 = _ema(close, 13)

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        # ── Current vol (rolling std of returns, 15 bars) ──
        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0:
                returns[i] = (close[i] - close[i - 1]) / close[i - 1]
        current_vol = np.zeros(n, dtype=np.float64)
        for i in range(15, n):
            current_vol[i] = np.std(returns[i - 15:i])

        # ── Expected vol: rolling mean of current_vol over 120 bars ──
        expected_vol = np.zeros(n, dtype=np.float64)
        for i in range(120, n):
            expected_vol[i] = np.mean(current_vol[i - 120:i])
        expected_vol = np.clip(expected_vol, 1e-10, None)
        vol_surprise = current_vol / expected_vol

        # ── ATR(10) ──
        atr = _compute_atr(high, low, close, 10)

        # ── Session period classification ──
        # open: <600 (10:00), lunch: 690-810 (11:30-13:30),
        # afternoon: 810-870 (13:30-14:30), close: >870
        is_open = time_min < 600
        is_lunch = (time_min >= 690) & (time_min < 810)
        is_afternoon = (time_min >= 810) & (time_min < 870)
        is_close_period = time_min >= 870

        # ── 20-bar high/low for breakout confirmation ──
        high_20 = np.zeros(n, dtype=np.float64)
        low_20 = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            high_20[i] = np.max(close[i - 20:i])
            low_20[i] = np.min(close[i - 20:i])

        # ── Long entries by period ──
        lunch_long = is_lunch & (close < bb_lower) & (vol_surprise < vol_surprise_low) & (rsi < 30)
        open_close_long = (is_open | is_close_period) & (ema5 > ema13) & (vol_surprise > vol_surprise_high) & (close > high_20)
        afternoon_long = is_afternoon & (close > sma) & (vol_surprise > 1.0)

        long_entry = lunch_long | open_close_long | afternoon_long

        # ── Short entries by period ──
        lunch_short = is_lunch & (close > bb_upper) & (vol_surprise < vol_surprise_low) & (rsi > 70)
        open_close_short = (is_open | is_close_period) & (ema5 < ema13) & (vol_surprise > vol_surprise_high) & (close < low_20)
        afternoon_short = is_afternoon & (close < sma) & (vol_surprise > 1.0)

        short_entry = lunch_short | open_close_short | afternoon_short

        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=sma.copy(),
            stop_loss_pct=stop_pct,
            target_atr_mult=target_atr_mult,
            time_stop_bars=45,
        )
