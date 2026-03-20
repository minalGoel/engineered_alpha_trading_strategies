"""VIX Regime Switch v1 — cursor_opus46max_071

Thesis: Adaptive strategy switching between mean-reversion (low VIX) and
momentum (elevated VIX). No trading in crisis VIX (>25).
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff = close[i] - close[i-1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    if avg_loss > 1e-10:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 1e-10:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i] = 100.0
    return rsi


def _ema(data, period):
    n = len(data)
    out = np.zeros(n, dtype=np.float64)
    out[0] = data[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_071"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_low_thresh", default=13.0, low=10.0, high=16.0),
            TunableParam("vix_elevated_thresh", default=18.0, low=15.0, high=22.0),
            TunableParam("vix_crisis_thresh", default=25.0, low=22.0, high=30.0),
            TunableParam("bb_period", default=20.0, low=15.0, high=30.0),
            TunableParam("rsi_low", default=30.0, low=20.0, high=40.0),
            TunableParam("rsi_high", default=70.0, low=60.0, high=80.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_low = params.get("vix_low_thresh", 13.0)
        vix_elev = params.get("vix_elevated_thresh", 18.0)
        vix_crisis = params.get("vix_crisis_thresh", 25.0)
        bb_period = int(params.get("bb_period", 20.0))
        rsi_low = params.get("rsi_low", 30.0)
        rsi_high = params.get("rsi_high", 70.0)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high_ = df["high"].to_numpy().astype(np.float64)
        low_ = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        atr14 = _compute_atr(high_, low_, close, 14)
        rsi14 = _compute_rsi(close, 14)
        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)

        # Bollinger Bands
        bb_mid = np.zeros(n, dtype=np.float64)
        bb_upper = np.zeros(n, dtype=np.float64)
        bb_lower = np.zeros(n, dtype=np.float64)
        for i in range(bb_period, n):
            seg = close[i - bb_period + 1:i + 1]
            bb_mid[i] = np.mean(seg)
            std = np.std(seg)
            bb_upper[i] = bb_mid[i] + 2.0 * std
            bb_lower[i] = bb_mid[i] - 2.0 * std

        # ADX approximation: using ATR as trend filter
        # Simplified: use EMA slope as trend indicator
        ema_slope = np.zeros(n, dtype=np.float64)
        for i in range(3, n):
            ema_slope[i] = ema9[i] - ema9[i-3]

        # VIX regime
        regime_mr = (vix < vix_elev)  # mean reversion
        regime_mom = (vix >= vix_elev) & (vix < vix_crisis)  # momentum
        regime_crisis = vix >= vix_crisis

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        for i in range(1, n):
            if regime_crisis[i]:
                continue

            if regime_mr[i]:
                # Mean reversion: BB touch + RSI extreme
                if close[i] < bb_lower[i] and rsi14[i] < rsi_low and rsi14[i] > rsi14[i-1]:
                    long_entry[i] = True
                if close[i] > bb_upper[i] and rsi14[i] > rsi_high and rsi14[i] < rsi14[i-1]:
                    short_entry[i] = True
            elif regime_mom[i]:
                # Momentum: EMA cross + positive slope
                if (close[i] > ema9[i] and ema9[i] > ema21[i]
                        and ema_slope[i] > 0 and ema_slope[i-1] > 0):
                    long_entry[i] = True
                if (close[i] < ema9[i] and ema9[i] < ema21[i]
                        and ema_slope[i] < 0 and ema_slope[i-1] < 0):
                    short_entry[i] = True

        # Signal exit: regime change or indicator reversal
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if regime_crisis[i]:
                signal_exit_long[i] = True
                signal_exit_short[i] = True
            elif regime_mr[i]:
                if close[i] >= bb_mid[i]:
                    signal_exit_long[i] = True
                if close[i] <= bb_mid[i]:
                    signal_exit_short[i] = True
            elif regime_mom[i]:
                if ema9[i] < ema21[i]:
                    signal_exit_long[i] = True
                if ema9[i] > ema21[i]:
                    signal_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=120,
        )
