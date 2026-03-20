"""Intraday RSI Divergence — Opus_7

Thesis: RSI(9) bullish/bearish divergence on 1-min bars signals
exhaustion of the current move. Bullish: new price swing low but
higher RSI low. Bearish: new price swing high but lower RSI high.
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


def _find_swing_lows(arr, lookback):
    """Find swing lows: bar is lower than all bars in lookback window on both sides."""
    n = len(arr)
    is_swing = np.zeros(n, dtype=np.bool_)
    for i in range(lookback, n - lookback):
        left = arr[i - lookback:i]
        right = arr[i + 1:i + lookback + 1]
        if arr[i] <= np.min(left) and arr[i] <= np.min(right):
            is_swing[i] = True
    return is_swing


def _find_swing_highs(arr, lookback):
    """Find swing highs: bar is higher than all bars in lookback window on both sides."""
    n = len(arr)
    is_swing = np.zeros(n, dtype=np.bool_)
    for i in range(lookback, n - lookback):
        left = arr[i - lookback:i]
        right = arr[i + 1:i + lookback + 1]
        if arr[i] >= np.max(left) and arr[i] >= np.max(right):
            is_swing[i] = True
    return is_swing


class Strategy(BaseStrategy):
    name = "opus_intraday_rsi_divergence_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_long_thresh", default=35.0, low=20.0, high=45.0),
            TunableParam("rsi_short_thresh", default=65.0, low=55.0, high=80.0),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("breakeven_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rsi_long = params.get("rsi_long_thresh", 35.0)
        rsi_short = params.get("rsi_short_thresh", 65.0)
        target_pct = params.get("target_pct", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.003)
        be_pct = params.get("breakeven_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── RSI(9) ──
        rsi = _compute_rsi(close, 9)

        # ── Find swing lows/highs with 5-bar lookback ──
        swing_lookback = 5
        price_swing_lows = _find_swing_lows(low, swing_lookback)
        rsi_swing_lows = _find_swing_lows(rsi, swing_lookback)
        price_swing_highs = _find_swing_highs(high, swing_lookback)
        rsi_swing_highs = _find_swing_highs(rsi, swing_lookback)

        # ── Detect divergences ──
        # Bullish divergence: current swing low in price < prev swing low,
        #   but current RSI swing low > prev RSI swing low
        # Look back up to 30 bars for previous swing
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        # Track last swing low/high values
        last_price_low = np.nan
        last_rsi_low = np.nan
        last_price_low_idx = -100
        last_price_high = np.nan
        last_rsi_high = np.nan
        last_price_high_idx = -100

        time_ok = (time_mins >= 570) & (time_mins <= 885)

        for i in range(swing_lookback, n - swing_lookback):
            if not time_ok[i]:
                continue

            # Check for bullish divergence at swing lows
            if price_swing_lows[i]:
                if (not np.isnan(last_price_low)
                        and (i - last_price_low_idx) <= 30
                        and low[i] < last_price_low
                        and rsi[i] > last_rsi_low
                        and rsi[i] < rsi_long):
                    long_entry[i] = True
                last_price_low = low[i]
                last_rsi_low = rsi[i]
                last_price_low_idx = i

            # Check for bearish divergence at swing highs
            if price_swing_highs[i]:
                if (not np.isnan(last_price_high)
                        and (i - last_price_high_idx) <= 30
                        and high[i] > last_price_high
                        and rsi[i] < last_rsi_high
                        and rsi[i] > rsi_short):
                    short_entry[i] = True
                last_price_high = high[i]
                last_rsi_high = rsi[i]
                last_price_high_idx = i

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            breakeven_pct=be_pct,
            time_stop_bars=45,
        )
