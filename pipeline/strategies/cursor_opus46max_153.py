"""Trend Counter-Trend v1 — cursor_opus46max_153

Thesis: Supertrend-based trend entry combined with counter-trend exit
(RSI divergence + VWAP distance extremes). Captures the meat of the
move while exiting before the trend exhausts.
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


def _compute_ema(arr, period):
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = arr[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)
    return ema


def _compute_supertrend(high, low, close, atr, period, multiplier):
    """Compute Supertrend. Returns supertrend array and direction (+1 up, -1 down)."""
    n = len(close)
    supertrend = np.zeros(n, dtype=np.float64)
    direction = np.ones(n, dtype=np.float64)  # +1 = bullish
    upper_band = np.zeros(n, dtype=np.float64)
    lower_band = np.zeros(n, dtype=np.float64)

    for i in range(n):
        mid = (high[i] + low[i]) / 2.0
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]

    for i in range(1, n):
        if lower_band[i] < lower_band[i - 1] and close[i - 1] > lower_band[i - 1]:
            lower_band[i] = lower_band[i - 1]
        if upper_band[i] > upper_band[i - 1] and close[i - 1] < upper_band[i - 1]:
            upper_band[i] = upper_band[i - 1]

        if direction[i - 1] == 1:
            if close[i] < lower_band[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = 1
                supertrend[i] = lower_band[i]
        else:
            if close[i] > upper_band[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            else:
                direction[i] = -1
                supertrend[i] = upper_band[i]

    return supertrend, direction


class Strategy(BaseStrategy):
    name = "cursor_opus46max_153"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("st_multiplier", default=3.0, low=2.0, high=4.0),
            TunableParam("vwap_exit_bps", default=60.0, low=30.0, high=100.0),
            TunableParam("bar_range_thresh", default=0.7, low=0.5, high=0.9),
            TunableParam("stop_atr_mult", default=1.0, low=0.5, high=1.5),
            TunableParam("target_atr_mult", default=1.5, low=1.0, high=2.5),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        st_mult = params.get("st_multiplier", 3.0)
        vwap_exit_bps = params.get("vwap_exit_bps", 60.0)
        range_thresh = params.get("bar_range_thresh", 0.7)
        stop_mult = params.get("stop_atr_mult", 1.0)
        target_mult = params.get("target_atr_mult", 1.5)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        idx_close = df["index_close"].to_numpy().astype(np.float64)
        idx_close = np.nan_to_num(idx_close, nan=0.0)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ATR
        atr = _compute_atr(high, low, close, 14)

        # Supertrend
        atr_st = _compute_atr(high, low, close, 10)
        st, st_dir = _compute_supertrend(high, low, close, atr_st, 10, st_mult)

        # RSI
        rsi = _compute_rsi(close, 14)

        # Index EMA(20)
        idx_ema20 = _compute_ema(idx_close, 20)

        # Bar range ratio
        bar_range = np.zeros(n, dtype=np.float64)
        for i in range(n):
            rng = high[i] - low[i]
            if rng > 0:
                bar_range[i] = (close[i] - low[i]) / rng
            else:
                bar_range[i] = 0.5

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 560) & (time_mins <= 910)

        # Supertrend flip detection
        st_flip_bull = np.zeros(n, dtype=np.bool_)
        st_flip_bear = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if st_dir[i - 1] <= 0 and st_dir[i] > 0:
                st_flip_bull[i] = True
            if st_dir[i - 1] >= 0 and st_dir[i] < 0:
                st_flip_bear[i] = True

        # Entry
        long_entry = (st_flip_bull | ((st_dir > 0) & (close > vwap))) & \
                     (close > vwap) & (idx_close > idx_ema20) & \
                     (bar_range > range_thresh) & vol_ok & time_ok
        short_entry = (st_flip_bear | ((st_dir < 0) & (close < vwap))) & \
                      (close < vwap) & (idx_close < idx_ema20) & \
                      (bar_range < (1.0 - range_thresh)) & vol_ok & time_ok

        # Counter-trend signal exit: VWAP distance extreme
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        vwap_dist = np.abs((close - vwap) / safe_vwap * 10000.0)
        sig_exit_long = (vwap_dist > vwap_exit_bps) & (rsi > 65.0)
        sig_exit_short = (vwap_dist > vwap_exit_bps) & (rsi < 35.0)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_mult,
            target_atr_mult=target_mult,
            trailing_stop_pct=0.0,
            trailing_activate_pct=0.0,
            time_stop_bars=120,
        )
