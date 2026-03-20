"""HMM Regime v1 — cursor_opus46max_166

Thesis: Approximate a Hidden Markov Model regime classifier using observable
indicators. State 1 (trending): ADX > 25, EMA crossover. State 2
(mean-reverting): ADX < 18, BB reversion. State 3 (choppy): high ATR
acceleration, stay flat.
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


def _compute_adx(high, low, close, period):
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    if n < 2 * period:
        return adx, plus_di, minus_di
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
    atr_s = np.zeros(n, dtype=np.float64)
    pdm_s = np.zeros(n, dtype=np.float64)
    mdm_s = np.zeros(n, dtype=np.float64)
    atr_s[period] = np.sum(tr[1:period + 1])
    pdm_s[period] = np.sum(plus_dm[1:period + 1])
    mdm_s[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
        pdm_s[i] = pdm_s[i - 1] - pdm_s[i - 1] / period + plus_dm[i]
        mdm_s[i] = mdm_s[i - 1] - mdm_s[i - 1] / period + minus_dm[i]
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if atr_s[i] > 0:
            plus_di[i] = 100.0 * pdm_s[i] / atr_s[i]
            minus_di[i] = 100.0 * mdm_s[i] / atr_s[i]
        s = plus_di[i] + minus_di[i]
        if s > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / s
    start = 2 * period
    if start < n:
        adx[start] = np.mean(dx[period:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx, plus_di, minus_di


class Strategy(BaseStrategy):
    name = "cursor_opus46max_166"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_trend_thresh", default=25.0, low=20.0, high=35.0),
            TunableParam("adx_range_thresh", default=18.0, low=12.0, high=22.0),
            TunableParam("trend_target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("trend_stop_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("rev_stop_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_trend = params.get("adx_trend_thresh", 25.0)
        adx_range = params.get("adx_range_thresh", 18.0)
        trend_target = params.get("trend_target_pct", 0.005)
        trend_stop = params.get("trend_stop_pct", 0.003)
        rev_stop = params.get("rev_stop_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        adx, plus_di, minus_di = _compute_adx(high, low, close, 14)
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)

        # BB
        sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        sma20 = np.nan_to_num(sma20, nan=0.0)
        std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        std20 = np.nan_to_num(std20, nan=1e10)
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20

        # ATR acceleration for choppy detection
        atr14 = _compute_atr(high, low, close, 14)
        atr_accel = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            if atr14[i - 10] > 1e-10:
                atr_accel[i] = atr14[i] / atr14[i - 10] - 1.0

        # State classification
        is_trending = (adx > adx_trend) & (atr_accel < 0.5)
        is_ranging = (adx < adx_range)
        is_choppy = atr_accel > 0.5

        # EMA crossover detection
        ema_cross_up = np.zeros(n, dtype=np.bool_)
        ema_cross_dn = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if ema9[i - 1] <= ema21[i - 1] and ema9[i] > ema21[i]:
                ema_cross_up[i] = True
            if ema9[i - 1] >= ema21[i - 1] and ema9[i] < ema21[i]:
                ema_cross_dn[i] = True

        # Volume filter
        avg_vol = df["volume"].rolling_mean(15).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 560) & (time_mins <= 915)

        # Trending entries
        trend_long = is_trending & (ema9 > ema21) & (close > vwap) & vol_ok & time_ok
        trend_short = is_trending & (ema9 < ema21) & (close < vwap) & vol_ok & time_ok
        # Ranging entries
        range_long = is_ranging & (close < bb_lower) & time_ok
        range_short = is_ranging & (close > bb_upper) & time_ok

        long_entry = trend_long | range_long
        short_entry = trend_short | range_short

        # Signal exit: state transitions to choppy
        sig_exit_long = is_choppy
        sig_exit_short = is_choppy

        avg_stop = (trend_stop + rev_stop) / 2.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr14,
            target_indicator=sma20.copy(),
            stop_loss_pct=avg_stop,
            target_pct=trend_target,
            trailing_stop_pct=0.002,
            trailing_activate_pct=0.003,
            time_stop_bars=75,
        )
