"""Market State Classifier v1 — cursor_opus46max_169

Thesis: Classify market into TRENDING (ADX>25, low ATR accel), RANGING
(ADX<18, narrow BB), VOLATILE (high ATR accel) using decision tree.
Trending: EMA crossover. Ranging: BB fade. Volatile: flat.
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


def _compute_adx(high, low, close, period):
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    if n < 2 * period:
        return adx
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
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
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
    return adx


class Strategy(BaseStrategy):
    name = "cursor_opus46max_169"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_trend", default=25.0, low=20.0, high=35.0),
            TunableParam("adx_range", default=18.0, low=12.0, high=22.0),
            TunableParam("atr_accel_thresh", default=0.5, low=0.3, high=0.8),
            TunableParam("trend_target_pct", default=0.0045, low=0.003, high=0.007),
            TunableParam("range_target_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("trend_stop_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("range_stop_pct", default=0.0015, low=0.001, high=0.0025),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_trend_t = params.get("adx_trend", 25.0)
        adx_range_t = params.get("adx_range", 18.0)
        atr_accel_t = params.get("atr_accel_thresh", 0.5)
        t_target = params.get("trend_target_pct", 0.0045)
        r_target = params.get("range_target_pct", 0.0025)
        t_stop = params.get("trend_stop_pct", 0.0025)
        r_stop = params.get("range_stop_pct", 0.0015)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high_arr = df["high"].to_numpy().astype(np.float64)
        low_arr = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        adx = _compute_adx(high_arr, low_arr, close, 14)
        atr = _compute_atr(high_arr, low_arr, close, 14)
        rsi = _compute_rsi(close, 14)
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)

        # BB
        sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        sma20 = np.nan_to_num(sma20, nan=0.0)
        std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        std20 = np.nan_to_num(std20, nan=1e10)
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20

        # ATR acceleration
        atr_accel = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            if atr[i - 10] > 1e-10:
                atr_accel[i] = atr[i] / atr[i - 10] - 1.0

        # State classification
        is_volatile = atr_accel > atr_accel_t
        is_trending = (adx > adx_trend_t) & (~is_volatile)
        is_ranging = (adx < adx_range_t) & (~is_volatile)

        # State stability (15 bars)
        state_stable = np.zeros(n, dtype=np.bool_)
        for i in range(15, n):
            stable = True
            for j in range(1, 15):
                prev_trend = adx[i - j] > adx_trend_t and atr_accel[i - j] <= atr_accel_t
                prev_range = adx[i - j] < adx_range_t and atr_accel[i - j] <= atr_accel_t
                cur_trend = is_trending[i]
                cur_range = is_ranging[i]
                if cur_trend and not prev_trend:
                    stable = False
                    break
                if cur_range and not prev_range:
                    stable = False
                    break
            state_stable[i] = stable

        # EMA cross for trending
        ema_cross_up = np.zeros(n, dtype=np.bool_)
        ema_cross_dn = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if ema9[i - 1] <= ema21[i - 1] and ema9[i] > ema21[i]:
                ema_cross_up[i] = True
            if ema9[i - 1] >= ema21[i - 1] and ema9[i] < ema21[i]:
                ema_cross_dn[i] = True

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 570) & (time_mins <= 910)
        vix_ok = vix < 25.0

        # Trending long/short
        t_long = is_trending & (ema9 > ema21) & (close > vwap) & vol_ok & state_stable & time_ok & vix_ok
        t_short = is_trending & (ema9 < ema21) & (close < vwap) & vol_ok & state_stable & time_ok & vix_ok

        # Ranging long/short
        r_long = is_ranging & (close < bb_lower) & (close < vwap) & (rsi < 35.0) & state_stable & time_ok & vix_ok
        r_short = is_ranging & (close > bb_upper) & (close > vwap) & (rsi > 65.0) & state_stable & time_ok & vix_ok

        long_entry = t_long | r_long
        short_entry = t_short | r_short

        # Signal exit: state transition to volatile or opposite
        sig_exit_long = is_volatile
        sig_exit_short = is_volatile

        avg_stop = (t_stop + r_stop) / 2.0
        avg_target = (t_target + r_target) / 2.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=sma20.copy(),
            stop_loss_pct=avg_stop,
            target_pct=avg_target,
            trailing_stop_pct=0.0015,
            trailing_activate_pct=0.0025,
            time_stop_bars=75,
        )
