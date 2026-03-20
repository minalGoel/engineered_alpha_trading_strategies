"""VIX Bucket Strategy v1 — cursor_opus46max_170

Thesis: VIX-bucketed strategy selection. Calm (<13): aggressive BB
mean-reversion. Normal (13-18): EMA crossover. Elevated (18-24): wide-stop
Supertrend. Crisis (>24): flat.
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


def _compute_supertrend(high, low, close, atr, multiplier):
    n = len(close)
    supertrend = np.zeros(n, dtype=np.float64)
    direction = np.ones(n, dtype=np.float64)
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
    name = "cursor_opus46max_170"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 910     # 15:10
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_calm", default=13.0, low=10.0, high=15.0),
            TunableParam("vix_normal", default=18.0, low=15.0, high=21.0),
            TunableParam("vix_elevated", default=24.0, low=21.0, high=28.0),
            TunableParam("calm_target_pct", default=0.002, low=0.001, high=0.003),
            TunableParam("normal_target_pct", default=0.0035, low=0.002, high=0.005),
            TunableParam("elevated_target_pct", default=0.006, low=0.004, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_calm = params.get("vix_calm", 13.0)
        vix_normal = params.get("vix_normal", 18.0)
        vix_elevated = params.get("vix_elevated", 24.0)
        calm_target = params.get("calm_target_pct", 0.002)
        normal_target = params.get("normal_target_pct", 0.0035)
        elevated_target = params.get("elevated_target_pct", 0.006)

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

        # VIX buckets
        is_calm = vix < vix_calm
        is_normal = (vix >= vix_calm) & (vix < vix_normal)
        is_elevated = (vix >= vix_normal) & (vix < vix_elevated)
        is_crisis = vix >= vix_elevated

        # Calm: BB(15, 1.5) + RSI(10)
        sma15 = df["close"].rolling_mean(15).to_numpy().astype(np.float64)
        sma15 = np.nan_to_num(sma15, nan=0.0)
        std15 = df["close"].rolling_std(15).to_numpy().astype(np.float64)
        std15 = np.nan_to_num(std15, nan=1e10)
        bb_upper_calm = sma15 + 1.5 * std15
        bb_lower_calm = sma15 - 1.5 * std15
        rsi10 = _compute_rsi(close, 10)

        # Normal: EMA(9)/EMA(21)
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)

        # Elevated: Supertrend(10, 4.0)
        atr10 = _compute_atr(high_arr, low_arr, close, 10)
        st, st_dir = _compute_supertrend(high_arr, low_arr, close, atr10, 4.0)

        # Uptick/downtick
        uptick = np.zeros(n, dtype=np.bool_)
        downtick = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            uptick[i] = close[i] > close[i - 1]
            downtick[i] = close[i] < close[i - 1]

        # Volume
        avg_vol = df["volume"].rolling_mean(15).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol
        vol_ok_elev = volume > 2.0 * avg_vol

        # Supertrend held for 3 bars
        st_held_bull = np.zeros(n, dtype=np.bool_)
        st_held_bear = np.zeros(n, dtype=np.bool_)
        for i in range(3, n):
            if st_dir[i] > 0 and st_dir[i - 1] > 0 and st_dir[i - 2] > 0:
                st_held_bull[i] = True
            if st_dir[i] < 0 and st_dir[i - 1] < 0 and st_dir[i - 2] < 0:
                st_held_bear[i] = True

        time_ok = (time_mins >= 565) & (time_mins <= 910)

        # Calm entries
        calm_long = is_calm & (close < bb_lower_calm) & (close < vwap) & (rsi10 < 30.0) & uptick & time_ok
        calm_short = is_calm & (close > bb_upper_calm) & (close > vwap) & (rsi10 > 70.0) & downtick & time_ok

        # Normal entries
        normal_long = is_normal & (ema9 > ema21) & (close > vwap) & vol_ok & time_ok
        normal_short = is_normal & (ema9 < ema21) & (close < vwap) & vol_ok & time_ok

        # Elevated entries
        elev_long = is_elevated & (st_dir > 0) & (close > vwap) & vol_ok_elev & st_held_bull & time_ok
        elev_short = is_elevated & (st_dir < 0) & (close < vwap) & vol_ok_elev & st_held_bear & time_ok

        long_entry = calm_long | normal_long | elev_long
        short_entry = calm_short | normal_short | elev_short

        # Signal exit: VIX bucket transition
        sig_exit_long = is_crisis.copy()
        sig_exit_short = is_crisis.copy()

        # Average targets/stops
        avg_target = (calm_target + normal_target + elevated_target) / 3.0
        avg_stop = avg_target * 0.6

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr10,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=avg_stop,
            target_pct=avg_target,
            trailing_stop_pct=0.0012,
            trailing_activate_pct=0.002,
            time_stop_bars=90,
        )
