"""Ensemble Signal v1 — cursor_opus46max_160

Thesis: Run 4 sub-strategies (VWAP reversion, EMA crossover, BB breakout,
RSI reversal) and enter when >= 3/4 agree on direction for 2 consecutive bars.
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


class Strategy(BaseStrategy):
    name = "cursor_opus46max_160"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ensemble_thresh", default=3.0, low=2.0, high=4.0),
            TunableParam("vwap_dist_thresh", default=0.003, low=0.001, high=0.005),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.003),
            TunableParam("trailing_stop_pct", default=0.0012, low=0.0008, high=0.002),
            TunableParam("trailing_activate_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ens_thresh = int(params.get("ensemble_thresh", 3.0))
        vwap_dist_t = params.get("vwap_dist_thresh", 0.003)
        target_pct = params.get("target_pct", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.002)
        trail_pct = params.get("trailing_stop_pct", 0.0012)
        trail_act = params.get("trailing_activate_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
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

        # Sub1: VWAP reversion signal
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        vwap_dev = (close - vwap) / safe_vwap
        sub1 = np.where(vwap_dev < -vwap_dist_t, 1,
               np.where(vwap_dev > vwap_dist_t, -1, 0)).astype(np.float64)

        # Sub2: EMA crossover signal
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)
        sub2 = np.where(ema9 > ema21, 1, -1).astype(np.float64)

        # Sub3: Bollinger Band breakout
        sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        sma20 = np.nan_to_num(sma20, nan=0.0)
        std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        std20 = np.nan_to_num(std20, nan=1e10)
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20
        sub3 = np.where(close > bb_upper, 1,
               np.where(close < bb_lower, -1, 0)).astype(np.float64)

        # Sub4: RSI reversal signal
        rsi = _compute_rsi(close, 14)
        sub4 = np.where(rsi < 30.0, 1,
               np.where(rsi > 70.0, -1, 0)).astype(np.float64)

        ensemble = sub1 + sub2 + sub3 + sub4

        # 2-bar confirmation
        long_raw = ensemble >= ens_thresh
        short_raw = ensemble <= -ens_thresh

        long_conf = np.zeros(n, dtype=np.bool_)
        short_conf = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if long_raw[i] and long_raw[i - 1]:
                long_conf[i] = True
            if short_raw[i] and short_raw[i - 1]:
                short_conf[i] = True

        # Filters
        time_ok = (time_mins >= 570) & (time_mins <= 910)
        vix_ok = vix < 25.0
        avg_vol = df["volume"].rolling_mean(15).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        long_entry = long_conf & time_ok & vix_ok & vol_ok
        short_entry = short_conf & time_ok & vix_ok & vol_ok

        # Signal exit: ensemble crosses zero
        sig_exit_long = ensemble <= 0
        sig_exit_short = ensemble >= 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=75,
        )
