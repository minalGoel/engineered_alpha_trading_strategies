"""Vol Regime Adaptive v1 — cursor_opus46max_167

Thesis: ATR percentile buckets determine regime. Low-vol (ATR pctile < 30):
BB mean-reversion. High-vol (ATR pctile > 70): EMA momentum. Medium-vol:
flat. Simplified using rolling 60-bar ATR percentile.
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


class Strategy(BaseStrategy):
    name = "cursor_opus46max_167"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 910     # 15:10
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("low_vol_pctile", default=30.0, low=15.0, high=40.0),
            TunableParam("high_vol_pctile", default=70.0, low=60.0, high=85.0),
            TunableParam("low_target_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("high_target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("low_stop_pct", default=0.0015, low=0.001, high=0.0025),
            TunableParam("high_stop_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        low_pctile = params.get("low_vol_pctile", 30.0)
        high_pctile = params.get("high_vol_pctile", 70.0)
        low_target = params.get("low_target_pct", 0.0025)
        high_target = params.get("high_target_pct", 0.005)
        low_stop = params.get("low_stop_pct", 0.0015)
        high_stop = params.get("high_stop_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high_arr = df["high"].to_numpy().astype(np.float64)
        low_arr = df["low"].to_numpy().astype(np.float64)
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

        atr = _compute_atr(high_arr, low_arr, close, 14)

        # Rolling ATR percentile (60-bar window)
        atr_pctile = np.full(n, 50.0, dtype=np.float64)
        for i in range(59, n):
            window = atr[i - 59:i + 1]
            if np.max(window) > 0:
                rank = np.sum(window <= atr[i]) / 60.0 * 100.0
                atr_pctile[i] = rank

        is_low_vol = atr_pctile < low_pctile
        is_high_vol = atr_pctile > high_pctile

        # BB for low-vol regime (1.5 sigma)
        sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        sma20 = np.nan_to_num(sma20, nan=0.0)
        std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        std20 = np.nan_to_num(std20, nan=1e10)
        bb_upper = sma20 + 1.5 * std20
        bb_lower = sma20 - 1.5 * std20

        # EMA for high-vol regime
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)

        # Confirmation
        uptick = np.zeros(n, dtype=np.bool_)
        downtick = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            uptick[i] = close[i] > close[i - 1]
            downtick[i] = close[i] < close[i - 1]

        # Volume filter (stricter for high-vol)
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok_low = volume > 0  # no strict filter for low vol
        vol_ok_high = volume > 2.0 * avg_vol

        # Regime stability: same regime for 10 bars
        regime_stable = np.zeros(n, dtype=np.bool_)
        for i in range(10, n):
            all_low = all(atr_pctile[i - j] < low_pctile for j in range(10))
            all_high = all(atr_pctile[i - j] > high_pctile for j in range(10))
            regime_stable[i] = all_low or all_high

        time_ok = (time_mins >= 565) & (time_mins <= 910)

        # Low-vol long: BB reversion
        low_long = is_low_vol & (close < bb_lower) & (close < vwap) & \
                   uptick & regime_stable & time_ok
        # High-vol long: EMA momentum
        high_long = is_high_vol & (ema9 > ema21) & (close > vwap) & \
                    vol_ok_high & regime_stable & time_ok

        # Low-vol short
        low_short = is_low_vol & (close > bb_upper) & (close > vwap) & \
                    downtick & regime_stable & time_ok
        # High-vol short
        high_short = is_high_vol & (ema9 < ema21) & (close < vwap) & \
                     vol_ok_high & regime_stable & time_ok

        long_entry = low_long | high_long
        short_entry = low_short | high_short

        # Signal exit: regime transitions to medium vol
        is_med_vol = (~is_low_vol) & (~is_high_vol)
        sig_exit_long = is_med_vol
        sig_exit_short = is_med_vol

        avg_stop = (low_stop + high_stop) / 2.0
        avg_target = (low_target + high_target) / 2.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=sma20.copy(),
            stop_loss_pct=avg_stop,
            target_pct=avg_target,
            trailing_stop_pct=0.002,
            trailing_activate_pct=0.003,
            time_stop_bars=75,
        )
