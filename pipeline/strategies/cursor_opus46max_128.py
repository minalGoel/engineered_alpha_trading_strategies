"""Monthly Settlement Day — cursor_opus46max_128

Thesis: Monthly F&O settlement creates predictable patterns: morning rollover
(range-bound, fade extremes) and afternoon settlement-fix (directional bias).
Proxied by high-volume + VWAP-based directional bias in the afternoon session.
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


def _compute_ema(close, period):
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = close[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = alpha * close[i] + (1 - alpha) * ema[i-1]
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_128"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 929     # 15:29
    max_trades_per_day = 4
    assumptions = [
        "Settlement calendar not available; strategy fires on all days",
        "Rollover/calendar-spread data not available; using VWAP + volume proxy",
    ]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=1.5, low=1.2, high=2.5),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("pm_target_bps", default=20.0, low=10.0, high=35.0),
            TunableParam("pm_stop_bps", default=12.0, low=6.0, high=20.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 1.5)
        vix_max = params.get("vix_max", 25.0)
        pm_target = params.get("pm_target_bps", 20.0)
        pm_stop = params.get("pm_stop_bps", 12.0)

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

        atr = _compute_atr(high, low, close, 14)

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        vix_ok = vix < vix_max

        # Settlement-fix phase: 14:00-15:25 (840-925 minutes)
        settle_time = (time_mins >= 840) & (time_mins <= 925)

        # Two consecutive up/down bars for confirmation
        up_bars = np.zeros(n, dtype=np.bool_)
        down_bars = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            up_bars[i] = close[i] > close[i-1] and close[i-1] > close[i-2]
            down_bars[i] = close[i] < close[i-1] and close[i-1] < close[i-2]

        long_entry = (close > vwap) & vol_ok & vix_ok & settle_time & up_bars
        short_entry = (close < vwap) & vol_ok & vix_ok & settle_time & down_bars

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=pm_stop / 10000.0,
            target_pct=pm_target / 10000.0,
            trailing_stop_pct=0.0008,
            trailing_activate_pct=0.001,
            time_stop_bars=60,
        )
