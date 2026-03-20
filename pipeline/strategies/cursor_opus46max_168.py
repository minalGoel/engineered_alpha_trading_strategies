"""Trend Strength Adaptive v1 — cursor_opus46max_168

Thesis: Use ADX as a continuous parameter modifier instead of a binary filter.
EMA lookback = max(5, 25 - floor(ADX/5)). Higher ADX = shorter lookback =
faster signals. +DI/-DI determines direction.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


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
    name = "cursor_opus46max_168"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("min_adx", default=15.0, low=10.0, high=20.0),
            TunableParam("adx_rising_bars", default=3.0, low=2.0, high=5.0),
            TunableParam("base_target_bps", default=30.0, low=20.0, high=50.0),
            TunableParam("base_stop_bps", default=20.0, low=12.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        min_adx = params.get("min_adx", 15.0)
        adx_bars = int(params.get("adx_rising_bars", 3.0))
        base_target = params.get("base_target_bps", 30.0)
        base_stop = params.get("base_stop_bps", 20.0)

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

        # Adaptive EMA lookback: max(5, 25 - floor(ADX/5))
        adaptive_lookback = np.clip(25 - np.floor(adx / 5.0), 5, 25).astype(np.int32)

        # Compute adaptive fast and slow EMAs bar-by-bar
        ema_fast = np.zeros(n, dtype=np.float64)
        ema_slow = np.zeros(n, dtype=np.float64)
        ema_fast[0] = close[0]
        ema_slow[0] = close[0]
        for i in range(1, n):
            lb = max(5, adaptive_lookback[i])
            k_fast = 2.0 / (lb + 1)
            k_slow = 2.0 / (2 * lb + 1)
            ema_fast[i] = close[i] * k_fast + ema_fast[i - 1] * (1.0 - k_fast)
            ema_slow[i] = close[i] * k_slow + ema_slow[i - 1] * (1.0 - k_slow)

        # ADX rising confirmation
        adx_rising = np.zeros(n, dtype=np.bool_)
        for i in range(adx_bars, n):
            if adx[i] > adx[i - adx_bars]:
                adx_rising[i] = True

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 570) & (time_mins <= 910)

        # Entry
        long_entry = (plus_di > minus_di) & (adx > min_adx) & \
                     (ema_fast > ema_slow) & (close > vwap) & \
                     adx_rising & vol_ok & time_ok

        short_entry = (minus_di > plus_di) & (adx > min_adx) & \
                      (ema_fast < ema_slow) & (close < vwap) & \
                      adx_rising & vol_ok & time_ok

        # Signal exit: +DI/-DI cross against position or ADX drops too low
        sig_exit_long = (minus_di > plus_di) | (adx < 12.0)
        sig_exit_short = (plus_di > minus_di) | (adx < 12.0)

        # ADX-adaptive target and stop (use max of base, ADX*1.5 for target)
        target_pct = max(base_target, 30.0 * 1.5) / 10000.0
        stop_pct = max(base_stop, 20.0 * 0.8) / 10000.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=target_pct * 0.4,
            trailing_activate_pct=target_pct * 0.6,
            time_stop_bars=90,
        )
