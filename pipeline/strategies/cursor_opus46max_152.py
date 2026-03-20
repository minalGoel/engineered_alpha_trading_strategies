"""Momentum Reversion Switch v1 — cursor_opus46max_152

Thesis: Markets alternate between momentum and mean-reversion regimes.
ADX(14) > 25 => trend-follow via EMA crossover. ADX(14) < 18 => BB
mean-reversion. ADX 18-25 dead zone => no trade.
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
    """Compute ADX, +DI, -DI using Wilder's smoothing."""
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

    # Wilder smoothing
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

    # ADX = Wilder smooth of DX
    start = 2 * period
    if start < n:
        adx[start] = np.mean(dx[period:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx, plus_di, minus_di


class Strategy(BaseStrategy):
    name = "cursor_opus46max_152"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_trend_thresh", default=25.0, low=20.0, high=35.0),
            TunableParam("adx_range_thresh", default=18.0, low=12.0, high=22.0),
            TunableParam("mom_stop_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("rev_stop_pct", default=0.0035, low=0.002, high=0.005),
            TunableParam("mom_target_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_trend = params.get("adx_trend_thresh", 25.0)
        adx_range = params.get("adx_range_thresh", 18.0)
        mom_stop = params.get("mom_stop_pct", 0.003)
        rev_stop = params.get("rev_stop_pct", 0.0035)
        mom_target = params.get("mom_target_pct", 0.005)

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

        # ADX
        adx, plus_di, minus_di = _compute_adx(high, low, close, 14)

        # EMAs
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)

        # Bollinger Bands
        sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        sma20 = np.nan_to_num(sma20, nan=0.0)
        std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        std20 = np.nan_to_num(std20, nan=1e10)
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20

        # Volume confirmation
        avg_vol = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 560) & (time_mins <= 900)

        # Regime detection
        is_momentum = adx > adx_trend
        is_reversion = adx < adx_range

        # EMA crossover detection
        ema_cross_up = np.zeros(n, dtype=np.bool_)
        ema_cross_dn = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if ema9[i - 1] <= ema21[i - 1] and ema9[i] > ema21[i]:
                ema_cross_up[i] = True
            if ema9[i - 1] >= ema21[i - 1] and ema9[i] < ema21[i]:
                ema_cross_dn[i] = True

        # Momentum long: EMA cross up + above VWAP
        mom_long = is_momentum & ema_cross_up & (close > vwap) & vol_ok & time_ok
        # Momentum short: EMA cross down + below VWAP
        mom_short = is_momentum & ema_cross_dn & (close < vwap) & vol_ok & time_ok
        # Reversion long: close below BB lower + below VWAP
        rev_long = is_reversion & (close < bb_lower) & (close < vwap) & vol_ok & time_ok
        # Reversion short: close above BB upper + above VWAP
        rev_short = is_reversion & (close > bb_upper) & (close > vwap) & vol_ok & time_ok

        long_entry = mom_long | rev_long
        short_entry = mom_short | rev_short

        # Signal exit: regime switch (ADX crosses threshold)
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # If was in a regime and now in dead zone or opposite
            was_mom = adx[i - 1] > adx_trend
            was_rev = adx[i - 1] < adx_range
            now_dead = (adx[i] >= adx_range) and (adx[i] <= adx_trend)
            if (was_mom or was_rev) and now_dead:
                sig_exit_long[i] = True
                sig_exit_short[i] = True

        # Use blended stop: average of momentum and reversion params
        avg_stop = (mom_stop + rev_stop) / 2.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=sma20.copy(),
            stop_loss_pct=avg_stop,
            target_pct=mom_target,
            trailing_stop_pct=0.002,
            trailing_activate_pct=0.003,
            time_stop_bars=90,
        )
