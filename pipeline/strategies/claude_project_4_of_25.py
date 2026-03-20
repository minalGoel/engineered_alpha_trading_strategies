"""Intraday Momentum EMA Cross v1 — claude_project_4_of_25

Thesis: EMA(9)/EMA(21) crossover with ADX trend filter and VWAP alignment
captures intraday momentum moves.
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
    alpha = 2.0 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
    return ema


def _compute_adx(high, low, close, period=14):
    """Simplified ADX computation."""
    n = len(close)
    adx = np.full(n, 20.0, dtype=np.float64)
    if n < period * 2:
        return adx
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    # Smoothed using Wilder's method
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
    pdi = np.where(atr_s > 0, 100.0 * pdm_s / atr_s, 0.0)
    mdi = np.where(atr_s > 0, 100.0 * mdm_s / atr_s, 0.0)
    dx = np.where((pdi + mdi) > 0, 100.0 * np.abs(pdi - mdi) / (pdi + mdi), 0.0)
    # Smooth DX into ADX
    start = period * 2
    if start < n:
        adx[start] = np.mean(dx[period:start + 1]) if start > period else dx[start]
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


class Strategy(BaseStrategy):
    name = "intraday_momentum_ema_cross_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_thresh", default=20.0, low=12.0, high=30.0),
            TunableParam("vix_min", default=13.0, low=8.0, high=18.0),
            TunableParam("vix_max", default=28.0, low=20.0, high=40.0),
            TunableParam("target_atr_mult", default=2.0, low=1.0, high=4.0),
            TunableParam("stop_atr_mult", default=1.0, low=0.5, high=2.0),
            TunableParam("trail_atr_mult", default=1.5, low=0.8, high=3.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_thresh = params.get("adx_thresh", 20.0)
        vix_min = params.get("vix_min", 13.0)
        vix_max = params.get("vix_max", 28.0)
        target_atr = params.get("target_atr_mult", 2.0)
        stop_atr = params.get("stop_atr_mult", 1.0)
        trail_atr = params.get("trail_atr_mult", 1.5)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Indicators ──
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)
        atr = _compute_atr(high, low, close, 20)
        adx = _compute_adx(high, low, close, 14)

        # ── Filters ──
        vix_ok = (vix >= vix_min) & (vix <= vix_max)
        time_ok = time_mins <= 870  # no entry after 14:30

        # ── EMA cross detection ──
        ema_bull = ema9 > ema21
        ema_bear = ema9 < ema21

        # ── Entry ──
        long_entry = ema_bull & (adx > adx_thresh) & (close > vwap) & vix_ok & time_ok
        short_entry = ema_bear & (adx > adx_thresh) & (close < vwap) & vix_ok & time_ok

        # ── Signal exit: EMA cross against position ──
        signal_exit_long = ema_bear
        signal_exit_short = ema_bull

        # Convert trailing from ATR mult to pct using median ATR/close
        median_atr = np.median(atr[atr > 0]) if np.any(atr > 0) else 0.0
        median_close = np.median(close[close > 0]) if np.any(close > 0) else 1.0
        trailing_pct = trail_atr * median_atr / median_close if median_close > 0 else 0.005

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_atr_mult=target_atr,
            trailing_stop_pct=trailing_pct,
            time_stop_bars=240,
        )
