"""Keltner Channel Trend — claude_project_16_of_25

Thesis: When ADX confirms a strong trend and price breaks above/below
Keltner Channels with VWAP alignment, ride the trend with ATR-based exits.
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
        ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]
    return ema


def _compute_adx(high, low, close, period=14):
    """Compute ADX from high/low/close arrays."""
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    if n < period * 2:
        return adx

    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0

    # Smoothed TR, +DM, -DM using Wilder's smoothing
    smooth_tr = np.zeros(n, dtype=np.float64)
    smooth_plus = np.zeros(n, dtype=np.float64)
    smooth_minus = np.zeros(n, dtype=np.float64)

    smooth_tr[period] = np.sum(tr[1:period + 1])
    smooth_plus[period] = np.sum(plus_dm[1:period + 1])
    smooth_minus[period] = np.sum(minus_dm[1:period + 1])

    for i in range(period + 1, n):
        smooth_tr[i] = smooth_tr[i - 1] - smooth_tr[i - 1] / period + tr[i]
        smooth_plus[i] = smooth_plus[i - 1] - smooth_plus[i - 1] / period + plus_dm[i]
        smooth_minus[i] = smooth_minus[i - 1] - smooth_minus[i - 1] / period + minus_dm[i]

    # +DI, -DI, DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if smooth_tr[i] < 1e-10:
            continue
        plus_di = 100.0 * smooth_plus[i] / smooth_tr[i]
        minus_di = 100.0 * smooth_minus[i] / smooth_tr[i]
        di_sum = plus_di + minus_di
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di - minus_di) / di_sum

    # ADX = smoothed DX
    start = 2 * period
    if start < n:
        adx[start] = np.mean(dx[period:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


class Strategy(BaseStrategy):
    name = "keltner_channel_trend_v1"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("kc_atr_mult", default=2.0, low=1.0, high=3.0),
            TunableParam("adx_thresh", default=25.0, low=15.0, high=35.0),
            TunableParam("target_atr_mult", default=3.0, low=1.5, high=5.0),
            TunableParam("trailing_atr_mult", default=2.0, low=1.0, high=3.5),
            TunableParam("vix_low", default=13.0, low=8.0, high=18.0),
            TunableParam("vix_high", default=25.0, low=20.0, high=32.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        kc_atr_mult = params.get("kc_atr_mult", 2.0)
        adx_thresh = params.get("adx_thresh", 25.0)
        target_atr_mult = params.get("target_atr_mult", 3.0)
        trailing_atr_mult = params.get("trailing_atr_mult", 2.0)
        vix_low = params.get("vix_low", 13.0)
        vix_high = params.get("vix_high", 25.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── EMA(20) and ATR(20) ──
        ema20 = _compute_ema(close, 20)
        atr = _compute_atr(high, low, close, 20)

        # ── Keltner Channels ──
        kc_upper = ema20 + kc_atr_mult * atr
        kc_lower = ema20 - kc_atr_mult * atr

        # ── ADX(14) ──
        adx = _compute_adx(high, low, close, 14)

        # ── VIX filter ──
        vix_ok = (vix >= vix_low) & (vix <= vix_high)

        # ── Entry ──
        long_entry = (adx > adx_thresh) & (close > kc_upper) & (close > vwap) & vix_ok
        short_entry = (adx > adx_thresh) & (close < kc_lower) & (close < vwap) & vix_ok

        # ── Signal exit: close crosses KC mid (EMA20) ──
        signal_exit_long = close < ema20
        signal_exit_short = close > ema20

        # ── Trailing stop from ATR ──
        median_atr = np.nanmedian(atr[atr > 0]) if np.any(atr > 0) else 0.0
        median_close = np.nanmedian(close[close > 0]) if np.any(close > 0) else 1.0
        trailing_pct = (trailing_atr_mult * median_atr / median_close) if median_close > 0 else 0.005

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=ema20.copy(),
            target_atr_mult=target_atr_mult,
            trailing_stop_pct=trailing_pct,
            trailing_activate_pct=0.0,
            time_stop_bars=240,
        )
