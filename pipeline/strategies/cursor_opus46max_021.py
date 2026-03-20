"""EMA Crossover Momentum v1 — cursor_opus46max_021

Thesis: EMA(9) crossing EMA(21) with ADX(14)>25 and VWAP alignment
captures intraday trend legs. Enter on cross bar with volume surge
and range position confirmation.
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


def _compute_ema(close, period):
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = close[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


def _compute_adx(high, low, close, period):
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return adx
    atr = _compute_atr(high, low, close, period)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    smooth_plus = np.zeros(n, dtype=np.float64)
    smooth_minus = np.zeros(n, dtype=np.float64)
    smooth_plus[period] = np.sum(plus_dm[1:period + 1])
    smooth_minus[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        smooth_plus[i] = smooth_plus[i - 1] - smooth_plus[i - 1] / period + plus_dm[i]
        smooth_minus[i] = smooth_minus[i - 1] - smooth_minus[i - 1] / period + minus_dm[i]
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * smooth_plus[i] / atr[i] / period
            minus_di[i] = 100.0 * smooth_minus[i] / atr[i] / period
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        denom = plus_di[i] + minus_di[i]
        if denom > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / denom
    if n >= 2 * period:
        adx[2 * period - 1] = np.mean(dx[period:2 * period])
        for i in range(2 * period, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


class Strategy(BaseStrategy):
    name = "cursor_opus46max_021"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_thresh", default=25.0, low=18.0, high=35.0),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.007),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("trailing_pct", default=0.0015, low=0.0008, high=0.003),
            TunableParam("trailing_activate", default=0.0025, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_thresh = params.get("adx_thresh", 25.0)
        tgt_pct = params.get("target_pct", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.002)
        trail_pct = params.get("trailing_pct", 0.0015)
        trail_act = params.get("trailing_activate", 0.0025)

        n = len(df)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── EMAs ──
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)

        # ── ADX ──
        adx = _compute_adx(high, low, close, 14)

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        avg_vol_10 = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol_10 = np.nan_to_num(avg_vol_10, nan=1.0)
        avg_vol_10 = np.clip(avg_vol_10, 1.0, None)
        vol_surge = volume > 1.2 * avg_vol_10

        # ── Bar range position ──
        bar_range = high - low
        bar_range = np.where(bar_range > 0, bar_range, 1e-10)
        range_pos = (close - low) / bar_range

        # ── Cross detection ──
        cross_up = np.zeros(n, dtype=np.bool_)
        cross_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if ema9[i] > ema21[i] and ema9[i - 1] <= ema21[i - 1]:
                cross_up[i] = True
            if ema9[i] < ema21[i] and ema9[i - 1] >= ema21[i - 1]:
                cross_down[i] = True

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 885)

        # ── Entries ──
        long_entry = (
            cross_up
            & (adx > adx_thresh)
            & (close > vwap)
            & vol_ok & vol_surge
            & (range_pos > 0.6)
            & time_ok
        )
        short_entry = (
            cross_down
            & (adx > adx_thresh)
            & (close < vwap)
            & vol_ok & vol_surge
            & (range_pos < 0.4)
            & time_ok
        )

        # ── Signal exit: reverse EMA cross ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if ema9[i] < ema21[i] and ema9[i - 1] >= ema21[i - 1]:
                sig_exit_long[i] = True
            if ema9[i] > ema21[i] and ema9[i - 1] <= ema21[i - 1]:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=45,
        )
