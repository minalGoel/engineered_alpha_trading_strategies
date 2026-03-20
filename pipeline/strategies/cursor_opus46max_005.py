"""VWAP Slope Trend v1 — cursor_opus46max_005

Thesis: Linear regression slope of VWAP over 20 bars reveals direction of
institutional order flow. Trade when slope accelerates in one direction,
confirmed by ADX > 20 and price above/below EMA(9).
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


def _linreg_slope(arr, period):
    """Rolling linear regression slope."""
    n = len(arr)
    slope = np.zeros(n, dtype=np.float64)
    x = np.arange(period, dtype=np.float64)
    x_mean = x.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if ss_xx < 1e-10:
        return slope
    for i in range(period - 1, n):
        window = arr[i - period + 1:i + 1]
        y_mean = np.mean(window)
        ss_xy = np.sum((x - x_mean) * (window - y_mean))
        slope[i] = ss_xy / ss_xx
    return slope


def _compute_adx(high, low, close, period):
    """Compute ADX, +DI, -DI."""
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return adx, plus_di, minus_di

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

    return adx, plus_di, minus_di


class Strategy(BaseStrategy):
    name = "cursor_opus46max_005"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 900     # 15:00
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_thresh", default=20.0, low=15.0, high=30.0),
            TunableParam("slope_confirm_bars", default=5.0, low=3.0, high=10.0),
            TunableParam("target_pct", default=0.005, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.005),
            TunableParam("trailing_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("trailing_activate", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_thresh = params.get("adx_thresh", 20.0)
        confirm_bars = int(params.get("slope_confirm_bars", 5))
        tgt_pct = params.get("target_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.003)
        trail_pct = params.get("trailing_pct", 0.002)
        trail_act = params.get("trailing_activate", 0.003)

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
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP slope ──
        vwap_slope = _linreg_slope(vwap, 20) * 1000.0
        vwap_slope_ma = np.zeros(n, dtype=np.float64)
        for i in range(9, n):
            vwap_slope_ma[i] = np.mean(vwap_slope[i - 9:i + 1])

        # ── EMA(9) ──
        ema9 = _compute_ema(close, 9)

        # ── ADX ──
        adx, plus_di, minus_di = _compute_adx(high, low, close, 14)
        atr = _compute_atr(high, low, close, 14)

        # ── Slope confirmation: positive/negative for N consecutive bars + increasing ──
        pos_confirm = np.zeros(n, dtype=np.bool_)
        neg_confirm = np.zeros(n, dtype=np.bool_)
        for i in range(confirm_bars + 2, n):
            all_pos = True
            all_neg = True
            for j in range(confirm_bars):
                if vwap_slope[i - j] <= 0:
                    all_pos = False
                if vwap_slope[i - j] >= 0:
                    all_neg = False
            increasing_3 = (vwap_slope[i] > vwap_slope[i - 1] > vwap_slope[i - 2])
            decreasing_3 = (vwap_slope[i] < vwap_slope[i - 1] < vwap_slope[i - 2])
            pos_confirm[i] = all_pos and increasing_3
            neg_confirm[i] = all_neg and decreasing_3

        # ── Filters ──
        time_ok = (time_mins >= 585) & (time_mins <= 900)

        # ── Entries ──
        long_entry = (
            (vwap_slope > 0)
            & (vwap_slope > vwap_slope_ma)
            & (close > vwap)
            & (close > ema9)
            & (adx > adx_thresh)
            & pos_confirm
            & time_ok
        )
        short_entry = (
            (vwap_slope < 0)
            & (vwap_slope < vwap_slope_ma)
            & (close < vwap)
            & (close < ema9)
            & (adx > adx_thresh)
            & neg_confirm
            & time_ok
        )

        # ── Signal exit: slope crosses zero ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if vwap_slope[i - 1] > 0 and vwap_slope[i] <= 0:
                sig_exit_long[i] = True
            if vwap_slope[i - 1] < 0 and vwap_slope[i] >= 0:
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
            time_stop_bars=60,
        )
