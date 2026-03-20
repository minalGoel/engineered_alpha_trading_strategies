"""Intraday Trend Following — Opus_11

Thesis: When a strong intraday trend is confirmed by EMA(20)>EMA(50) and
ADX(14)>25, pullbacks to EMA(20) offer high-probability continuation entries.
Requires close > VWAP for longs (below for shorts) as trend confirmation.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_ema(close, period):
    """EMA using standard multiplier."""
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = close[0]
    mult = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = close[i] * mult + ema[i - 1] * (1.0 - mult)
    return ema


def _compute_atr(high, low, close, period):
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


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


def _compute_adx(high, low, close, period):
    """ADX using Wilder's smoothing."""
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    if n < period * 2 + 1:
        return adx

    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))

    # +DM and -DM
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    # Wilder's smoothing for TR, +DM, -DM
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

    # +DI and -DI
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if smooth_tr[i] > 1e-10:
            plus_di[i] = 100.0 * smooth_plus[i] / smooth_tr[i]
            minus_di[i] = 100.0 * smooth_minus[i] / smooth_tr[i]

    # DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX: Wilder's smoothing of DX
    start = period * 2
    if start < n:
        adx[start] = np.mean(dx[period:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


class Strategy(BaseStrategy):
    name = "opus_intraday_trend_following_v1"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_thresh", default=25.0, low=18.0, high=35.0),
            TunableParam("rsi_long_max", default=65.0, low=55.0, high=75.0),
            TunableParam("rsi_short_min", default=35.0, low=25.0, high=45.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_thresh = params.get("adx_thresh", 25.0)
        rsi_long_max = params.get("rsi_long_max", 65.0)
        rsi_short_min = params.get("rsi_short_min", 35.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── EMAs ──
        ema20 = _compute_ema(close, 20)
        ema50 = _compute_ema(close, 50)

        # ── ADX(14) ──
        adx = _compute_adx(high, low, close, 14)

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Filters ──
        time_ok = (time_mins >= 585) & (time_mins <= 870)
        adx_ok = adx > adx_thresh

        # ── Long: uptrend pullback to EMA20 ──
        long_entry = ((ema20 > ema50)
                      & adx_ok
                      & (low <= ema20 * 1.001)
                      & (close > ema20)
                      & (rsi < rsi_long_max)
                      & (close > vwap)
                      & time_ok)

        # ── Short: downtrend pullback to EMA20 ──
        short_entry = ((ema20 < ema50)
                       & adx_ok
                       & (high >= ema20 * 0.999)
                       & (close < ema20)
                       & (rsi > rsi_short_min)
                       & (close < vwap)
                       & time_ok)

        # ── Trailing stop: activate after +1.0*ATR profit ──
        # Approximate trailing_activate_pct from median ATR/close ratio
        safe_close = np.clip(np.abs(close), 1e-10, None)
        atr_pct = atr / safe_close
        valid_atr_pct = atr_pct[atr_pct > 0]
        median_atr_pct = float(np.median(valid_atr_pct)) if len(valid_atr_pct) > 0 else 0.003
        trailing_pct = median_atr_pct      # ~1.0*ATR as pct
        trailing_act = median_atr_pct      # activate after +1.0*ATR

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_atr_mult=2.0,
            stop_loss_atr_mult=1.0,
            trailing_stop_pct=trailing_pct,
            trailing_activate_pct=trailing_act,
            time_stop_bars=120,
        )
