"""Vol-of-Vol Signal v1 — cursor_opus46max_082

Thesis: Vol-of-vol (rolling_std of ATR) detects regime transitions.
Low vol-of-vol => stable regime => mean-reversion.
High vol-of-vol => transitioning => breakout via EMA cross.
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


def _ema(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = arr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss > 0:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss > 0:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i + 1] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_082"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vov_z_stable", default=1.0, low=0.5, high=1.5),
            TunableParam("vov_z_volatile", default=2.0, low=1.5, high=3.0),
            TunableParam("bb_pct_long", default=0.05, low=0.0, high=0.15),
            TunableParam("bb_pct_short", default=0.95, low=0.85, high=1.0),
            TunableParam("rsi_long", default=28.0, low=20.0, high=35.0),
            TunableParam("rsi_short", default=72.0, low=65.0, high=80.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_atr_mult", default=1.5, low=1.0, high=2.5),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vov_z_stable = params.get("vov_z_stable", 1.0)
        vov_z_volatile = params.get("vov_z_volatile", 2.0)
        bb_pct_long = params.get("bb_pct_long", 0.05)
        bb_pct_short = params.get("bb_pct_short", 0.95)
        rsi_long_thresh = params.get("rsi_long", 28.0)
        rsi_short_thresh = params.get("rsi_short", 72.0)
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_atr_mult = params.get("target_atr_mult", 1.5)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── Vol-of-vol: rolling_std(ATR, 30) ──
        vov = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            vov[i] = np.std(atr[i - 30:i])

        # ── VOV z-score over 120 bars ──
        vov_z = np.zeros(n, dtype=np.float64)
        for i in range(120, n):
            window = vov[i - 120:i + 1]
            mu = np.mean(window)
            sd = np.std(window)
            if sd > 0:
                vov_z[i] = (vov[i] - mu) / sd

        stable = vov_z < vov_z_stable
        transitioning = (vov_z >= vov_z_stable) & (vov_z < vov_z_volatile)
        # volatile = vov_z >= vov_z_volatile  # no trade

        # ── BB %B ──
        sma20 = np.zeros(n, dtype=np.float64)
        std20 = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            sma20[i] = np.mean(close[i - 20:i])
            std20[i] = np.std(close[i - 20:i])
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20
        bb_range = bb_upper - bb_lower
        bb_pct_b = np.where(bb_range > 0, (close - bb_lower) / bb_range, 0.5)

        # ── EMA(8) and EMA(21) ──
        ema8 = _ema(close, 8)
        ema21 = _ema(close, 21)

        # ── EMA cross detection ──
        ema8_above = ema8 > ema21
        ema8_cross_up = np.zeros(n, dtype=np.bool_)
        ema8_cross_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if ema8_above[i] and not ema8_above[i - 1]:
                ema8_cross_up[i] = True
            if not ema8_above[i] and ema8_above[i - 1]:
                ema8_cross_down[i] = True

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_surge = volume > 1.5 * avg_vol

        vix_ok = vix < 25.0

        # ── Stable regime: mean reversion entries ──
        stable_long = stable & (bb_pct_b < bb_pct_long) & (rsi < rsi_long_thresh) & vix_ok
        stable_short = stable & (bb_pct_b > bb_pct_short) & (rsi > rsi_short_thresh) & vix_ok

        # ── Transitioning regime: breakout entries ──
        trans_long = transitioning & ema8_cross_up & vol_surge & vix_ok
        trans_short = transitioning & ema8_cross_down & vol_surge & vix_ok

        long_entry = stable_long | trans_long
        short_entry = stable_short | trans_short

        # ── Signal exit: regime changes to volatile ──
        volatile_regime = vov_z >= vov_z_volatile
        signal_exit_long = volatile_regime
        signal_exit_short = volatile_regime.copy()

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=sma20.copy(),
            stop_loss_pct=stop_pct,
            target_atr_mult=target_atr_mult,
            time_stop_bars=45,
        )
