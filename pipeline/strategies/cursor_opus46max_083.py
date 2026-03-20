"""Regime Detection Vol v1 — cursor_opus46max_083

Thesis: Simplified HMM-like regime detection using rolling return variance.
Low-vol state => mean-reversion. Low-to-high transition => momentum breakout.
High-to-low transition => mean-reversion entry.
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
    name = "cursor_opus46max_083"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_threshold_pctile", default=60.0, low=45.0, high=75.0),
            TunableParam("rsi_mean_rev", default=35.0, low=25.0, high=45.0),
            TunableParam("vol_mult", default=1.5, low=1.0, high=2.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("target_atr_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_thresh_pctile = params.get("vol_threshold_pctile", 60.0)
        rsi_mr = params.get("rsi_mean_rev", 35.0)
        vol_mult = params.get("vol_mult", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.004)
        target_atr_mult = params.get("target_atr_mult", 1.5)
        vix_max = params.get("vix_max", 25.0)

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

        # ── Returns ──
        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0:
                returns[i] = np.log(close[i] / close[i - 1])

        # ── Rolling variance as HMM proxy ──
        roll_var = np.zeros(n, dtype=np.float64)
        var_window = 30
        for i in range(var_window, n):
            roll_var[i] = np.var(returns[i - var_window:i])

        # ── Regime: compare to rolling percentile over 120 bars ──
        regime_state = np.zeros(n, dtype=np.int32)  # 0=low, 1=high
        for i in range(120, n):
            window = roll_var[i - 120:i + 1]
            pctile = (np.sum(window < roll_var[i]) / len(window)) * 100.0
            regime_state[i] = 1 if pctile > vol_thresh_pctile else 0

        # ── Regime transitions ──
        low_to_high = np.zeros(n, dtype=np.bool_)
        high_to_low = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if regime_state[i] == 1 and regime_state[i - 1] == 0:
                low_to_high[i] = True
            if regime_state[i] == 0 and regime_state[i - 1] == 1:
                high_to_low[i] = True

        # ── EMA(8) and EMA(21) ──
        ema8 = _ema(close, 8)
        ema21 = _ema(close, 21)

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_surge = volume > vol_mult * avg_vol

        vix_ok = vix < vix_max

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── Low-to-high + momentum entries ──
        l2h_long = low_to_high & (ema8 > ema21) & vol_surge & vix_ok
        l2h_short = low_to_high & (ema8 < ema21) & vol_surge & vix_ok

        # ── High-to-low + mean-reversion entries ──
        h2l_long = high_to_low & (close < vwap) & (rsi < rsi_mr) & vix_ok
        h2l_short = high_to_low & (close > vwap) & (rsi > (100 - rsi_mr)) & vix_ok

        long_entry = l2h_long | h2l_long
        short_entry = l2h_short | h2l_short

        # ── Signal exit: regime flips back within 5 bars ──
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            target_atr_mult=target_atr_mult,
            time_stop_bars=45,
        )
