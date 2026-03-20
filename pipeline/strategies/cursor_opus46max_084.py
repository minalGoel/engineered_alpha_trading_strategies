"""Overnight Vol Carry v1 — cursor_opus46max_084

Thesis: After abnormally large overnight gaps (>1.5x normal), intraday
session mean-reverts. After abnormally small gaps (<0.3x normal), intraday
session trends (pent-up energy). Session type drives entry logic.
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
    name = "cursor_opus46max_084"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 910     # 15:10
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("mean_rev_ratio", default=1.5, low=1.2, high=2.0),
            TunableParam("momentum_ratio", default=0.3, low=0.1, high=0.5),
            TunableParam("gap_rsi_mr", default=35.0, low=25.0, high=45.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("target_atr_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("vix_max", default=22.0, low=18.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        mr_ratio = params.get("mean_rev_ratio", 1.5)
        mom_ratio = params.get("momentum_ratio", 0.3)
        gap_rsi_mr = params.get("gap_rsi_mr", 35.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        target_atr_mult = params.get("target_atr_mult", 1.5)
        vix_max = params.get("vix_max", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_id = df["day_id"].to_numpy()

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Compute gap pct and session type per day ──
        # First bar open as day open, prev day last bar close as prev_close
        prev_close = np.zeros(n, dtype=np.float64)
        day_open = np.zeros(n, dtype=np.float64)
        gap_pct = np.zeros(n, dtype=np.float64)
        overnight_abs = np.zeros(n, dtype=np.float64)

        last_close_prev = close[0]
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                prev_close[i] = last_close_prev
                day_open[i] = opn[i]
                if last_close_prev > 0:
                    gap_pct[i] = (opn[i] - last_close_prev) / last_close_prev * 100.0
                    overnight_abs[i] = abs(gap_pct[i])
            else:
                prev_close[i] = prev_close[i - 1]
                day_open[i] = day_open[i - 1]
                gap_pct[i] = gap_pct[i - 1]
                overnight_abs[i] = overnight_abs[i - 1]
            if i + 1 < n and day_id[i] != day_id[i + 1]:
                last_close_prev = close[i]
            elif i == n - 1:
                last_close_prev = close[i]

        # ── Average overnight gap (rolling, per day's first bar) ──
        # Use EMA-like approach: track running avg over days
        avg_overnight = np.zeros(n, dtype=np.float64)
        avg_val = 0.5  # initial
        day_count = 0
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                day_count += 1
                if day_count > 1:
                    avg_val = (avg_val * 19.0 + overnight_abs[i]) / 20.0
            avg_overnight[i] = max(avg_val, 0.01)

        overnight_ratio = overnight_abs / avg_overnight

        # ── Session type ──
        is_mean_rev = overnight_ratio > mr_ratio
        is_momentum = overnight_ratio < mom_ratio
        # neutral = neither

        # ── EMA(9), EMA(21) ──
        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        vix_ok = vix < vix_max

        # ── 30-bar high/low for momentum confirmation ──
        bar_high_30 = np.zeros(n, dtype=np.float64)
        bar_low_30 = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            bar_high_30[i] = np.max(close[i - 30:i])
            bar_low_30[i] = np.min(close[i - 30:i])

        # ── Mean reversion: fade the gap ──
        mr_long = is_mean_rev & (gap_pct < -0.5) & (rsi < gap_rsi_mr) & (close > vwap) & vix_ok
        mr_short = is_mean_rev & (gap_pct > 0.5) & (rsi > (100 - gap_rsi_mr)) & (close < vwap) & vix_ok

        # ── Momentum: ride pent-up energy ──
        mom_long = is_momentum & (ema9 > ema21) & (close >= bar_high_30) & vix_ok
        mom_short = is_momentum & (ema9 < ema21) & (close <= bar_low_30) & vix_ok

        long_entry = mr_long | mom_long
        short_entry = mr_short | mom_short

        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            target_atr_mult=target_atr_mult,
            breakeven_pct=0.003,
            time_stop_bars=180,
        )
