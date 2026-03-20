"""Harmonic Pattern (ABCD) — cursor_opus46max_141

Thesis: AB=CD patterns on 1-min charts identify institutional VWAP-execution
waves. When D completes at a Fibonacci extension, the move is exhausted and
mean-reversion begins. Uses zigzag pivots + Fibonacci ratio validation.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
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
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_141"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 915     # 15:15
    max_trades_per_day = 8
    assumptions = [
        "Zigzag pivot detection simplified to local min/max over lookback window",
    ]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zigzag_pct", default=0.1, low=0.05, high=0.2),
            TunableParam("abcd_ratio_lo", default=0.836, low=0.7, high=0.95),
            TunableParam("abcd_ratio_hi", default=1.322, low=1.1, high=1.7),
            TunableParam("rsi_long_thresh", default=30.0, low=20.0, high=40.0),
            TunableParam("rsi_short_thresh", default=70.0, low=60.0, high=80.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zz_pct = params.get("zigzag_pct", 0.1) / 100.0
        ratio_lo = params.get("abcd_ratio_lo", 0.836)
        ratio_hi = params.get("abcd_ratio_hi", 1.322)
        rsi_long = params.get("rsi_long_thresh", 30.0)
        rsi_short = params.get("rsi_short_thresh", 70.0)
        vix_max = params.get("vix_max", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        atr = _compute_atr(high, low, close, 14)
        rsi = _compute_rsi(close, 14)

        # Simplified zigzag: detect local highs/lows using 3-bar window
        pivots = np.zeros(n, dtype=np.int8)  # +1=high, -1=low, 0=none
        for i in range(2, n - 2):
            if high[i] >= high[i-1] and high[i] >= high[i-2] and high[i] >= high[i+1] and high[i] >= high[i+2]:
                pivots[i] = 1
            if low[i] <= low[i-1] and low[i] <= low[i-2] and low[i] <= low[i+1] and low[i] <= low[i+2]:
                pivots[i] = -1

        # Find ABCD patterns: look back for 4 alternating pivots
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        for i in range(10, n):
            if not (time_mins[i] >= 565 and time_mins[i] <= 900):
                continue
            if vix[i] > vix_max:
                continue

            # Collect recent pivots (max 30 bars back)
            recent_pivots = []
            for j in range(max(0, i - 30), i):
                if pivots[j] != 0:
                    recent_pivots.append((j, pivots[j]))

            if len(recent_pivots) < 4:
                continue

            # Take last 4 pivots
            pts = recent_pivots[-4:]
            # Check alternating: high-low-high-low or low-high-low-high
            types = [p[1] for p in pts]
            if not (types[0] == -types[1] and types[1] == -types[2] and types[2] == -types[3]):
                continue

            a_idx, a_type = pts[0]
            b_idx, _ = pts[1]
            c_idx, _ = pts[2]
            d_idx, _ = pts[3]

            a_price = high[a_idx] if a_type == 1 else low[a_idx]
            b_price = low[b_idx] if a_type == 1 else high[b_idx]
            c_price = high[c_idx] if a_type == 1 else low[c_idx]
            d_price = low[d_idx] if a_type == 1 else high[d_idx]

            ab = abs(b_price - a_price)
            cd = abs(d_price - c_price)
            if ab < 1e-10:
                continue

            ratio = cd / ab
            bc_ret = abs(c_price - b_price) / ab if ab > 1e-10 else 0.0

            if ratio_lo <= ratio <= ratio_hi and 0.382 <= bc_ret <= 0.886:
                # Bearish ABCD (D is bottom) -> long
                if a_type == 1 and d_price <= close[i] and rsi[i] < rsi_long + 20 and rsi[i] > rsi[i-1]:
                    long_entry[i] = True
                # Bullish ABCD (D is top) -> short
                elif a_type == -1 and d_price >= close[i] and rsi[i] > rsi_short - 20 and rsi[i] < rsi[i-1]:
                    short_entry[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=1.0,
            target_pct=0.002,
            trailing_stop_pct=0.0005,
            trailing_activate_pct=0.001,
            time_stop_bars=20,
        )
