"""VWAP Cross Momentum v1 — cursor_opus46max_003

Thesis: Price crossing above VWAP after 20+ bars below signals regime shift.
Volume surge on the cross bar + close in top 30% of range confirms.
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


class Strategy(BaseStrategy):
    name = "cursor_opus46max_003"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 885     # 14:45
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("dwell_bars", default=20.0, low=10.0, high=40.0),
            TunableParam("vol_surge_thresh", default=2.0, low=1.2, high=3.0),
            TunableParam("range_pos_thresh", default=0.7, low=0.5, high=0.85),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.005),
            TunableParam("trailing_pct", default=0.0015, low=0.0008, high=0.003),
            TunableParam("trailing_activate", default=0.0025, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        dwell = int(params.get("dwell_bars", 20))
        vol_surge = params.get("vol_surge_thresh", 2.0)
        rp_thresh = params.get("range_pos_thresh", 0.7)
        tgt_pct = params.get("target_pct", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.0025)
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

        # ── EMA(5) ──
        ema5 = _compute_ema(close, 5)

        # ── Volume surge ──
        avg_vol_10 = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol_10 = np.nan_to_num(avg_vol_10, nan=1.0)
        avg_vol_10 = np.clip(avg_vol_10, 1.0, None)
        vol_ratio = volume / avg_vol_10

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Consecutive bars below/above VWAP ──
        bars_below = np.zeros(n, dtype=np.int32)
        bars_above = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if close[i] < vwap[i]:
                bars_below[i] = (bars_below[i - 1] + 1) if i > 0 else 1
                bars_above[i] = 0
            elif close[i] > vwap[i]:
                bars_above[i] = (bars_above[i - 1] + 1) if i > 0 else 1
                bars_below[i] = 0
            else:
                bars_below[i] = 0
                bars_above[i] = 0

        # ── Bar range position ──
        bar_range = high - low
        bar_range = np.where(bar_range > 0, bar_range, 1e-10)
        range_pos = (close - low) / bar_range

        # ── Crossover detection ──
        cross_above = np.zeros(n, dtype=np.bool_)
        cross_below = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            cross_above[i] = (close[i] > vwap[i]) and (close[i - 1] < vwap[i - 1])
            cross_below[i] = (close[i] < vwap[i]) and (close[i - 1] > vwap[i - 1])

        # ── Dwell count on previous side before cross ──
        prev_bars_below = np.zeros(n, dtype=np.int32)
        prev_bars_above = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if cross_above[i]:
                prev_bars_below[i] = bars_below[i - 1]
            if cross_below[i]:
                prev_bars_above[i] = bars_above[i - 1]

        # ── EMA rising/falling ──
        ema_rising = np.zeros(n, dtype=np.bool_)
        ema_falling = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            ema_rising[i] = ema5[i] > ema5[i - 1]
            ema_falling[i] = ema5[i] < ema5[i - 1]

        # ── Time filter ──
        time_ok = (time_mins >= 585) & (time_mins <= 885)

        # ── Entries ──
        long_entry = (
            cross_above
            & (prev_bars_below >= dwell)
            & (vol_ratio > vol_surge)
            & ema_rising
            & (range_pos > rp_thresh)
            & time_ok
        )
        short_entry = (
            cross_below
            & (prev_bars_above >= dwell)
            & (vol_ratio > vol_surge)
            & ema_falling
            & (range_pos < (1.0 - rp_thresh))
            & time_ok
        )

        # ── Signal exit: re-cross VWAP ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if close[i] < vwap[i] and close[i - 1] >= vwap[i - 1]:
                sig_exit_long[i] = True
            if close[i] > vwap[i] and close[i - 1] <= vwap[i - 1]:
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
            time_stop_bars=30,
        )
