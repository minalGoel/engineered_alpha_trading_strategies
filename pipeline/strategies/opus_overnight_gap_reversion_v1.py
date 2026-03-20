"""Overnight Gap Reversion — Opus_6

Thesis: Small-to-medium opening gaps (0.3%-1%) tend to fill intraday.
Fade gap direction with RSI and VIX confirmation. Enter on first
confirming bar with above-average volume.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


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


class Strategy(BaseStrategy):
    name = "opus_overnight_gap_reversion_v1"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 645     # 10:45
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min", default=0.003, low=0.002, high=0.005),
            TunableParam("gap_max", default=0.01, low=0.007, high=0.015),
            TunableParam("rsi_long_thresh", default=40.0, low=25.0, high=50.0),
            TunableParam("rsi_short_thresh", default=60.0, low=50.0, high=75.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_min = params.get("gap_min", 0.003)
        gap_max = params.get("gap_max", 0.01)
        rsi_long = params.get("rsi_long_thresh", 40.0)
        rsi_short = params.get("rsi_short_thresh", 60.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        # ── Compute gap per day and prev_day_close ──
        gap_pct = np.zeros(n, dtype=np.float64)
        prev_day_close_arr = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)

        prev_close = np.nan
        for d in unique_days:
            day_mask = day_ids == d
            day_indices = np.where(day_mask)[0]
            if len(day_indices) == 0:
                continue
            day_open = open_[day_indices[0]]
            if not np.isnan(prev_close) and prev_close > 0:
                gap = (day_open - prev_close) / prev_close
                gap_pct[day_indices] = gap
                prev_day_close_arr[day_indices] = prev_close
            prev_close = close[day_indices[-1]]

        # ── Volume filter: first green/red bar with vol > 20-bar avg ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        vol_ok = volume > avg_vol_20

        # ── Bar direction ──
        bar_bullish = close > open_

        # ── Filters ──
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 555) & (time_mins <= 645)

        # ── Gap range filters ──
        gap_down_ok = (gap_pct >= -gap_max) & (gap_pct <= -gap_min)
        gap_up_ok = (gap_pct >= gap_min) & (gap_pct <= gap_max)

        # ── Entries ──
        long_entry = (gap_down_ok & (rsi < rsi_long) & vix_ok
                      & bar_bullish & vol_ok & time_ok)
        short_entry = (gap_up_ok & (rsi > rsi_short) & vix_ok
                       & (~bar_bullish) & vol_ok & time_ok)

        # ── Target: prev_day_close ──
        target_indicator = prev_day_close_arr.copy()

        # ── Breakeven after 50% fill: approximate as breakeven_pct = gap_size/2 ──
        # Use a fixed proxy since breakeven_pct is a scalar
        breakeven_pct = 0.003  # ~half of typical gap

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=target_indicator,
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            breakeven_pct=breakeven_pct,
            time_stop_bars=90,
        )
