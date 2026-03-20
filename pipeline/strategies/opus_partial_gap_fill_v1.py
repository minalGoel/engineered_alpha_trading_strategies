"""Partial Gap Fill — Opus_30

Thesis: Medium gaps (0.5-1.5%) tend to partially fill. Fade the gap on
RSI extremes and first reversal bar, targeting a 50% fill.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


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
    """Wilder's RSI."""
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
    name = "opus_partial_gap_fill_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 630     # 10:30
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min", default=0.005, low=0.003, high=0.008),
            TunableParam("gap_max", default=0.015, low=0.01, high=0.025),
            TunableParam("rsi_long_thresh", default=30.0, low=20.0, high=40.0),
            TunableParam("rsi_short_thresh", default=70.0, low=60.0, high=80.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_min = params.get("gap_min", 0.005)
        gap_max = params.get("gap_max", 0.015)
        rsi_long_thresh = params.get("rsi_long_thresh", 30.0)
        rsi_short_thresh = params.get("rsi_short_thresh", 70.0)
        stop_pct = params.get("stop_loss_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        day_id = df["day_id"].to_numpy().astype(np.int64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Previous day close ──
        day_last_close = {}
        for i in range(n):
            day_last_close[day_id[i]] = close[i]

        prev_day_close = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d = day_id[i]
            if d - 1 in day_last_close:
                prev_day_close[i] = day_last_close[d - 1]

        # ── Per-day gap ──
        day_first_open = {}
        for i in range(n):
            d = day_id[i]
            if d not in day_first_open:
                day_first_open[d] = open_[i]

        day_gap = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d = day_id[i]
            if prev_day_close[i] > 1e-10:
                day_gap[i] = (day_first_open[d] - prev_day_close[i]) / prev_day_close[i]

        # ── RSI(7) ──
        rsi = _compute_rsi(close, 7)

        # ── First green / red bar detection ──
        green_bar = close > open_
        red_bar = close < open_

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Filters ──
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Gap down: long (fade the gap)
        gap_down = (day_gap < -gap_min) & (day_gap > -gap_max)
        # Gap up: short (fade the gap)
        gap_up = (day_gap > gap_min) & (day_gap < gap_max)

        # ── Entries ──
        long_entry = gap_down & (rsi < rsi_long_thresh) & green_bar & time_ok
        short_entry = gap_up & (rsi > rsi_short_thresh) & red_bar & time_ok

        # ── Target: 50% of gap fill → approximate ──
        abs_gap = np.abs(day_gap)
        median_gap = np.median(abs_gap[abs_gap > 0]) if np.any(abs_gap > 0) else 0.005
        target_pct = max(0.5 * median_gap, 0.002)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=75,
        )
