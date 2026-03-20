"""Kernel Density v1 — cursor_opus46max_193

Thesis: KDE on the last 120 bar closes identifies objective support/resistance
levels (density peaks). Buy near strong support with RSI confirmation,
sell near strong resistance. Simplified: use histogram-based mode detection.
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


def _find_kde_modes(data, n_bins=50):
    """Find modes (local maxima) using histogram approximation."""
    if len(data) < 5:
        return [], []
    lo, hi = np.min(data), np.max(data)
    if hi - lo < 1e-10:
        return [lo], [1.0]
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(data, bins=edges)
    centers = (edges[:-1] + edges[1:]) / 2

    # Smooth
    smooth = np.zeros_like(counts, dtype=np.float64)
    for i in range(len(counts)):
        lo_i = max(0, i - 2)
        hi_i = min(len(counts), i + 3)
        smooth[i] = np.mean(counts[lo_i:hi_i])

    # Find local maxima
    modes = []
    strengths = []
    mean_density = np.mean(smooth) if np.mean(smooth) > 0 else 1.0
    for i in range(1, len(smooth) - 1):
        if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1]:
            modes.append(centers[i])
            strengths.append(smooth[i] / mean_density)

    return modes, strengths


class Strategy(BaseStrategy):
    name = "cursor_opus46max_193"
    is_long_only = False
    session_start = 675   # ~11:15
    session_end = 910     # 15:10
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("kde_window", default=120.0, low=60.0, high=180.0),
            TunableParam("proximity_pct", default=0.1, low=0.05, high=0.2),
            TunableParam("mode_strength_min", default=1.5, low=1.0, high=2.5),
            TunableParam("rsi_long_thresh", default=40.0, low=25.0, high=45.0),
            TunableParam("rsi_short_thresh", default=60.0, low=55.0, high=75.0),
            TunableParam("stop_loss_pct", default=0.0015, low=0.001, high=0.003),
            TunableParam("target_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        kde_win = int(params.get("kde_window", 120.0))
        prox_pct = params.get("proximity_pct", 0.1)
        strength_min = params.get("mode_strength_min", 1.5)
        rsi_long = params.get("rsi_long_thresh", 40.0)
        rsi_short = params.get("rsi_short_thresh", 60.0)
        stop_pct = params.get("stop_loss_pct", 0.0015)
        target_pct = params.get("target_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        rsi = _compute_rsi(close, 14)

        # Find nearest support/resistance using KDE modes
        near_support = np.zeros(n, dtype=np.bool_)
        near_resistance = np.zeros(n, dtype=np.bool_)
        support_strong = np.zeros(n, dtype=np.bool_)
        resist_strong = np.zeros(n, dtype=np.bool_)

        for i in range(kde_win, n):
            seg = close[i-kde_win+1:i+1]
            modes, strengths = _find_kde_modes(seg)

            if len(modes) < 2:
                continue

            current = close[i]
            prox_dist = current * prox_pct / 100.0

            # Find nearest support (mode below current)
            supports = [(m, s) for m, s in zip(modes, strengths) if m < current]
            # Find nearest resistance (mode above current)
            resists = [(m, s) for m, s in zip(modes, strengths) if m > current]

            if supports:
                nearest_sup = max(supports, key=lambda x: x[0])
                if abs(current - nearest_sup[0]) < prox_dist and current > nearest_sup[0]:
                    near_support[i] = True
                    support_strong[i] = nearest_sup[1] > strength_min

            if resists:
                nearest_res = min(resists, key=lambda x: x[0])
                if abs(current - nearest_res[0]) < prox_dist and current < nearest_res[0]:
                    near_resistance[i] = True
                    resist_strong[i] = nearest_res[1] > strength_min

        # Bounce/rejection confirmation
        bounce = np.zeros(n, dtype=np.bool_)
        reject = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            bounce[i] = (close[i] > close[i-1]) and (close[i-1] > close[i-2])
            reject[i] = (close[i] < close[i-1]) and (close[i-1] < close[i-2])

        time_ok = (time_mins >= 675) & (time_mins <= 910)

        long_entry = (
            near_support & support_strong
            & (rsi < rsi_long)
            & bounce
            & time_ok
        )
        short_entry = (
            near_resistance & resist_strong
            & (rsi > rsi_short)
            & reject
            & time_ok
        )

        # Signal exit: price breaks through level
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        # Simple: exit when RSI normalizes
        sig_exit_long = rsi > 55
        sig_exit_short = rsi < 45

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_activate_pct=0.0020,
            trailing_stop_pct=0.0012,
            time_stop_bars=45,
        )
