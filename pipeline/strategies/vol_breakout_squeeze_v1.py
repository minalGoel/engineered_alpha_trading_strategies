"""vol_breakout_squeeze_v1 — Dual-squeeze volatility breakout on NIFTY.

Detects when both ATR(12) is at its 5-minute minimum AND Bollinger Band width
is in the bottom 10th percentile of the past 5 minutes (dual squeeze). Enters
on the first expansion bar after the squeeze in the direction of that bar.

Adapted from equity Strategy_218: compressed from 14-min ATR / 2-hr minimum
on 1-min equity bars to 60s ATR / 5-min minimum on 5s NIFTY bars.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ATR. Returns array of length n; first (period-1) bars are 0."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        k = 1.0 / period  # Wilder smoothing factor
        for i in range(period, n):
            atr[i] = atr[i - 1] * (1.0 - k) + tr[i] * k
    return atr


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum over window bars; first (window-1) positions are 0."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window - 1, n):
        result[i] = np.min(arr[i - window + 1: i + 1])
    return result


def _rolling_sma(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average; first (window-1) positions are 0."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1: i + 1])
    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling population std dev; first (window-1) positions are 0."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window - 1, n):
        result[i] = np.std(arr[i - window + 1: i + 1])
    return result


def _rolling_percentile_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """Percentile rank (0-100) of current value within trailing window.

    Returns 0 for first (window-1) positions (not enough data yet).
    """
    n = len(arr)
    result = np.zeros(n)
    for i in range(window - 1, n):
        window_data = arr[i - window + 1: i + 1]
        result[i] = np.sum(window_data <= arr[i]) / window * 100.0
    return result


class Strategy(BaseStrategy):
    name = "vol_breakout_squeeze_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — allow 15 min for squeeze baseline to form
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 120            # 10-min warmup (120 × 5s); covers 60-bar rolling windows

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("atr_squeeze_pct",      1.05, 1.01, 1.15),  # ATR within X% of minimum
            TunableParam("bb_width_pctl_thresh", 10.0,  5.0, 20.0),  # BB width percentile threshold
            TunableParam("vol_ratio_thresh",      1.2,   1.0,  2.0),  # volume ratio at expansion bar
            TunableParam("squeeze_lookback",      5.0,   3.0, 10.0),  # bars to look back for recent squeeze
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before numpy conversion) ──
        high   = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low    = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        close  = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        open_  = spot_df["open"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──
        atr_squeeze_pct     = float(params.get("atr_squeeze_pct",     1.05))
        bb_width_pctl_thresh = float(params.get("bb_width_pctl_thresh", 10.0))
        vol_ratio_thresh    = float(params.get("vol_ratio_thresh",     1.2))
        squeeze_lookback    = int(params.get("squeeze_lookback",       5))

        # ── Indicator 1: ATR(12) = 60-second Wilder ATR ──
        # Compressed from original 14-min ATR to 60s to match our hold window.
        atr_12 = _compute_atr(high, low, close, 12)

        # ── Indicator 2: 5-min rolling minimum of ATR (60 bars) ──
        # Defines the "squeezed" ATR level. Shorter than original 2-hr minimum
        # because intraday squeezes on NIFTY 5s bars resolve in minutes, not hours.
        atr_min_60 = _rolling_min(atr_12, 60)

        # ── Indicator 3: Bollinger Band width as % of midpoint, BB(12) ──
        sma_12  = _rolling_sma(close, 12)
        std_12  = _rolling_std(close, 12)
        sma_safe = np.where(sma_12 > 0, sma_12, 1.0)
        bb_width = (4.0 * std_12) / sma_safe * 100.0  # (upper-lower)/mid*100

        # ── Indicator 4: 5-min percentile rank of BB width (60 bars) ──
        bb_width_pctl = _rolling_percentile_rank(bb_width, 60)

        # ── Indicator 5: Volume ratio vs 60s SMA ──
        vol_sma_12 = _rolling_sma(volume, 12)
        vol_sma_safe = np.where(vol_sma_12 > 0, vol_sma_12, 1.0)
        vol_ratio = volume / vol_sma_safe

        # ── Dual squeeze: ATR near 5-min minimum AND BB width in bottom decile ──
        atr_min_safe = np.where(atr_min_60 > 0, atr_min_60, 1e-9)
        in_squeeze = (
            (atr_12 <= atr_min_safe * atr_squeeze_pct) &
            (bb_width_pctl < bb_width_pctl_thresh) &
            (atr_min_60 > 0)  # require warmup: rolling min must be non-zero
        )

        # ── Recent squeeze: was in_squeeze True within the last squeeze_lookback bars? ──
        # We look at bars [i-squeeze_lookback, i) — not including current bar i.
        recent_squeeze = np.zeros(n, dtype=bool)
        for i in range(squeeze_lookback, n):
            if np.any(in_squeeze[i - squeeze_lookback: i]):
                recent_squeeze[i] = True

        # ── ATR expansion inflection: ATR increasing from prior bar ──
        atr_expanding = np.zeros(n, dtype=bool)
        atr_expanding[1:] = atr_12[1:] > atr_12[:-1]

        # ── Direction of expansion bar ──
        bullish_bar = close > open_
        bearish_bar = close < open_

        # ── Volume confirmation at expansion ──
        vol_confirming = vol_ratio > vol_ratio_thresh

        # ── Warmup guard: require 120 bars of data before firing ──
        has_warmup = np.zeros(n, dtype=bool)
        if n > 120:
            has_warmup[120:] = True

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Final signals ──
        base_condition = has_warmup & in_session & recent_squeeze & atr_expanding & vol_confirming

        buy_ce = base_condition & bullish_bar
        buy_pe = base_condition & bearish_bar

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop 3 pts: genuine NIFTY dual-squeeze expansions do not retrace 6 spot pts
            # within 15s; if they do the expansion is noise, not institutional flow.
            stop_points=np.full(n, 3.0),
            # Target 6 pts: captures ~60% of the 12-20 spot pt first-burst after dual squeeze.
            target_points=np.full(n, 6.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120s max hold
            max_trades_per_day=self.max_trades_per_day,
        )
