"""trend_counter_trend_v1 — Asymmetric Supertrend Entry / RSI-Divergence Exit

Thesis: On NIFTY, Supertrend flips (2-min ATR) signal institutional directional
commitment. We enter trend direction and exit early via RSI divergence + VWAP
overshoot — capturing the burst while avoiding the mean-reversion tail.

Original: equity Supertrend(10,3) + RSI divergence exit on NIFTY50 stocks, 1-min bars.
Conversion: index options, 5s bars, compressed lookbacks to match 30-120s hold time.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_supertrend_direction(
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    period: int, multiplier: float,
) -> np.ndarray:
    """Return direction array: +1 = bullish, -1 = bearish."""
    n = len(close)
    atr = _compute_atr(high, low, close, period)
    hl2 = (high + low) / 2.0

    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    direction = np.full(n, -1, dtype=np.int8)

    for i in range(1, n):
        # Ratchet bands: upper only tightens, lower only rises
        if basic_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        if basic_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Direction: previous direction determines which band is active
        if direction[i - 1] == -1:  # bearish: close must break above upper to flip
            direction[i] = 1 if close[i] > final_upper[i] else -1
        else:  # bullish: close must break below lower to flip
            direction[i] = -1 if close[i] < final_lower[i] else 1

    return direction


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed RSI; returns 50.0 for bars without enough history."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    deltas = np.zeros(n)
    deltas[1:] = close[1:] - close[:-1]
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[1: period + 1])
    avg_loss = np.mean(losses[1: period + 1])

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return rsi


def _compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    ema = np.zeros(n)
    k = 2.0 / (period + 1)
    ema[0] = close[0]
    for i in range(1, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Cumulative VWAP, reset each day."""
    n = len(close)
    vwap = close.copy()
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -999999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = int(day_id[i])
        cum_pv += close[i] * volume[i]
        cum_v += volume[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


# ── Strategy ───────────────────────────────────────────────────────────────────

class Strategy(BaseStrategy):
    """Supertrend flip entry with RSI-divergence + VWAP-overshoot counter-trend exit."""

    name = "trend_counter_trend_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 120            # 10-min warmup for EMA(60) and Supertrend(24)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("st_multiplier", 3.0, 2.0, 4.0),
            TunableParam("vwap_dist_threshold", 15.0, 8.0, 30.0),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # Replace any remaining zeros in close/high/low with a forward fill fallback
        for arr in (close, high, low):
            for i in range(1, n):
                if arr[i] == 0.0:
                    arr[i] = arr[i - 1]

        # Parameters
        st_multiplier = float(params.get("st_multiplier", 3.0))
        vwap_threshold = float(params.get("vwap_dist_threshold", 15.0))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))
        div_lookback = 18  # 90 seconds — fixed, not tunable (period, not threshold)

        # ── Indicators ────────────────────────────────────────────────────────
        # Supertrend: ATR(24) = 2-min. Detects micro-trend flips on NIFTY.
        direction = _compute_supertrend_direction(high, low, close, 24, st_multiplier)

        # RSI(36) = 3-min. Detects divergence developing within the 30-120s hold window.
        rsi = _compute_rsi(close, 36)

        # EMA(60) = 5-min. Context filter for trend alignment.
        ema60 = _compute_ema(close, 60)

        # Cumulative VWAP — no lookback scaling needed.
        vwap = _compute_vwap(close, volume, day_id)
        vwap_dist = close - vwap  # signed deviation in spot points

        # ── Filters ───────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # Supertrend flip detection (single bar)
        flip_bull = np.zeros(n, dtype=bool)
        flip_bear = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if direction[i] == 1 and direction[i - 1] == -1:
                flip_bull[i] = True
            elif direction[i] == -1 and direction[i - 1] == 1:
                flip_bear[i] = True

        # RSI divergence over 18-bar (90s) window:
        # Bearish: price higher-high but RSI fails to confirm (lower by >3 pts)
        # Bullish: price lower-low but RSI fails to confirm (higher by >3 pts)
        bearish_div = np.zeros(n, dtype=bool)
        bullish_div = np.zeros(n, dtype=bool)
        for i in range(div_lookback, n):
            if close[i] > close[i - div_lookback] and rsi[i] < rsi[i - div_lookback] - 3.0:
                bearish_div[i] = True
            if close[i] < close[i - div_lookback] and rsi[i] > rsi[i - div_lookback] + 3.0:
                bullish_div[i] = True

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: Supertrend flips bullish + price above VWAP + 5-min EMA bullish
        #         + no bearish divergence (don't enter trend that's already diverging)
        buy_ce = (
            in_session
            & flip_bull
            & (close > vwap)
            & (close > ema60)
            & ~bearish_div
        )

        # buy_pe: Supertrend flips bearish + price below VWAP + 5-min EMA bearish
        #         + no bullish divergence
        buy_pe = (
            in_session
            & flip_bear
            & (close < vwap)
            & (close < ema60)
            & ~bullish_div
        )

        # ── Counter-trend exit signals ────────────────────────────────────────
        # sell_ce: bearish RSI divergence + NIFTY has overshot VWAP (VWAP algos fading)
        sell_ce = in_session & bearish_div & (vwap_dist > vwap_threshold)

        # sell_pe: bullish RSI divergence + NIFTY has undershot VWAP
        sell_pe = in_session & bullish_div & (vwap_dist < -vwap_threshold)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=sell_ce,
            sell_pe=sell_pe,
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
