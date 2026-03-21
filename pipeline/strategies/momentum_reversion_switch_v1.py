"""Momentum-Reversion Regime Switch — NIFTY 5-second options strategy.

Uses ADX (5-min window) to classify the current NIFTY microstructure regime:
- Trending (ADX > threshold): trade EMA crossovers in VWAP direction
- Range-bound (ADX < threshold): fade Bollinger Band 2σ extremes back toward midline

The classifier is the edge: it prevents momentum signals from firing in chop and
reversion signals from firing during sustained institutional directional flow.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _compute_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Standard EMA with alpha = 2/(period+1). NaN for warmup bars."""
    n = len(arr)
    out = np.full(n, np.nan)
    alpha = 2.0 / (period + 1)
    # Seed with first non-nan value
    start = 0
    while start < n and np.isnan(arr[start]):
        start += 1
    if start >= n:
        return out
    out[start] = arr[start]
    for i in range(start + 1, n):
        val = arr[i] if not np.isnan(arr[i]) else out[i - 1]
        out[i] = alpha * val + (1.0 - alpha) * out[i - 1]
    return out


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int) -> np.ndarray:
    """ADX via Wilder's smoothing.  Returns 0.0 for warmup bars (<2*period)."""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        h_diff = high[i] - high[i - 1]
        l_diff = low[i - 1] - low[i]
        plus_dm[i] = h_diff if (h_diff > l_diff and h_diff > 0.0) else 0.0
        minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0.0) else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # Wilder smoothed ATR, +DM sum, -DM sum
    atr = np.zeros(n)
    s_pdm = np.zeros(n)
    s_mdm = np.zeros(n)

    if n <= period:
        return np.zeros(n)

    # Seed with simple sum of first `period` bars
    atr[period] = np.sum(tr[1: period + 1])
    s_pdm[period] = np.sum(plus_dm[1: period + 1])
    s_mdm[period] = np.sum(minus_dm[1: period + 1])

    for i in range(period + 1, n):
        atr[i] = atr[i - 1] - atr[i - 1] / period + tr[i]
        s_pdm[i] = s_pdm[i - 1] - s_pdm[i - 1] / period + plus_dm[i]
        s_mdm[i] = s_mdm[i - 1] - s_mdm[i - 1] / period + minus_dm[i]

    # +DI, -DI, DX
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)

    for i in range(period, n):
        if atr[i] > 0.0:
            plus_di[i] = 100.0 * s_pdm[i] / atr[i]
            minus_di[i] = 100.0 * s_mdm[i] / atr[i]
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0.0:
                dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX = Wilder-smoothed DX, seeded at 2*period
    adx = np.zeros(n)
    if n <= 2 * period:
        return adx

    adx[2 * period] = np.mean(dx[period: 2 * period + 1])
    for i in range(2 * period + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx


def _compute_vwap(close: np.ndarray, volume: np.ndarray,
                  day_id: np.ndarray) -> np.ndarray:
    """Session VWAP, reset per day."""
    n = len(close)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -9999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_v += volume[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


def _compute_bollinger(close: np.ndarray, period: int, n_std: float):
    """Bollinger Bands (upper, mid, lower). NaN during warmup."""
    n = len(close)
    upper = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(period - 1, n):
        w = close[i - period + 1: i + 1]
        m = np.mean(w)
        s = np.std(w)
        mid[i] = m
        upper[i] = m + n_std * s
        lower[i] = m - n_std * s
    return upper, mid, lower


# ── Strategy class ────────────────────────────────────────────────────────────

class Strategy(BaseStrategy):
    """ADX-gated regime switch between EMA-crossover momentum and BB reversion."""

    name = "momentum_reversion_switch_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip pre-open noise
    session_end_minutes = 920     # 15:20 IST — EOD flatten
    max_trades_per_day = 8
    # ADX needs 2*period = 120 bars for valid values; add margin for BB (60 bars)
    max_lookback = 240            # 20-minute warmup

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_trend_threshold", 25.0, 18.0, 32.0),
            TunableParam("adx_range_threshold", 18.0, 10.0, 24.0),
            TunableParam("bb_std_mult", 2.0, 1.5, 2.5),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN in Polars first) ────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        adx_trend = params.get("adx_trend_threshold", 25.0)
        adx_range = params.get("adx_range_threshold", 18.0)
        bb_std = params.get("bb_std_mult", 2.0)

        # ── Indicators ────────────────────────────────────────────────────────
        # ADX(60) — 5-minute regime classifier
        adx = _compute_adx(high, low, close, 60)

        # EMA crossover: fast=18 (1.5 min), slow=42 (3.5 min) — trend trigger
        ema_fast = _compute_ema(close, 18)
        ema_slow = _compute_ema(close, 42)

        # VWAP — directional filter
        vwap = _compute_vwap(close, volume, day_id)

        # Bollinger Bands(60, 2σ) — 5-minute range boundaries for reversion
        bb_upper, bb_mid, bb_lower = _compute_bollinger(close, 60, bb_std)

        # ── NaN / warmup handling ─────────────────────────────────────────────
        # Replace EMA NaN with 0 — prevents cross signals during warmup
        ema_fast = np.nan_to_num(ema_fast, nan=0.0)
        ema_slow = np.nan_to_num(ema_slow, nan=0.0)

        # BB validity mask — no reversion signal when bands are not yet computed
        bb_valid = ~np.isnan(bb_upper) & ~np.isnan(bb_lower)

        # ADX validity — 0.0 during warmup (2*period=120 bars); guard against
        # range_regime firing on the dead-zero warmup period
        adx_valid = adx > 0.0

        # ── EMA crossover detection ───────────────────────────────────────────
        # Bullish cross: ema_fast just crossed above ema_slow
        ema_cross_up = np.zeros(n, dtype=bool)
        ema_cross_dn = np.zeros(n, dtype=bool)
        ema_cross_up[1:] = (
            (ema_fast[1:] > ema_slow[1:]) & (ema_fast[:-1] <= ema_slow[:-1])
        )
        ema_cross_dn[1:] = (
            (ema_fast[1:] < ema_slow[1:]) & (ema_fast[:-1] >= ema_slow[:-1])
        )

        # ── Regime masks ─────────────────────────────────────────────────────
        trend_regime = adx_valid & (adx > adx_trend)
        range_regime = adx_valid & (adx < adx_range)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        # Momentum regime: EMA crossover in VWAP direction
        buy_ce_momentum = in_session & trend_regime & ema_cross_up & (close > vwap)
        buy_pe_momentum = in_session & trend_regime & ema_cross_dn & (close < vwap)

        # Reversion regime: price at BB extreme, confirmed by VWAP side
        buy_ce_reversion = (
            in_session & range_regime & bb_valid
            & (close < bb_lower) & (close < vwap)
        )
        buy_pe_reversion = (
            in_session & range_regime & bb_valid
            & (close > bb_upper) & (close > vwap)
        )

        buy_ce = buy_ce_momentum | buy_ce_reversion
        buy_pe = buy_pe_momentum | buy_pe_reversion

        # ── Per-bar stops / targets differentiated by regime ──────────────────
        # Momentum: 4 stop / 7 target — larger move expected from trend continuation
        # Reversion: 3 stop / 5 target — smaller, faster bounce from BB overshoot
        stop_pts = np.where(trend_regime, 4.0, 3.0)
        target_pts = np.where(trend_regime, 7.0, 5.0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_pts.astype(np.float64),
            target_points=target_pts.astype(np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
