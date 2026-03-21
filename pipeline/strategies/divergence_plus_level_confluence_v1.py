"""Divergence + Level Confluence Strategy (5-second NIFTY index options).

RSI divergence at structural levels (session VWAP, PDH, PDL, round numbers) on NIFTY.
When momentum exhaustion coincides with institutional order clustering at key levels,
the probability of a directional reversal increases.

Bullish divergence (price lower low, RSI higher low) at PDL/VWAP-below/round support
→ buy CE (expect bounce of 15-25 NIFTY spot points → 7-12 option premium points).

Bearish divergence (price higher high, RSI lower high) at PDH/VWAP-above/round resistance
→ buy PE (expect sell-off of 15-25 NIFTY spot points → 7-12 option premium points).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI. Returns array length n; fills with 50.0 (neutral) before warmup."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gains[1: period + 1])
    avg_loss[period] = np.mean(losses[1: period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain[i] / avg_loss[i]))
        else:
            rsi[i] = 100.0
    return rsi


def _compute_session_vwap(
    close: np.ndarray, volume: np.ndarray, day_id: np.ndarray
) -> np.ndarray:
    """Cumulative intraday VWAP, resets at each session open."""
    n = len(close)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_vol = 0.0
    cur_day = -99999
    for i in range(n):
        if day_id[i] != cur_day:
            cur_day = day_id[i]
            cum_pv = 0.0
            cum_vol = 0.0
        v = max(volume[i], 1.0)
        cum_pv += close[i] * v
        cum_vol += v
        vwap[i] = cum_pv / cum_vol
    return vwap


def _compute_pdh_pdl(
    high: np.ndarray, low: np.ndarray, day_id: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Previous session high/low for each bar. NaN on the first session."""
    n = len(high)
    pdh = np.full(n, np.nan)
    pdl = np.full(n, np.nan)

    unique_days: list[int] = []
    day_start: dict[int, int] = {}
    day_end: dict[int, int] = {}
    prev_d = -99999
    for i in range(n):
        d = int(day_id[i])
        if d != prev_d:
            unique_days.append(d)
            day_start[d] = i
            if prev_d != -99999:
                day_end[prev_d] = i - 1
            prev_d = d
    if prev_d != -99999:
        day_end[prev_d] = n - 1

    day_h: dict[int, float] = {}
    day_l: dict[int, float] = {}
    for d in unique_days:
        s = day_start[d]
        e = day_end[d] + 1
        day_h[d] = float(np.max(high[s:e]))
        day_l[d] = float(np.min(low[s:e]))

    for j, d in enumerate(unique_days):
        if j > 0:
            pd_val = unique_days[j - 1]
            s = day_start[d]
            e = day_end[d] + 1
            pdh[s:e] = day_h[pd_val]
            pdl[s:e] = day_l[pd_val]

    return pdh, pdl


def _detect_divergence(
    close: np.ndarray,
    rsi: np.ndarray,
    window: int,
    rsi_low: float,
    rsi_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect RSI price divergence over a rolling window.

    Bullish: current close is a new window low but RSI is above its window low value
             AND current RSI < rsi_low  → selling momentum is exhausting.
    Bearish: current close is a new window high but RSI is below its window high value
             AND current RSI > rsi_high → buying momentum is exhausting.
    """
    n = len(close)
    bull_div = np.zeros(n, dtype=bool)
    bear_div = np.zeros(n, dtype=bool)
    for i in range(window, n):
        w_c = close[i - window: i]
        w_r = rsi[i - window: i]

        # Bullish divergence: price lower low, RSI higher low
        min_idx = int(np.argmin(w_c))
        if (
            close[i] < w_c[min_idx]
            and rsi[i] > w_r[min_idx]
            and rsi[i] < rsi_low
        ):
            bull_div[i] = True

        # Bearish divergence: price higher high, RSI lower high
        max_idx = int(np.argmax(w_c))
        if (
            close[i] > w_c[max_idx]
            and rsi[i] < w_r[max_idx]
            and rsi[i] > rsi_high
        ):
            bear_div[i] = True

    return bull_div, bear_div


def _at_level(
    close: np.ndarray,
    vwap: np.ndarray,
    pdh: np.ndarray,
    pdl: np.ndarray,
    proximity_pct: float,
    round_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect proximity to structural support/resistance.

    Support: near PDL, near VWAP when price is below VWAP, or near a round number.
    Resistance: near PDH, near VWAP when price is above VWAP, or near a round number.
    """
    n = len(close)
    at_support = np.zeros(n, dtype=bool)
    at_resistance = np.zeros(n, dtype=bool)

    for i in range(n):
        c = close[i]
        prox = proximity_pct * c
        nearest = round(c / round_step) * round_step
        at_round = abs(c - nearest) < prox

        v = vwap[i]
        at_vwap = abs(c - v) < prox
        below_vwap = c < v
        above_vwap = c > v

        ph = pdh[i]
        pl_ = pdl[i]
        at_pdh = (not np.isnan(ph)) and abs(c - ph) < prox
        at_pdl = (not np.isnan(pl_)) and abs(c - pl_) < prox

        at_support[i] = at_pdl or (at_vwap and below_vwap) or at_round
        at_resistance[i] = at_pdh or (at_vwap and above_vwap) or at_round

    return at_support, at_resistance


class Strategy(BaseStrategy):
    """RSI divergence + structural level confluence on NIFTY.

    Bullish divergence (price lower low, RSI higher low) at PDL/VWAP/round support
    signals absorption of selling pressure by resting institutional buy orders.
    Bearish divergence at PDH/VWAP/round resistance signals exhaustion of buy flow.
    """

    name = "divergence_plus_level_confluence_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min pre-open noise
    session_end_minutes = 920     # 15:20 IST — avoid EOD option distortion
    max_trades_per_day = 4
    max_lookback = 120            # 10 min: covers RSI(36) warmup + 60-bar div window

    # NIFTY major round number step (100-pt psychological levels: 22000, 22100, …)
    _round_step = 100.0

    # Fixed indicator periods (not tunable — determines the strategy's time horizon)
    _RSI_PERIOD = 36       # 3-min RSI
    _DIV_WINDOW = 60       # 5-min divergence lookback

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_low", 42.0, 30.0, 50.0),        # RSI ceiling for bullish div
            TunableParam("rsi_high", 58.0, 50.0, 70.0),       # RSI floor for bearish div
            TunableParam("proximity_pct", 0.0012, 0.0005, 0.0025),  # level proximity
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Extract arrays — forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].fill_null(0).to_numpy().astype(int)
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy().astype(int)

        # Parameters
        rsi_low = float(params.get("rsi_low", 42.0))
        rsi_high = float(params.get("rsi_high", 58.0))
        proximity_pct = float(params.get("proximity_pct", 0.0012))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # ── Indicators ────────────────────────────────────────────────────────────
        rsi = _compute_rsi(close, self._RSI_PERIOD)
        vwap = _compute_session_vwap(close, volume, day_id)
        pdh, pdl = _compute_pdh_pdl(high, low, day_id)

        # ── Divergence (5-min window) ─────────────────────────────────────────────
        bull_div, bear_div = _detect_divergence(
            close, rsi, self._DIV_WINDOW, rsi_low, rsi_high
        )

        # ── Key level proximity ───────────────────────────────────────────────────
        at_support, at_resistance = _at_level(
            close, vwap, pdh, pdl, proximity_pct, self._round_step
        )

        # ── Session filter ────────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        buy_ce = in_session & bull_div & at_support
        buy_pe = in_session & bear_div & at_resistance

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,               # max 120s hold (24 × 5s)
            max_trades_per_day=self.max_trades_per_day,
        )
