"""pullback_to_vwap_trend_ride — NIFTY 5-second index options strategy.

When NIFTY is in a strong intraday trend (ADX > 25, DI lines aligned), institutional
TWAP/VWAP algorithms periodically pause their execution, letting price drift back to
session VWAP. At the VWAP touch, VWAP-benchmarked algos see favorable fills and
re-enter aggressively, creating a 10-20 second re-acceleration burst. This strategy
enters on the 5s bar where price re-touches VWAP within a confirmed trend, catching
the re-acceleration before a 1-minute bar could confirm it.

Hold: 30-90 seconds (6-18 bars).  Stop: 4 pts.  Target: 7 pts.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_vwap(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                  volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP, resets at each new day_id."""
    n = len(close)
    typical = (high + low + close) / 3.0
    vwap = np.zeros(n)
    cum_tv = 0.0
    cum_v = 0.0
    prev_day = -999999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_tv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        v = volume[i] if volume[i] > 0 else 1.0
        cum_tv += typical[i] * v
        cum_v += v
        vwap[i] = cum_tv / cum_v
    return vwap


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wilder's ADX, +DI, -DI with given period."""
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        h, lo, ph, pl_, pc = high[i], low[i], high[i - 1], low[i - 1], close[i - 1]
        tr[i] = max(h - lo, abs(h - pc), abs(lo - pc))
        up = h - ph
        dn = pl_ - lo
        plus_dm[i] = up if (up > dn and up > 0.0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0.0) else 0.0

    # Wilder's initial smoothed values (sum of first `period` bars)
    smoothed_tr = np.zeros(n)
    smoothed_pdm = np.zeros(n)
    smoothed_mdm = np.zeros(n)

    if n > period:
        smoothed_tr[period] = np.sum(tr[1: period + 1])
        smoothed_pdm[period] = np.sum(plus_dm[1: period + 1])
        smoothed_mdm[period] = np.sum(minus_dm[1: period + 1])
        inv = 1.0 / period
        for i in range(period + 1, n):
            smoothed_tr[i] = smoothed_tr[i - 1] - smoothed_tr[i - 1] * inv + tr[i]
            smoothed_pdm[i] = smoothed_pdm[i - 1] - smoothed_pdm[i - 1] * inv + plus_dm[i]
            smoothed_mdm[i] = smoothed_mdm[i - 1] - smoothed_mdm[i - 1] * inv + minus_dm[i]

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)

    for i in range(period, n):
        if smoothed_tr[i] > 0:
            plus_di[i] = 100.0 * smoothed_pdm[i] / smoothed_tr[i]
            minus_di[i] = 100.0 * smoothed_mdm[i] / smoothed_tr[i]
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum

    # ADX = Wilder's EMA of DX over same period
    adx = np.zeros(n)
    start = 2 * period
    if n > start:
        adx[start] = np.mean(dx[period: start + 1])
        inv = 1.0 / period
        for i in range(start + 1, n):
            adx[i] = adx[i - 1] - adx[i - 1] * inv + dx[i]

    return adx, plus_di, minus_di


class Strategy(BaseStrategy):
    name = "pullback_to_vwap_trend_ride"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — allow 15 min for VWAP/ADX warmup
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 180            # 15-min warmup: ADX(60) needs 120 bars + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_threshold", 25.0, 18.0, 35.0),
            TunableParam("vwap_dist_threshold", 0.0015, 0.0005, 0.003),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df: pl.DataFrame, option_df: pl.DataFrame,
                vix_df: pl.DataFrame, params: dict) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # --- VWAP (session-cumulative, no scaling needed) ---
        vwap = _compute_vwap(close, high, low, volume, day_id)

        # --- ADX(60) = 5-minute trend context ---
        adx_period = 60
        adx, plus_di, minus_di = _compute_adx(high, low, close, adx_period)

        # --- Parameters ---
        adx_thr = params.get("adx_threshold", 25.0)
        dist_thr = params.get("vwap_dist_threshold", 0.0015)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # --- Distance from VWAP (relative, element-wise) ---
        dist_to_vwap = np.where(close > 0, np.abs(close - vwap) / close, 1.0)

        # --- Single-bar directional confirmation ---
        bullish_bar = close > open_
        bearish_bar = close < open_

        # --- Session gate ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # --- Trend conditions ---
        trending = adx > adx_thr
        bull_trend = plus_di > minus_di
        bear_trend = minus_di > plus_di

        # --- Entry signals ---
        # BUY CE: uptrend + price touches VWAP + bullish confirmation bar
        buy_ce = in_session & trending & bull_trend & (dist_to_vwap < dist_thr) & bullish_bar

        # BUY PE: downtrend + price touches VWAP + bearish confirmation bar
        buy_pe = in_session & trending & bear_trend & (dist_to_vwap < dist_thr) & bearish_bar

        # Mutual exclusion (edge case: ADX high but DI indeterminate)
        both = buy_ce & buy_pe
        buy_ce = buy_ce & ~both
        buy_pe = buy_pe & ~both

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
