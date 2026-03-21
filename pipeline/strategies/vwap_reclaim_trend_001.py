"""
vwap_reclaim_trend_001 — VWAP Reclaim Trend Continuation on NIFTY

Mechanism:
On NIFTY, when the index has been trading below session VWAP for most of a 3-minute
window, every VWAP-benchmarked institutional algorithm has been accumulating passively
at better-than-benchmark prices. The moment price reclaims VWAP with the 3-min EMA
crossing back above the 15-min EMA, those accumulated long positions shift from passive
absorption to active continuation — creating a 20-35 spot point burst over the next
30-90 seconds. The bearish mirror applies when sustained above-VWAP drift reverses
through VWAP with EMA structure turning negative.

Original: vwap_reclaim_trend_001 (Strategy_138.json) — Nifty200 stocks, 1-min bars,
8-24 min hold. Converted to NIFTY index, 5-second bars, 15-90 second hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with forward-filled NaN."""
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1)
    # Seed with mean of first `period` bars (or all bars if fewer)
    seed_end = min(period, n)
    out[0] = arr[0]
    for i in range(1, seed_end):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    for i in range(seed_end, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rolling_count(cond: np.ndarray, window: int) -> np.ndarray:
    """Count of True values in the last `window` bars (inclusive)."""
    n = len(cond)
    out = np.zeros(n, dtype=np.float64)
    c = cond.astype(np.float64)
    running = 0.0
    for i in range(n):
        running += c[i]
        if i >= window:
            running -= c[i - window]
        out[i] = running
    return out


class Strategy(BaseStrategy):
    name = "vwap_reclaim_trend_001"
    underlying = "NIFTY"
    session_start_minutes = 575   # 09:35 IST — avoids opening range noise
    session_end_minutes = 810     # 13:30 IST — avoids afternoon chop
    max_trades_per_day = 4
    max_lookback = 180            # 15-min warmup for EMA_slow(180)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("dislocation_pct", 0.75, 0.60, 0.90),
            TunableParam("stop_pts", 5.0, 2.0, 7.0),
            TunableParam("target_pts", 8.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill nulls in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high  = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low   = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        vol   = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        day_id   = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        dislocation_pct = float(params.get("dislocation_pct", 0.75))
        stop_pts  = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # ── Session VWAP (cumulative, resets each day) ──
        vwap = np.zeros(n, dtype=np.float64)
        cum_pv  = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_pv  = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            typical = (high[i] + low[i] + close[i]) / 3.0
            v = vol[i] if vol[i] > 0.0 else 1.0
            cum_pv  += typical * v
            cum_vol += v
            vwap[i]  = cum_pv / cum_vol

        # ── EMA indicators ──
        # EMA_fast(36) = 3-min EMA — reclaim momentum trigger
        # EMA_slow(180) = 15-min EMA — session structural trend
        ema_fast = _ema(close, 36)
        ema_slow = _ema(close, 180)

        # ── Rolling VWAP dislocation counts (36 bars = 3 minutes) ──
        WINDOW = 36
        dislocation_thresh = dislocation_pct * WINDOW  # default 27/36

        below_vwap = close < vwap
        above_vwap = close > vwap
        below_count = _rolling_count(below_vwap, WINDOW)
        above_count = _rolling_count(above_vwap, WINDOW)

        # ── VWAP cross detection (one bar look-back) ──
        prev_close = np.empty(n, dtype=np.float64)
        prev_close[0] = close[0]
        prev_close[1:] = close[:-1]

        prev_vwap = np.empty(n, dtype=np.float64)
        prev_vwap[0] = vwap[0]
        prev_vwap[1:] = vwap[:-1]

        # Bullish reclaim: previous bar below VWAP, current bar >= VWAP
        vwap_reclaim   = (prev_close < prev_vwap) & (close >= vwap)
        # Bearish breakdown: previous bar above VWAP, current bar <= VWAP
        vwap_breakdown = (prev_close > prev_vwap) & (close <= vwap)

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry signals ──
        # BUY CE: NIFTY was below VWAP for 75%+ of last 3 min → now reclaims → EMA bullish
        buy_ce = (
            in_session
            & (below_count >= dislocation_thresh)
            & vwap_reclaim
            & (ema_fast > ema_slow)
        )

        # BUY PE: NIFTY was above VWAP for 75%+ of last 3 min → now breaks down → EMA bearish
        buy_pe = (
            in_session
            & (above_count >= dislocation_thresh)
            & vwap_breakdown
            & (ema_fast < ema_slow)
        )

        # Ensure no simultaneous signals
        buy_ce = buy_ce & ~buy_pe

        stop_arr   = np.where(buy_ce | buy_pe, stop_pts,   0.0)
        target_arr = np.where(buy_ce | buy_pe, target_pts, 0.0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_arr.astype(np.float64),
            target_points=target_arr.astype(np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,                    # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
