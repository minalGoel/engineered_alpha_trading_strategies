"""EMA Crossover Momentum — NIFTY 5-second index options.

Mechanism:
    On NIFTY, when the 1-minute EMA (12 bars) crosses above the 3-minute EMA
    (36 bars), it marks the moment short-term institutional buying has overtaken
    the recent medium-term trend average — the microstructure footprint of FII/DII
    TWAP algorithms that are ~50% through a directional order. ADX(36) > 20 ensures
    the cross occurs in a directed regime rather than a noise-dominated oscillation,
    and VWAP alignment confirms the broader institutional bias direction.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


# ── Indicator helpers ──────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Standard EMA with k = 2 / (period + 1)."""
    n = len(arr)
    out = np.zeros(n)
    if n == 0:
        return out
    k = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def _wilder_avg(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothed average: seed = simple mean of first `period` values."""
    n = len(arr)
    out = np.zeros(n)
    if n < period:
        return out
    out[period - 1] = np.mean(arr[:period])
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + arr[i]) / period
    return out


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average Directional Index (Wilder, 1978)."""
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if (up > dn and up > 0.0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0.0) else 0.0

    atr_w = _wilder_avg(tr, period)
    pdm_w = _wilder_avg(plus_dm, period)
    mdm_w = _wilder_avg(minus_dm, period)

    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = atr_w > 1e-9
    plus_di[mask] = 100.0 * pdm_w[mask] / atr_w[mask]
    minus_di[mask] = 100.0 * mdm_w[mask] / atr_w[mask]

    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.zeros(n)
    nz = di_sum > 1e-9
    dx[nz] = 100.0 * di_diff[nz] / di_sum[nz]

    return _wilder_avg(dx, period)


def _session_vwap(spot_df: pl.DataFrame) -> np.ndarray:
    """Cumulative session VWAP (typical price × volume), reset each day."""
    n = len(spot_df)
    high = spot_df["high"].fill_null(strategy="forward").to_numpy()
    low = spot_df["low"].fill_null(strategy="forward").to_numpy()
    close_arr = spot_df["close"].fill_null(strategy="forward").to_numpy()
    volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
    day_id = spot_df["day_id"].to_numpy()

    typical = (high + low + close_arr) / 3.0
    vwap = np.zeros(n)
    cum_tpv = 0.0
    cum_vol = 0.0
    prev_day = day_id[0] - 1  # sentinel to force reset on first bar

    for i in range(n):
        if day_id[i] != prev_day:
            cum_tpv = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        cum_tpv += typical[i] * volume[i]
        cum_vol += volume[i]
        vwap[i] = cum_tpv / cum_vol if cum_vol > 0.0 else typical[i]

    return vwap


# ── Strategy ───────────────────────────────────────────────────────────────────

class Strategy(BaseStrategy):
    name = "ema_crossover_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip opening 15-min noise
    session_end_minutes = 915     # 15:15 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10-min warmup; ADX(36) needs ~72 bars, EMA(36) ~72 bars

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_threshold", 20.0, 15.0, 30.0),
            TunableParam("stop_pts",       5.0,   2.0,  8.0),
            TunableParam("target_pts",     8.0,   4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (forward-fill NaN in Polars before numpy) ──────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high  = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low   = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        adx_threshold = params.get("adx_threshold", 20.0)
        stop_pts      = params.get("stop_pts",       5.0)
        target_pts    = params.get("target_pts",     8.0)

        # ── Indicators ────────────────────────────────────────────────────────
        # EMA(12): 1-min fast EMA — spans approx one hold period (60s)
        # EMA(36): 3-min slow EMA — 3× fast; captures prior institutional trend leg
        ema12 = _ema(close, 12)
        ema36 = _ema(close, 36)

        # ADX(36): 3-min directional strength — same window as slow EMA so trend
        # confirmation and signal reference the same timescale
        adx36 = _adx(high, low, close, 36)

        # Session VWAP — cumulative, no scaling needed
        vwap = _session_vwap(spot_df)

        # ── Masks ─────────────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # Fresh EMA crossovers only (state change, not persistence)
        cross_above = np.zeros(n, dtype=bool)
        cross_below = np.zeros(n, dtype=bool)
        cross_above[1:] = (ema12[1:] > ema36[1:]) & (ema12[:-1] <= ema36[:-1])
        cross_below[1:] = (ema12[1:] < ema36[1:]) & (ema12[:-1] >= ema36[:-1])

        trending = adx36 > adx_threshold

        # buy_ce: EMA bullish cross + trending + price above VWAP
        buy_ce = in_session & cross_above & trending & (close > vwap)

        # buy_pe: EMA bearish cross + trending + price below VWAP
        buy_pe = in_session & cross_below & trending & (close < vwap)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,                  # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
