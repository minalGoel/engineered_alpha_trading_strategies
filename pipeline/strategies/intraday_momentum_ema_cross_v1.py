"""
intraday_momentum_ema_cross_v1 — EMA Crossover Momentum on NIFTY Index

Thesis: Institutional TWAP/VWAP algorithms executing large directional orders create
sustained micro-trends on NIFTY visible as EMA alignment across 1-min and 3-min windows.
A fresh EMA(12) crossover of EMA(36) with ADX > 20 and VWAP alignment signals
directionally persistent order flow — enter ATM call/put to ride 30-90s continuation.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


# ── Indicator helpers ─────────────────────────────────────────────────────────


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. Returns NaN for warmup bars."""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = np.mean(arr[:period])
    for i in range(period, n):
        result[i] = arr[i] * k + result[i - 1] * (1.0 - k)
    return result


def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing: sum seed then decay. Handles NaN in input by treating as 0."""
    n = len(arr)
    result = np.full(n, np.nan)
    k = 1.0 / period

    # Find first finite value index
    first_valid = -1
    for i in range(n):
        if np.isfinite(arr[i]):
            first_valid = i
            break
    if first_valid < 0:
        return result

    start = first_valid + period - 1
    if start >= n:
        return result

    # Seed: sum of first `period` valid values
    result[start] = 0.0
    count = 0
    idx = first_valid
    while count < period and idx < n:
        v = arr[idx] if np.isfinite(arr[idx]) else 0.0
        result[start] += v
        count += 1
        idx += 1

    for i in range(start + 1, n):
        v = arr[i] if np.isfinite(arr[i]) else 0.0
        result[i] = result[i - 1] * (1.0 - k) + v
    return result


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    ADX using Wilder smoothing.
    Returns array of ADX values; NaN during warmup (~2*period bars).
    """
    n = len(close)
    tr = np.full(n, np.nan)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i - 1])
        lpc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hpc, lpc)

        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0.0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0.0) else 0.0

    smooth_tr = _wilder_smooth(tr, period)
    smooth_plus = _wilder_smooth(plus_dm, period)
    smooth_minus = _wilder_smooth(minus_dm, period)

    plus_di = np.where(smooth_tr > 0, 100.0 * smooth_plus / smooth_tr, 0.0)
    minus_di = np.where(smooth_tr > 0, 100.0 * smooth_minus / smooth_tr, 0.0)

    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)

    adx = _wilder_smooth(dx, period)
    return adx


def _linreg_slope(arr: np.ndarray, period: int) -> np.ndarray:
    """Linear regression slope of last `period` bars. Returns 0 during warmup."""
    n = len(arr)
    result = np.zeros(n)
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_denom = ((x - x_mean) ** 2).sum()
    if x_denom == 0.0:
        return result
    for i in range(period - 1, n):
        y = arr[i - period + 1 : i + 1]
        if np.any(np.isnan(y)):
            continue
        result[i] = ((x - x_mean) * (y - y.mean())).sum() / x_denom
    return result


def _session_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP reset at the start of each day."""
    n = len(close)
    vwap = np.full(n, np.nan)
    cum_pv = 0.0
    cum_v = 0.0
    current_day = -1

    for i in range(n):
        if day_id[i] != current_day:
            current_day = day_id[i]
            cum_pv = 0.0
            cum_v = 0.0
        v = float(volume[i]) if volume[i] > 0 else 1.0
        cum_pv += close[i] * v
        cum_v += v
        vwap[i] = cum_pv / cum_v
    return vwap


# ── Strategy ──────────────────────────────────────────────────────────────────


class Strategy(BaseStrategy):
    """
    EMA(12/36) crossover momentum on NIFTY index.

    Entry: fresh EMA(12) cross of EMA(36) + ADX(36) > 20 + price vs VWAP + slope confirmation.
    Exit:  4 pt stop, 6 pt target, 18-bar (90s) time stop.
    """

    name = "intraday_momentum_ema_cross_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 144            # 144 × 5s = 12 min; covers 2× ADX(36) Wilder init

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_threshold", 20.0, 15.0, 30.0),
            TunableParam("vix_low",        12.0, 10.0, 16.0),
            TunableParam("vix_high",       28.0, 22.0, 35.0),
            TunableParam("stop_pts",        4.0,  2.0,  8.0),
            TunableParam("target_pts",      6.0,  4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before to_numpy) ────────────
        close  = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high   = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low    = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        adx_threshold = params.get("adx_threshold", 20.0)
        vix_low       = params.get("vix_low",        12.0)
        vix_high      = params.get("vix_high",       28.0)
        stop_pts      = params.get("stop_pts",         4.0)
        target_pts    = params.get("target_pts",       6.0)

        # ── Indicators ────────────────────────────────────────────────────────
        ema_fast_raw = _ema(close, 12)           # 1-min EMA — trade trigger
        ema_slow_raw = _ema(close, 36)           # 3-min EMA — context filter
        adx_raw      = _adx(high, low, close, 36)
        vwap_raw     = _session_vwap(close, volume, day_id)

        # Fill NaN with neutral values before computing derived indicators
        ema_fast = np.where(np.isnan(ema_fast_raw), close, ema_fast_raw)
        ema_slow = np.where(np.isnan(ema_slow_raw), close, ema_slow_raw)
        adx_vals = np.where(np.isnan(adx_raw),      0.0,   adx_raw)
        vwap_val = np.where(np.isnan(vwap_raw),     close, vwap_raw)

        # Slope of slow EMA over last 12 bars (1 min) — confirm slow EMA is turning
        slope_slow = _linreg_slope(ema_slow, 12)

        # ── VIX (aligned to spot bars) ────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Session and regime filters ────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        vix_ok    = (vix_close >= vix_low) & (vix_close <= vix_high)
        adx_strong = adx_vals > adx_threshold

        # ── EMA crossover detection (FRESH — crossover on THIS bar only) ──────
        ema_diff = ema_fast - ema_slow
        bullish_cross = np.zeros(n, dtype=bool)
        bearish_cross = np.zeros(n, dtype=bool)
        # Cross: sign flips between t-1 and t
        bullish_cross[1:] = (ema_diff[1:] > 0.0) & (ema_diff[:-1] <= 0.0)
        bearish_cross[1:] = (ema_diff[1:] < 0.0) & (ema_diff[:-1] >= 0.0)

        # Fast EMA must still be accelerating in cross direction
        fast_rising  = np.zeros(n, dtype=bool)
        fast_falling = np.zeros(n, dtype=bool)
        fast_rising[1:]  = ema_fast[1:] > ema_fast[:-1]
        fast_falling[1:] = ema_fast[1:] < ema_fast[:-1]

        # VWAP direction
        above_vwap = close > vwap_val
        below_vwap = close < vwap_val

        # Slow EMA slope direction
        slope_up   = slope_slow > 0.0
        slope_down = slope_slow < 0.0

        # ── Entry signals ─────────────────────────────────────────────────────
        base_filter = in_session & vix_ok & adx_strong

        buy_ce = base_filter & bullish_cross & fast_rising  & above_vwap & slope_up
        buy_pe = base_filter & bearish_cross & fast_falling & below_vwap & slope_down

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,             # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
