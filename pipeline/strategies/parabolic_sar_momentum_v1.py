"""
parabolic_sar_momentum_v1 — Parabolic SAR flip momentum on NIFTY 5-second bars.

Mechanism: On NIFTY, Parabolic SAR flips (bear→bull or bull→bear) confirmed by a
3-minute ADX > 20 and VWAP alignment signal institutional order-flow regime changes.
VWAP-benchmarked algorithms pivot their execution direction on these flips, driving
10-20 spot-point continuation over 30-90 seconds. The anti-whipsaw filter (< 4 SAR
flips in prior 3 min) ensures signals only fire in genuine trending conditions.

Adapted from: Strategy_171.json (1-min equity Parabolic SAR + ADX + VWAP on NIFTY200)
Hold time: 15-120 seconds (time_stop_bars=24)
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_psar(
    high: np.ndarray,
    low: np.ndarray,
    af_step: float = 0.02,
    af_max: float = 0.20,
) -> np.ndarray:
    """Compute Parabolic SAR direction array (+1 bull, -1 bear).

    Returns int8 array of +1 (SAR below price, bullish) or -1 (SAR above price, bearish).
    """
    n = len(high)
    direction = np.ones(n, dtype=np.int8)  # default bull
    sar = np.zeros(n)

    if n < 2:
        return direction

    # Initialise from first bar
    sar[0] = low[0]
    ep = high[0]
    af = af_step
    direction[0] = 1

    for i in range(1, n):
        prev_dir = direction[i - 1]
        prev_sar = sar[i - 1]

        if prev_dir == 1:
            # Bullish phase — SAR trails below price
            new_sar = prev_sar + af * (ep - prev_sar)
            # SAR may not exceed the two prior lows
            new_sar = min(new_sar, low[i - 1])
            if i >= 2:
                new_sar = min(new_sar, low[i - 2])

            if low[i] < new_sar:
                # Flip to bearish
                direction[i] = -1
                sar[i] = ep
                ep = low[i]
                af = af_step
            else:
                direction[i] = 1
                sar[i] = new_sar
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            # Bearish phase — SAR trails above price
            new_sar = prev_sar + af * (ep - prev_sar)
            # SAR may not fall below the two prior highs
            new_sar = max(new_sar, high[i - 1])
            if i >= 2:
                new_sar = max(new_sar, high[i - 2])

            if high[i] > new_sar:
                # Flip to bullish
                direction[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_step
            else:
                direction[i] = -1
                sar[i] = new_sar
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    return direction


def _wilder_smooth(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder (RMA) smoothing: seed = sum of first `period` bars, then EMA alpha=1/period."""
    n = len(arr)
    result = np.zeros(n)
    if n < period:
        return result
    result[period - 1] = np.sum(arr[:period])
    for i in range(period, n):
        result[i] = result[i - 1] - result[i - 1] / period + arr[i]
    return result


def _compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 36,
) -> np.ndarray:
    """Compute Wilder ADX. Returns array length n; values are 0 before warmup."""
    n = len(close)
    tr = np.zeros(n)
    dm_pos = np.zeros(n)
    dm_neg = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        dm_pos[i] = up if (up > dn and up > 0.0) else 0.0
        dm_neg[i] = dn if (dn > up and dn > 0.0) else 0.0

    atr_s = _wilder_smooth(tr, period)
    sdm_pos = _wilder_smooth(dm_pos, period)
    sdm_neg = _wilder_smooth(dm_neg, period)

    dx = np.zeros(n)
    for i in range(period - 1, n):
        if atr_s[i] > 0.0:
            di_pos = 100.0 * sdm_pos[i] / atr_s[i]
            di_neg = 100.0 * sdm_neg[i] / atr_s[i]
            denom = di_pos + di_neg
            if denom > 0.0:
                dx[i] = 100.0 * abs(di_pos - di_neg) / denom

    adx = _wilder_smooth(dx, period)
    return adx


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP, reset each day."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -1

    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        v = float(volume[i]) if volume[i] > 0 else 0.0
        cum_pv += close[i] * v
        cum_v += v
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]

    return vwap


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean; returns 0 before first full window."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1 : i + 1])
    return result


def _flip_count(direction: np.ndarray, window: int) -> np.ndarray:
    """Count of SAR direction flips in last `window` bars (inclusive)."""
    n = len(direction)
    flips = np.zeros(n, dtype=bool)
    for i in range(1, n):
        flips[i] = direction[i] != direction[i - 1]

    counts = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        start = max(0, i - window + 1)
        counts[i] = int(np.sum(flips[start : i + 1]))
    return counts


class Strategy(BaseStrategy):
    name = "parabolic_sar_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — 2 min past open for SAR init
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 300            # 25 min warmup: ADX needs 2×36=72 bars to seed + ADX smooth

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_threshold", 20.0, 15.0, 30.0),
            TunableParam("vol_mult", 0.8, 0.5, 1.5),
            TunableParam("max_flips", 3.0, 2.0, 5.0),
            TunableParam("stop_pts", 4.0, 2.5, 7.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays ──────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].fill_null(0).to_numpy().astype(np.int32)
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy().astype(np.int32)

        # ── Parameters ──────────────────────────────────────────────────
        adx_threshold = float(params.get("adx_threshold", 20.0))
        vol_mult = float(params.get("vol_mult", 0.8))
        max_flips = int(params.get("max_flips", 3))
        stop_pts = float(params.get("stop_pts", 4.0))
        target_pts = float(params.get("target_pts", 7.0))

        # ── Indicators ──────────────────────────────────────────────────
        psar_dir = _compute_psar(high, low, af_step=0.02, af_max=0.20)
        adx = _compute_adx(high, low, close, period=36)
        vwap = _compute_vwap(close, volume, day_id)
        vol_avg = _rolling_mean(volume, window=60)
        flips = _flip_count(psar_dir, window=36)

        # ── Detect SAR flips (current bar vs previous) ──────────────────
        just_flipped_bull = np.zeros(n, dtype=bool)
        just_flipped_bear = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if psar_dir[i] == 1 and psar_dir[i - 1] == -1:
                just_flipped_bull[i] = True
            elif psar_dir[i] == -1 and psar_dir[i - 1] == 1:
                just_flipped_bear[i] = True

        # ── Filters ─────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        trending = adx > adx_threshold
        not_choppy = flips < max_flips
        vol_ok = (vol_avg > 0.0) & (volume >= vol_avg * vol_mult)

        # ── Signals ─────────────────────────────────────────────────────
        buy_ce = (
            in_session
            & just_flipped_bull
            & trending
            & (close > vwap)
            & not_choppy
            & vol_ok
        )
        buy_pe = (
            in_session
            & just_flipped_bear
            & trending
            & (close < vwap)
            & not_choppy
            & vol_ok
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
