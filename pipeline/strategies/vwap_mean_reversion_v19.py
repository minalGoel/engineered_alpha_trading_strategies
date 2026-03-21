"""VWAP Mean Reversion v19 — Moderate VWAP deviation basing bounce on NIFTY.

Mechanism: On NIFTY, moderate 20-35 spot point deviations from session VWAP
(z-score ~-1.4 to -2.0) occur 5-10 times per session. Market makers short
gamma from weekly ATM option writing must delta-hedge by buying the index
within 30-60 seconds — this is mechanical. We detect impending hedge flow via
30-second bar range contraction (selling exhaustion) combined with the close
near the top of that compressed range (buyers absorbing sellers), entering
faster than v18's 2-minute dev_momentum confirmation.

Original: vwap_mean_reversion_v19 — nifty50 stocks, 1-min bars, z-score < -2.1
with rel_volume > 1.7, 10-30 bar hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_v19"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min for VWAP warmup
    session_end_minutes = 915     # 15:15 IST
    max_trades_per_day = 8
    max_lookback = 240            # 20 min warmup for vwap_stddev_240

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", 1.4, 1.0, 2.2),
            TunableParam("range_contraction_ratio", 0.7, 0.4, 0.9),
            TunableParam("close_position_threshold", 0.6, 0.5, 0.85),
            TunableParam("stop_pts", 4.0, 2.5, 7.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before numpy conversion
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        zscore_thresh = params.get("zscore_thresh", 1.4)
        range_contraction_ratio = params.get("range_contraction_ratio", 0.7)
        close_position_threshold = params.get("close_position_threshold", 0.6)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Session VWAP (cumulative from open, reset each day) ──────────────
        vwap = _compute_session_vwap(close, volume, day_id, n)

        # ── 20-min rolling stddev of (close - vwap) for z-score denominator ─
        dev = close - vwap
        vwap_std = _rolling_std(dev, lookback=240, min_val=0.5, n=n)

        # ── VWAP z-score ─────────────────────────────────────────────────────
        vwap_zscore = dev / vwap_std

        # ── 30-second (6-bar) and 2-minute (24-bar) high-low ranges ─────────
        range_6, low_6, high_6 = _rolling_range(high, low, lookback=6, n=n)
        range_24, _, _ = _rolling_range(high, low, lookback=24, n=n)

        # Range contraction ratio: 30s range as fraction of 2-min range
        range_ratio = range_6 / np.maximum(range_24, 0.5)

        # Close position in 30s range (0=at range low, 1=at range high)
        close_vs_low_6 = (close - low_6) / np.maximum(range_6, 0.5)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # Explicit VWAP side guard (matches JSON spec exactly):
        # buy_ce requires close < vwap (index below fair value — precondition for reversion)
        # buy_pe requires close > vwap (index above fair value — precondition for reversion)
        # Note: a negative zscore implies close < vwap only roughly (via the stddev normalisation),
        # so the explicit guard is needed to match the JSON specification precisely.
        vwap_arr = vwap  # alias for clarity

        # buy_ce: moderate negative VWAP deviation + basing pattern
        # (range contracting, close near top of compressed range)
        buy_ce = (
            in_session
            & (close < vwap_arr)
            & (vwap_zscore < -zscore_thresh)
            & (range_ratio < range_contraction_ratio)
            & (close_vs_low_6 > close_position_threshold)
        )

        # buy_pe: moderate positive VWAP deviation + basing pattern at top
        # (range contracting, close near bottom of compressed range)
        buy_pe = (
            in_session
            & (close > vwap_arr)
            & (vwap_zscore > zscore_thresh)
            & (range_ratio < range_contraction_ratio)
            & (close_vs_low_6 < (1.0 - close_position_threshold))
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


# ── Helpers ──────────────────────────────────────────────────────────────────

def _compute_session_vwap(
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
    n: int,
) -> np.ndarray:
    """Compute cumulative session VWAP, resetting on each new day_id."""
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_vol = 0.0
    prev_day = day_id[0] - 1  # force reset on first bar
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_vol += volume[i]
        vwap[i] = cum_pv / max(cum_vol, 1.0)
    return vwap


def _rolling_std(
    arr: np.ndarray,
    lookback: int,
    min_val: float,
    n: int,
) -> np.ndarray:
    """Rolling standard deviation with a minimum floor."""
    result = np.full(n, min_val)
    for i in range(1, n):
        start = max(0, i - lookback)
        window = arr[start : i + 1]
        s = np.std(window)
        result[i] = s if s > min_val else min_val
    return result


def _rolling_range(
    high: np.ndarray,
    low: np.ndarray,
    lookback: int,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rolling high-low range, rolling low, and rolling high."""
    rng = np.zeros(n)
    rlow = low.copy()
    rhigh = high.copy()
    for i in range(1, n):
        start = max(0, i - lookback + 1)
        window_high = high[start : i + 1]
        window_low = low[start : i + 1]
        rhigh[i] = np.max(window_high)
        rlow[i] = np.min(window_low)
        rng[i] = rhigh[i] - rlow[i]
    return rng, rlow, rhigh
