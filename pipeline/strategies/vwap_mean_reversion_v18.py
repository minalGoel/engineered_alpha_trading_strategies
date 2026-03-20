"""VWAP Mean Reversion v18 — Deep VWAP deviation bounce on NIFTY.

Mechanism: On NIFTY, sharp intraday selloffs push the index 40-80 spot points
below session VWAP. Every VWAP-benchmarked institutional algorithm is then sitting
on large positive tracking error and begins accumulating aggressively. We detect
the onset of this accumulation via dev_momentum_24 (2-min rate of z-score change
turning positive while zscore is still extreme) and enter the ATM CE (or PE for
the symmetric case).

Original: vwap_mean_reversion_v18 — nifty200 stocks, 15-min bars, VWAP z-score < -1.7
with rel_volume > 1.6.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_v18"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min for VWAP warmup
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 360            # 30 min warmup for vwap_stddev_360

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_zscore_threshold", 1.7, 1.2, 2.5),
            TunableParam("stop_pts", 6.0, 4.0, 10.0),
            TunableParam("target_pts", 10.0, 6.0, 18.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before numpy conversion
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        zscore_thresh = params.get("vwap_zscore_threshold", 1.7)
        stop_pts = params.get("stop_pts", 6.0)
        target_pts = params.get("target_pts", 10.0)

        # ── Session VWAP (cumulative from open, reset each day) ──────────────
        vwap = _compute_session_vwap(close, volume, day_id, n)

        # ── 30-min rolling stddev of (close - vwap) for z-score denominator ─
        dev = close - vwap
        vwap_std = _rolling_std(dev, lookback=360, min_val=0.5, n=n)

        # ── VWAP z-score ─────────────────────────────────────────────────────
        vwap_zscore = dev / vwap_std

        # ── Dev momentum: 2-min (24 bar) change in z-score ───────────────────
        # Positive means z-score is rising toward 0 from negative (selling abating)
        dev_momentum = np.zeros(n)
        dev_momentum[24:] = vwap_zscore[24:] - vwap_zscore[:-24]

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # buy_ce: extreme negative deviation AND deviation momentum turning positive
        # (institutional accumulators kicking in — z-score still extreme but rate improving)
        buy_ce = (
            in_session
            & (vwap_zscore < -zscore_thresh)
            & (dev_momentum > 0.0)
        )

        # buy_pe: extreme positive deviation AND deviation momentum turning negative
        buy_pe = (
            in_session
            & (vwap_zscore > zscore_thresh)
            & (dev_momentum < 0.0)
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
    """Rolling standard deviation with a minimum floor to avoid division by near-zero."""
    result = np.full(n, min_val)
    for i in range(1, n):
        start = max(0, i - lookback)
        window = arr[start : i + 1]
        s = np.std(window)
        result[i] = s if s > min_val else min_val
    return result
