"""momentum_decay_adaptive_v1 — AR(1) momentum persistence regime strategy.

Detects whether NIFTY 5-second returns show positive autocorrelation (institutional
TWAP/VWAP algo still working directional order). When AR(1) coefficient > threshold,
trades in the direction of 1-minute momentum. Stops and targets scale adaptively
with AR(1) magnitude — higher persistence → wider parameters, longer hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_ar1(ret: np.ndarray, window: int) -> np.ndarray:
    """Rolling AR(1) coefficient: corr(ret[t-window:t-1], ret[t-window+1:t]).

    Returns array of length n; values before index `window` are 0.0.
    """
    n = len(ret)
    ar1 = np.zeros(n)
    for i in range(window, n):
        x = ret[i - window: i - 1]
        y = ret[i - window + 1: i]
        x_std = np.std(x)
        y_std = np.std(y)
        if x_std < 1e-12 or y_std < 1e-12:
            ar1[i] = 0.0
            continue
        cov = np.mean((x - x_std) * (y - y_std))  # simplified — use corrcoef
        # Use numpy corrcoef for correctness
        c = np.corrcoef(x, y)
        ar1[i] = c[0, 1] if not np.isnan(c[0, 1]) else 0.0
    return ar1


def _session_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Compute intraday VWAP resetting at each new day_id."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_vol = 0.0
    prev_day = -1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_vol += volume[i]
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else close[i]
    return vwap


class Strategy(BaseStrategy):
    name = "momentum_decay_adaptive_v1"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — need 10-min warmup for AR(1)
    session_end_minutes = 920      # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 240             # 20-min warmup (safety margin for AR1 window)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ar1_threshold", 0.12, 0.05, 0.25),
            TunableParam("mom_threshold_bps", 15.0, 8.0, 30.0),
            TunableParam("vwap_filter_bps", 5.0, 0.0, 20.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ──
        ar1_threshold = params.get("ar1_threshold", 0.12)
        mom_threshold_bps = params.get("mom_threshold_bps", 15.0)
        vwap_filter_bps = params.get("vwap_filter_bps", 5.0)

        # ── 5-second log returns ──
        ret = np.zeros(n)
        safe_close = np.where(close > 0, close, 1.0)
        ret[1:] = np.log(close[1:] / safe_close[:-1])

        # ── Rolling AR(1) coefficient over 120-bar (10-min) window ──
        ar1 = _rolling_ar1(ret, 120)

        # ── 1-minute ROC (12 bars) in basis points ──
        roc_12 = np.zeros(n)
        roc_12[12:] = (close[12:] - close[:-12]) / safe_close[:-12] * 10000.0

        # ── Session VWAP ──
        vwap = _session_vwap(close, volume, day_id)
        safe_vwap = np.where(vwap > 0, vwap, close)
        vwap_dist_bps = (close - safe_vwap) / safe_vwap * 10000.0

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Momentum regime: positive AR(1) → institutional directional flow active ──
        momentum_regime = ar1 > ar1_threshold

        # ── Entry signals ──
        buy_ce = (
            in_session
            & momentum_regime
            & (roc_12 > mom_threshold_bps)
            & (vwap_dist_bps > vwap_filter_bps)
        )
        buy_pe = (
            in_session
            & momentum_regime
            & (roc_12 < -mom_threshold_bps)
            & (vwap_dist_bps < -vwap_filter_bps)
        )

        # ── Adaptive stop/target: scale with AR(1) magnitude ──
        # AR1 in [ar1_threshold, ~0.40]; normalise to [0, 1] for scaling
        ar1_norm = np.clip((ar1 - ar1_threshold) / (0.40 - ar1_threshold), 0.0, 1.0)

        # Stop: 3 pts (weak persistence) → 6 pts (strong persistence)
        stop_pts = 3.0 + ar1_norm * 3.0

        # Target: 5 pts (weak) → 10 pts (strong); 1:1.67 R:R throughout
        target_pts = 5.0 + ar1_norm * 5.0

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_pts,
            target_points=target_pts,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds fixed (middle of 60-120s range)
            max_trades_per_day=self.max_trades_per_day,
        )
