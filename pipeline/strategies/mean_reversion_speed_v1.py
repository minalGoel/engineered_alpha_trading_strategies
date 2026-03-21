"""mean_reversion_speed_v1 — OU-process VWAP mean-reversion speed strategy for NIFTY 5s options.

Mechanism: Estimates Ornstein-Uhlenbeck theta (mean-reversion speed) via OLS on a rolling
5-minute (60-bar) window of VWAP deviation. When theta is high AND stable, institutional VWAP
algorithms are actively absorbing deviations. Enter on OU z-score extremes (±2.0), targeting
reversion to VWAP within 30-120 seconds.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "mean_reversion_speed_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — need ~70 bars warmup (60 OU window + 10 stability)
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 120            # 10 min warmup (120 × 5s)
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("theta_min", 0.05, 0.02, 0.20),
            TunableParam("zscore_entry", 2.0, 1.5, 3.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Parameters ──
        theta_min = float(params.get("theta_min", 0.05))
        zscore_entry = float(params.get("zscore_entry", 2.0))
        stop_pts = float(params.get("stop_pts", 4.0))
        target_pts = float(params.get("target_pts", 6.0))

        # ── Raw data ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy(allow_copy=True).astype(np.float64)
        volume = spot_df["volume"].fill_null(0).to_numpy(allow_copy=True).astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy(allow_copy=True)
        day_id = spot_df["day_id"].to_numpy(allow_copy=True)

        # ── Session VWAP ──
        vwap = _compute_session_vwap(close, volume, day_id, n)

        # ── VWAP deviation in bps ──
        vwap_safe = np.where(vwap > 0, vwap, close)
        vwap_dev = (close - vwap_safe) / vwap_safe * 10000.0  # bps

        # ── OU parameter estimation (rolling 60-bar OLS) ──
        ou_theta, ou_sigma = _estimate_ou_params(vwap_dev, window=60)

        # ── OU equilibrium z-score ──
        # Stationary std of OU process = sigma / sqrt(2 * theta)
        ou_zscore = _compute_ou_zscore(vwap_dev, ou_theta, ou_sigma)

        # ── Theta stability filter: theta > 0.6 * theta_min for 10 consecutive bars ──
        theta_stable = _rolling_min_above(ou_theta, window=10, threshold=0.6 * theta_min)

        # ── VIX filter ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            try:
                vix_joined = spot_df.select("datetime").join_asof(
                    vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                    on="datetime",
                    strategy="backward",
                )
                vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy(allow_copy=True).astype(np.float64)
            except Exception:
                pass

        # ── Masks ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < 22.0
        fast_rev = ou_theta > theta_min
        regime_ok = in_session & vix_ok & fast_rev & theta_stable

        # Below VWAP (negative z) → bounce up → buy CE
        buy_ce = regime_ok & (ou_zscore < -zscore_entry)
        # Above VWAP (positive z) → fade down → buy PE
        buy_pe = regime_ok & (ou_zscore > zscore_entry)

        # Prevent simultaneous signals
        both = buy_ce & buy_pe
        buy_ce = buy_ce & ~both
        buy_pe = buy_pe & ~both

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts, dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )


# ── Helper functions ──────────────────────────────────────────────────────────

def _compute_session_vwap(
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
    n: int,
) -> np.ndarray:
    """Cumulative session VWAP, reset on each new day_id."""
    vwap = np.empty(n, dtype=np.float64)
    cum_pv = 0.0
    cum_vol = 0.0
    prev_day = -9999

    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        v = volume[i] if volume[i] > 0 else 1.0
        cum_pv += close[i] * v
        cum_vol += v
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else close[i]

    return vwap


def _estimate_ou_params(
    dev: np.ndarray,
    window: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate OU theta and sigma via OLS on rolling window.

    OLS model: delta(dev)[t] = -theta * dev[t-1] + intercept + epsilon
    theta = -slope (positive theta means mean-reverting).
    sigma = std(residuals).
    """
    n = len(dev)
    ou_theta = np.zeros(n, dtype=np.float64)
    ou_sigma = np.zeros(n, dtype=np.float64)

    # Pre-compute first differences
    ddv = np.empty(n, dtype=np.float64)
    ddv[0] = 0.0
    ddv[1:] = dev[1:] - dev[:-1]

    for i in range(window + 1, n):
        # y = ddv[i-window+1 : i+1], x_lag = dev[i-window : i]
        y = ddv[i - window + 1: i + 1]      # length = window
        x_lag = dev[i - window: i]           # length = window
        if len(y) < 10:
            continue
        # OLS with intercept: y = a * x_lag + b
        A = np.empty((window, 2), dtype=np.float64)
        A[:, 0] = x_lag
        A[:, 1] = 1.0
        try:
            result = np.linalg.lstsq(A, y, rcond=None)
            coeffs = result[0]
            slope = coeffs[0]
            if slope < 0.0:
                ou_theta[i] = -slope
                resid = y - A @ coeffs
                ou_sigma[i] = float(np.std(resid))
        except Exception:
            pass

    return ou_theta, ou_sigma


def _compute_ou_zscore(
    dev: np.ndarray,
    ou_theta: np.ndarray,
    ou_sigma: np.ndarray,
) -> np.ndarray:
    """OU equilibrium z-score: dev / (sigma / sqrt(2 * theta))."""
    n = len(dev)
    zscore = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if ou_theta[i] > 1e-6 and ou_sigma[i] > 1e-6:
            eq_std = ou_sigma[i] / np.sqrt(2.0 * ou_theta[i])
            if eq_std > 1e-6:
                zscore[i] = dev[i] / eq_std
    return zscore


def _rolling_min_above(
    arr: np.ndarray,
    window: int,
    threshold: float,
) -> np.ndarray:
    """Returns True at bar i if min(arr[i-window:i]) > threshold."""
    n = len(arr)
    result = np.zeros(n, dtype=bool)
    for i in range(window, n):
        if np.min(arr[i - window: i]) > threshold:
            result[i] = True
    return result
