"""kalman_filter_trend_v1 — 5-second NIFTY index options strategy.

Mechanism:
    A Kalman filter with 2 states (price level + velocity) separates genuine
    institutional TWAP/VWAP directional flow from 5-second observation noise.
    When NIFTY's Kalman velocity/uncertainty ratio exceeds the signal threshold
    AND velocity has been consistently directional for >= 30 seconds (6 bars)
    AND price agrees with session VWAP — active institutional batch flow is
    detected. These confirmed velocity regimes persist 30–90 additional seconds
    on NIFTY, delivering 5–15 option premium points before the batch exhausts.

Original: trading_strategies/unique_strategies_all/Strategy_319.json
    (Kalman filter trend on NIFTY 50 equity stocks, 1-min bars, 15-60 min hold)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


# ---------------------------------------------------------------------------
# Kalman filter helpers
# ---------------------------------------------------------------------------

def _run_kalman(
    close: np.ndarray,
    day_ids: np.ndarray,
    q_level: float = 0.1,
    q_vel: float = 0.001,
    r_obs: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-state constant-velocity Kalman filter, reset each trading day.

    State: x = [level, velocity]
    Transition: F = [[1, 1], [0, 1]]  (level += velocity, velocity constant)
    Observation: H = [1, 0]           (we observe the price level only)
    Process noise Q = diag([q_level, q_vel])
    Observation noise R = r_obs (scalar)

    Returns (kf_level, kf_velocity, kf_uncertainty) as numpy arrays.
    """
    n = len(close)
    kf_level = np.empty(n)
    kf_velocity = np.empty(n)
    kf_uncertainty = np.empty(n)

    # Constant matrices
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([q_level, q_vel])
    I2 = np.eye(2)
    H_vec = np.array([1.0, 0.0])  # H as a vector for Kalman gain

    current_day = -1
    x = np.zeros(2)
    P = np.eye(2)

    for i in range(n):
        if day_ids[i] != current_day:
            # New session: initialize with first close, zero velocity, high uncertainty
            current_day = day_ids[i]
            x = np.array([close[i], 0.0])
            P = np.array([[100.0, 0.0], [0.0, 1.0]])

        # Store BEFORE prediction (these are the posterior estimates at bar i)
        kf_level[i] = x[0]
        kf_velocity[i] = x[1]
        kf_uncertainty[i] = np.sqrt(max(P[0, 0], 1e-10))

        # --- Kalman predict step ---
        x_pred = np.array([x[0] + x[1], x[1]])  # F @ x
        P_pred = F @ P @ F.T + Q

        # --- Kalman update step ---
        # Innovation (scalar): close[i] is the observation
        innov = close[i] - x_pred[0]  # H @ x_pred = x_pred[0]
        S = P_pred[0, 0] + r_obs       # H @ P_pred @ H.T + R (H=[1,0] picks [0,0])

        # Kalman gain K (2-vector): P_pred @ H.T / S
        # H.T as column picks first column of P_pred
        k = P_pred[:, 0] / S

        # State update
        x = x_pred + k * innov

        # Covariance update: (I - K @ H) @ P_pred
        # K @ H is outer(k, H_vec)
        KH = np.outer(k, H_vec)
        P = (I2 - KH) @ P_pred

    return kf_level, kf_velocity, kf_uncertainty


def _compute_session_vwap(
    close: np.ndarray,
    volume: np.ndarray,
    day_ids: np.ndarray,
) -> np.ndarray:
    """Session-cumulative VWAP, reset at start of each trading day."""
    n = len(close)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_v = 0.0
    current_day = -1

    for i in range(n):
        if day_ids[i] != current_day:
            current_day = day_ids[i]
            cum_pv = 0.0
            cum_v = 0.0
        cum_pv += close[i] * volume[i]
        cum_v += volume[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]

    return vwap


def _compute_velocity_consec(velocity: np.ndarray) -> np.ndarray:
    """Count of consecutive bars with the same-sign velocity (minimum 1)."""
    n = len(velocity)
    consec = np.ones(n, dtype=np.int32)
    for i in range(1, n):
        vel_i = velocity[i]
        vel_prev = velocity[i - 1]
        same_sign = (vel_i > 0.0 and vel_prev > 0.0) or (vel_i < 0.0 and vel_prev < 0.0)
        consec[i] = consec[i - 1] + 1 if same_sign else 1
    return consec


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

class Strategy(BaseStrategy):
    name = "kalman_filter_trend_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min for Kalman convergence
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup for covariance matrix convergence

    def tunable_params(self) -> list[TunableParam]:
        return [
            # velocity/uncertainty ratio threshold to classify directional flow
            TunableParam("signal_threshold", 1.5, 1.0, 3.0),
            # minimum consecutive same-sign velocity bars before entry (30s default)
            TunableParam("consec_bars", 6.0, 4.0, 12.0),
            # stop loss in ATM option premium points
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            # profit target in ATM option premium points
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # --- Raw arrays (forward-fill NaN before numpy conversion) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_ids = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # --- Parameters ---
        signal_threshold = params.get("signal_threshold", 1.5)
        consec_bars = int(params.get("consec_bars", 6.0))
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # --- Kalman filter (level + velocity) ---
        kf_level, kf_velocity, kf_uncertainty = _run_kalman(close, day_ids)

        # Signal strength: velocity t-statistic in Kalman's own uncertainty units
        kf_signal_strength = np.abs(kf_velocity) / np.maximum(kf_uncertainty, 0.001)

        # --- Session VWAP ---
        session_vwap = _compute_session_vwap(close, volume, day_ids)

        # --- Velocity persistence (consecutive same-sign bars) ---
        velocity_consec = _compute_velocity_consec(kf_velocity)

        # --- Session time filter ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # --- Entry signals ---
        # Bullish: Kalman velocity positive + statistically significant + price above
        # Kalman level + price above VWAP + 30s of persistent upward velocity
        buy_ce = (
            in_session
            & (kf_velocity > 0.0)
            & (kf_signal_strength > signal_threshold)
            & (close > kf_level)
            & (close > session_vwap)
            & (velocity_consec >= consec_bars)
        )

        # Bearish: mirror conditions
        buy_pe = (
            in_session
            & (kf_velocity < 0.0)
            & (kf_signal_strength > signal_threshold)
            & (close < kf_level)
            & (close < session_vwap)
            & (velocity_consec >= consec_bars)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120-second maximum hold
            max_trades_per_day=self.max_trades_per_day,
        )
