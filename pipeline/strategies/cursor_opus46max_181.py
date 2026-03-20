"""Kalman Filter Trend v1 — cursor_opus46max_181

Thesis: Kalman filter estimates hidden 'true trend' by optimally combining
prediction from a linear state-space model with noisy observations. When
KF_Velocity is strongly positive/negative and signal strength > 2.0 sigma,
enter in the trend direction. VWAP confirmation required.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "cursor_opus46max_181"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 915     # 15:15
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("q_level", default=0.01, low=0.001, high=0.05),
            TunableParam("q_velocity", default=0.001, low=0.0001, high=0.01),
            TunableParam("r_noise", default=0.1, low=0.01, high=0.5),
            TunableParam("signal_strength_thresh", default=2.0, low=1.5, high=3.0),
            TunableParam("velocity_confirm_bars", default=5.0, low=3.0, high=8.0),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("target_pct", default=0.0050, low=0.003, high=0.007),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        q_lev = params.get("q_level", 0.01)
        q_vel = params.get("q_velocity", 0.001)
        r_noise = params.get("r_noise", 0.1)
        sig_thresh = params.get("signal_strength_thresh", 2.0)
        vel_confirm = int(params.get("velocity_confirm_bars", 5.0))
        stop_pct = params.get("stop_loss_pct", 0.0025)
        target_pct = params.get("target_pct", 0.0050)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Kalman filter: state = [level, velocity], observation = close
        # State transition: F = [[1, 1], [0, 1]]
        # Observation: H = [1, 0]
        kf_level = np.zeros(n, dtype=np.float64)
        kf_velocity = np.zeros(n, dtype=np.float64)
        kf_uncertainty = np.zeros(n, dtype=np.float64)
        kf_signal = np.zeros(n, dtype=np.float64)

        # Initialize
        x = np.array([close[0], 0.0])  # [level, velocity]
        P = np.array([[1.0, 0.0], [0.0, 1.0]])
        Q = np.array([[q_lev, 0.0], [0.0, q_vel]])
        R = r_noise
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([1.0, 0.0])

        day_id = df["day_id"].to_numpy()

        for i in range(n):
            # Reset at new day
            if i > 0 and day_id[i] != day_id[i-1]:
                x = np.array([close[i], 0.0])
                P = np.array([[1.0, 0.0], [0.0, 1.0]])

            if i > 0 and day_id[i] == day_id[i-1]:
                # Predict
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q

                # Update
                y = close[i] - H @ x_pred
                S = H @ P_pred @ H + R
                if abs(S) > 1e-15:
                    K = P_pred @ H / S
                    x = x_pred + K * y
                    P = (np.eye(2) - np.outer(K, H)) @ P_pred
                else:
                    x = x_pred
                    P = P_pred

            kf_level[i] = x[0]
            kf_velocity[i] = x[1]
            kf_uncertainty[i] = np.sqrt(max(P[0, 0], 1e-15))
            if kf_uncertainty[i] > 1e-10:
                kf_signal[i] = abs(kf_velocity[i]) / kf_uncertainty[i]

        # Velocity persistence
        vel_pos_consec = np.zeros(n, dtype=np.int32)
        vel_neg_consec = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            vel_pos_consec[i] = (vel_pos_consec[i-1] + 1) if kf_velocity[i] > 0 else 0
            vel_neg_consec[i] = (vel_neg_consec[i-1] + 1) if kf_velocity[i] < 0 else 0

        # Volume filter
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(14, n):
            avg_vol[i] = np.mean(volume[i-14:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 565) & (time_mins <= 915)

        long_entry = (
            (kf_velocity > 0)
            & (kf_signal > sig_thresh)
            & (close > kf_level)
            & (close > vwap)
            & (vel_pos_consec >= vel_confirm)
            & vol_ok & time_ok
        )
        short_entry = (
            (kf_velocity < 0)
            & (kf_signal > sig_thresh)
            & (close < kf_level)
            & (close < vwap)
            & (vel_neg_consec >= vel_confirm)
            & vol_ok & time_ok
        )

        # Signal exit: velocity changes sign
        sig_exit_long = kf_velocity < 0
        sig_exit_short = kf_velocity > 0

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_activate_pct=0.0030,
            trailing_stop_pct=0.0015,
            time_stop_bars=75,
        )
