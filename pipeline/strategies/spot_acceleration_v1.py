"""Spot Acceleration: detects accelerating price moves (not just direction).

If the rate of change is INCREASING (acceleration), the move has more
energy and is more likely to continue. Deceleration suggests exhaustion.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "spot_acceleration_v1"
    underlying = "NIFTY"
    session_start_minutes = 558
    session_end_minutes = 925
    max_trades_per_day = 12
    max_lookback = 36

    def tunable_params(self):
        return [
            TunableParam("accel_threshold", 3.0, 1.5, 6.0),
            TunableParam("min_velocity", 4.0, 2.0, 8.0),
            TunableParam("stop_pts", 4.0, 2.0, 6.0),
            TunableParam("target_pts", 8.0, 5.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        accel_thresh = params.get("accel_threshold", 3.0)
        min_vel = params.get("min_velocity", 4.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # Velocity: 3-bar (15s) rate of change
        vel = np.zeros(n)
        for i in range(3, n):
            if day_id[i] == day_id[i - 3]:
                vel[i] = close[i] - close[i - 3]

        # Acceleration: change in velocity over 3 bars
        accel = np.zeros(n)
        for i in range(6, n):
            if day_id[i] == day_id[i - 3]:
                accel[i] = vel[i] - vel[i - 3]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # Bullish acceleration: price moving up AND accelerating upward
        buy_ce = in_session & warmed & (vel > min_vel) & (accel > accel_thresh)
        # Bearish acceleration: price moving down AND accelerating downward
        buy_pe = in_session & warmed & (vel < -min_vel) & (accel < -accel_thresh)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,
            max_trades_per_day=self.max_trades_per_day,
        )
