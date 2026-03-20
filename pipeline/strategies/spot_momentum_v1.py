"""Spot Momentum Continuation: 15-30 second momentum predicts next 30-60 seconds.

If NIFTY moved significantly in the last 6 bars (30s), predict continuation
for the next 6-12 bars. Momentum is a well-documented short-term effect.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "spot_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 558  # 09:18 IST
    session_end_minutes = 925    # 15:25 IST
    max_trades_per_day = 15
    max_lookback = 36  # 3 min warmup

    def tunable_params(self):
        return [
            TunableParam("momentum_threshold", 8.0, 5.0, 15.0),
            TunableParam("stop_pts", 4.0, 2.0, 6.0),
            TunableParam("target_pts", 8.0, 5.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        mom_thresh = params.get("momentum_threshold", 8.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # 6-bar (30s) momentum
        mom_6 = np.zeros(n)
        for i in range(6, n):
            if day_id[i] == day_id[i - 6]:
                mom_6[i] = close[i] - close[i - 6]

        # 3-bar (15s) momentum for confirmation
        mom_3 = np.zeros(n)
        for i in range(3, n):
            if day_id[i] == day_id[i - 3]:
                mom_3[i] = close[i] - close[i - 3]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # Bullish: 30s momentum > threshold AND 15s momentum confirms direction
        buy_ce = in_session & warmed & (mom_6 > mom_thresh) & (mom_3 > 0)

        # Bearish: 30s momentum < -threshold AND 15s momentum confirms
        buy_pe = in_session & warmed & (mom_6 < -mom_thresh) & (mom_3 < 0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,  # 90s hold
            max_trades_per_day=self.max_trades_per_day,
        )
