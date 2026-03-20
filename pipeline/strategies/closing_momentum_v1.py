"""Closing Momentum: last 30 minutes have institutional flow patterns.

The closing period (15:00-15:25) sees institutional rebalancing, creating
predictable directional flow. Momentum in the last 30 minutes is more
persistent due to portfolio adjustment urgency.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "closing_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 900  # 15:00 IST
    session_end_minutes = 925    # 15:25 IST
    max_trades_per_day = 5
    max_lookback = 36

    def tunable_params(self):
        return [
            TunableParam("momentum_threshold", 8.0, 4.0, 15.0),
            TunableParam("stop_pts", 4.0, 2.0, 7.0),
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

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        buy_ce = in_session & warmed & (mom_6 > mom_thresh)
        buy_pe = in_session & warmed & (mom_6 < -mom_thresh)

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
