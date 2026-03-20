"""Expiry Day Afternoon: gamma effects create amplified moves in last 2 hours.

On expiry days, options near ATM have extreme gamma. A 5-point spot move
can cause a 4-point option move instead of 2.5. This amplification means
directional bets on expiry afternoons have better risk/reward.
Strategy only fires on expiry days, 13:30-15:20 IST.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "expiry_afternoon_v1"
    underlying = "NIFTY"
    session_start_minutes = 810  # 13:30 IST
    session_end_minutes = 920    # 15:20 IST (expiry early flatten)
    max_trades_per_day = 8
    max_lookback = 36

    def tunable_params(self):
        return [
            TunableParam("momentum_threshold", 6.0, 3.0, 12.0),
            TunableParam("stop_pts", 3.0, 2.0, 5.0),
            TunableParam("target_pts", 6.0, 4.0, 10.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        mom_thresh = params.get("momentum_threshold", 6.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # Detect expiry days from option data
        is_expiry = np.zeros(n, dtype=bool)
        if option_df is not None and not option_df.is_empty():
            session_dates = spot_df["session_date"].to_list()
            expiry_dates = set(option_df["expiry"].unique().to_list())
            for i in range(n):
                if session_dates[i] in expiry_dates:
                    is_expiry[i] = True

        # 6-bar momentum
        mom_6 = np.zeros(n)
        for i in range(6, n):
            if day_id[i] == day_id[i - 6]:
                mom_6[i] = close[i] - close[i - 6]

        # 3-bar acceleration
        mom_3 = np.zeros(n)
        for i in range(3, n):
            if day_id[i] == day_id[i - 3]:
                mom_3[i] = close[i] - close[i - 3]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # Only trade expiry day afternoons
        # Lower threshold because gamma amplifies the option move
        buy_ce = in_session & warmed & is_expiry & (mom_6 > mom_thresh) & (mom_3 > 0)
        buy_pe = in_session & warmed & is_expiry & (mom_6 < -mom_thresh) & (mom_3 < 0)

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
