"""Opening Momentum: first 15 minutes have larger, more persistent moves.

The opening period (09:18-09:30) has outsized volatility and more directional
persistence due to overnight information being priced in. Momentum signals
in the first 15 minutes are more reliable than mid-day.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "opening_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 558  # 09:18
    session_end_minutes = 585    # 09:45 — ONLY trades in opening period
    max_trades_per_day = 6
    max_lookback = 12  # 1 min warmup

    def tunable_params(self):
        return [
            TunableParam("momentum_threshold", 6.0, 3.0, 15.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 10.0, 6.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        mom_thresh = params.get("momentum_threshold", 6.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 10.0)

        # 6-bar (30s) momentum
        mom_6 = np.zeros(n)
        for i in range(6, n):
            if day_id[i] == day_id[i - 6]:
                mom_6[i] = close[i] - close[i - 6]

        # Opening flag: between 09:18 and 09:45
        is_opening = (time_min >= 558) & (time_min < 585)

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # Note: session_end_minutes=585 already limits to opening period
        buy_ce = in_session & warmed & is_opening & (mom_6 > mom_thresh)
        buy_pe = in_session & warmed & is_opening & (mom_6 < -mom_thresh)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,  # 120s hold — opening moves are bigger
            max_trades_per_day=self.max_trades_per_day,
        )
