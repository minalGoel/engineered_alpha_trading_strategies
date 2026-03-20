"""Volume-Confirmed Momentum: momentum + above-average volume = stronger signal.

Volume spikes validate price moves. Momentum without volume fades;
momentum with volume continues.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "volume_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 558
    session_end_minutes = 925
    max_trades_per_day = 12
    max_lookback = 72  # 6 min warmup for volume average

    def tunable_params(self):
        return [
            TunableParam("momentum_threshold", 6.0, 4.0, 12.0),
            TunableParam("vol_multiplier", 1.5, 1.2, 3.0),
            TunableParam("stop_pts", 4.0, 2.0, 6.0),
            TunableParam("target_pts", 8.0, 5.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        mom_thresh = params.get("momentum_threshold", 6.0)
        vol_mult = params.get("vol_multiplier", 1.5)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # 6-bar momentum
        mom_6 = np.zeros(n)
        for i in range(6, n):
            if day_id[i] == day_id[i - 6]:
                mom_6[i] = close[i] - close[i - 6]

        # Rolling 60-bar (5-min) average volume
        vol_avg = np.zeros(n)
        lookback = 60
        for i in range(lookback, n):
            vol_avg[i] = np.mean(volume[i - lookback:i])

        # Current 6-bar volume sum
        vol_6 = np.zeros(n)
        for i in range(6, n):
            vol_6[i] = np.sum(volume[i - 6:i])

        # Volume spike: current 6-bar sum > vol_mult × average 6-bar sum
        avg_6bar_vol = vol_avg * 6  # scale to 6-bar window
        vol_spike = np.zeros(n, dtype=bool)
        for i in range(lookback, n):
            if avg_6bar_vol[i] > 0:
                vol_spike[i] = vol_6[i] > vol_mult * avg_6bar_vol[i]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        buy_ce = in_session & warmed & (mom_6 > mom_thresh) & vol_spike
        buy_pe = in_session & warmed & (mom_6 < -mom_thresh) & vol_spike

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
