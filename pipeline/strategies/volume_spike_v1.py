"""Volume Spike Detection: sudden volume surge predicts continuation.

A sudden, outsized volume spike on the underlying indicates aggressive
order flow. The direction of the price move during the spike predicts
continuation over the next 30-60 seconds.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "volume_spike_v1"
    underlying = "NIFTY"
    session_start_minutes = 558
    session_end_minutes = 925
    max_trades_per_day = 12
    max_lookback = 72

    def tunable_params(self):
        return [
            TunableParam("vol_spike_mult", 3.0, 2.0, 5.0),
            TunableParam("min_price_move", 2.0, 1.0, 5.0),
            TunableParam("stop_pts", 4.0, 2.0, 6.0),
            TunableParam("target_pts", 8.0, 5.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        vol_mult = params.get("vol_spike_mult", 3.0)
        min_move = params.get("min_price_move", 2.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # Rolling 60-bar (5-min) average volume per bar
        vol_avg = np.zeros(n)
        lookback = 60
        for i in range(lookback, n):
            vol_avg[i] = np.mean(volume[i - lookback:i])

        # Current bar price change
        bar_change = np.zeros(n)
        for i in range(1, n):
            if day_id[i] == day_id[i - 1]:
                bar_change[i] = close[i] - close[i - 1]

        # Volume spike: current bar volume > vol_mult × average
        is_spike = np.zeros(n, dtype=bool)
        for i in range(lookback, n):
            if vol_avg[i] > 0:
                is_spike[i] = volume[i] > vol_mult * vol_avg[i]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # Volume spike + up move → BUY CE
        buy_ce = in_session & warmed & is_spike & (bar_change > min_move)
        # Volume spike + down move → BUY PE
        buy_pe = in_session & warmed & is_spike & (bar_change < -min_move)

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
