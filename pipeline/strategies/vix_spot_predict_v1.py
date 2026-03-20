"""VIX Spike → Spot Drop Prediction: VIX moves before spot follows.

India VIX tends to spike 5-15 seconds before NIFTY drops. A sudden VIX
increase predicts imminent spot weakness. VIX declining = complacency → bullish.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "vix_spot_predict_v1"
    underlying = "NIFTY"
    session_start_minutes = 558
    session_end_minutes = 925
    max_trades_per_day = 10
    max_lookback = 72

    def tunable_params(self):
        return [
            TunableParam("vix_spike_threshold", 0.3, 0.15, 0.8),
            TunableParam("vix_drop_threshold", -0.2, -0.6, -0.1),
            TunableParam("stop_pts", 4.0, 2.0, 6.0),
            TunableParam("target_pts", 8.0, 5.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        vix_spike = params.get("vix_spike_threshold", 0.3)
        vix_drop = params.get("vix_drop_threshold", -0.2)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # Get VIX data aligned to spot bars
        vix_close = np.zeros(n)
        if vix_df is not None and not vix_df.is_empty():
            joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime", strategy="backward",
            )
            vix_close = joined["vix_close"].fill_null(strategy="forward").fill_null(15.0).to_numpy().astype(np.float64)

        # 6-bar (30s) VIX change in absolute points
        vix_change_6 = np.zeros(n)
        for i in range(6, n):
            if day_id[i] == day_id[i - 6]:
                vix_change_6[i] = vix_close[i] - vix_close[i - 6]

        # Rolling VIX volatility for normalization
        vix_std = np.zeros(n)
        lookback = 60
        for i in range(lookback, n):
            vix_std[i] = np.std(vix_close[i - lookback:i])

        # Normalized VIX change
        vix_z = np.zeros(n)
        for i in range(lookback, n):
            if vix_std[i] > 0.01:
                vix_z[i] = vix_change_6[i] / vix_std[i]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # VIX spike → expect spot drop → BUY PE
        buy_pe = in_session & warmed & (vix_change_6 > vix_spike)
        # VIX dropping → expect spot rally → BUY CE
        buy_ce = in_session & warmed & (vix_change_6 < vix_drop)

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
