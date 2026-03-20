"""BANKNIFTY Lead-Lag: BANKNIFTY moves first, NIFTY follows in 10-30 seconds.

Cross-instrument lead-lag is one of the most robust microstructure effects.
If BANKNIFTY surges up while NIFTY hasn't yet moved, NIFTY is likely to follow.
Trade NIFTY options based on BANKNIFTY's leading signal.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "banknifty_leadlag_v1"
    underlying = "NIFTY"  # trade NIFTY based on BANKNIFTY lead
    session_start_minutes = 558
    session_end_minutes = 925
    max_trades_per_day = 12
    max_lookback = 36

    def tunable_params(self):
        return [
            TunableParam("bn_move_threshold", 40.0, 20.0, 80.0),
            TunableParam("nifty_lag_max", 5.0, 2.0, 10.0),
            TunableParam("stop_pts", 4.0, 2.0, 6.0),
            TunableParam("target_pts", 8.0, 5.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        bn_thresh = params.get("bn_move_threshold", 40.0)
        nifty_lag = params.get("nifty_lag_max", 5.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # Load BANKNIFTY data
        if not hasattr(self, '_bn_cache') or self._bn_cache is None:
            from pipeline.config import BANKNIFTY_SYMBOL
            from pipeline.data_loader import load_spot_data
            self._bn_cache = load_spot_data(BANKNIFTY_SYMBOL)

        bn_df = self._bn_cache
        bn_close = np.full(n, 0.0)

        if bn_df is not None and not bn_df.is_empty():
            joined = spot_df.select("datetime").join_asof(
                bn_df.select(["datetime", pl.col("close").alias("bn_close")]).sort("datetime"),
                on="datetime", strategy="backward",
            )
            bn_close = joined["bn_close"].fill_null(strategy="forward").fill_null(50000.0).to_numpy().astype(np.float64)

        # 6-bar (30s) moves
        bn_move_6 = np.zeros(n)
        nifty_move_6 = np.zeros(n)
        for i in range(6, n):
            if day_id[i] == day_id[i - 6]:
                bn_move_6[i] = bn_close[i] - bn_close[i - 6]
                nifty_move_6[i] = close[i] - close[i - 6]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # BANKNIFTY surged up but NIFTY hasn't followed yet → BUY CE
        buy_ce = (in_session & warmed
                  & (bn_move_6 > bn_thresh)
                  & (nifty_move_6 < nifty_lag))

        # BANKNIFTY plunged but NIFTY hasn't followed → BUY PE
        buy_pe = (in_session & warmed
                  & (bn_move_6 < -bn_thresh)
                  & (nifty_move_6 > -nifty_lag))

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
