"""VWAP Deviation: price far from 5-min VWAP tends to revert.

VWAP (Volume Weighted Average Price) is a meaningful reference at multi-minute
scale. When price deviates significantly, it tends to snap back. Not a 5-second
oscillator — this measures deviation from a 5-minute rolling VWAP.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "vwap_deviation_v1"
    underlying = "NIFTY"
    session_start_minutes = 558
    session_end_minutes = 925
    max_trades_per_day = 10
    max_lookback = 72  # 6 min warmup

    def tunable_params(self):
        return [
            TunableParam("vwap_dev_threshold", 10.0, 6.0, 20.0),
            TunableParam("stop_pts", 4.0, 2.0, 7.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        dev_thresh = params.get("vwap_dev_threshold", 10.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # Typical price
        typical = (high + low + close) / 3.0

        # Rolling 60-bar (5-min) VWAP
        vwap = np.zeros(n)
        lookback = 60
        for i in range(lookback, n):
            if day_id[i] == day_id[i - lookback]:
                window_tp = typical[i - lookback:i]
                window_vol = volume[i - lookback:i]
                vol_sum = np.sum(window_vol)
                if vol_sum > 0:
                    vwap[i] = np.sum(window_tp * window_vol) / vol_sum
                else:
                    vwap[i] = np.mean(window_tp)
            else:
                # Day boundary — use available bars
                start = i - lookback
                for j in range(start, i):
                    if day_id[j] == day_id[i]:
                        start = j
                        break
                if start < i:
                    window_tp = typical[start:i]
                    window_vol = volume[start:i]
                    vol_sum = np.sum(window_vol)
                    if vol_sum > 0:
                        vwap[i] = np.sum(window_tp * window_vol) / vol_sum
                    else:
                        vwap[i] = np.mean(window_tp)

        # Deviation from VWAP
        vwap_dev = np.zeros(n)
        for i in range(lookback, n):
            if vwap[i] > 0:
                vwap_dev[i] = close[i] - vwap[i]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # Price far above VWAP → expect reversion down → BUY PE
        buy_pe = in_session & warmed & (vwap_dev > dev_thresh)
        # Price far below VWAP → expect reversion up → BUY CE
        buy_ce = in_session & warmed & (vwap_dev < -dev_thresh)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
