"""Three Bar Momentum v1 — NIFTY 5-second index options strategy.

Three consecutive same-direction 5s bars with monotonically expanding volume
and range signal an active institutional order still routing — enter ATM options
to ride the 15-90 second continuation before the order slice completes.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "three_bar_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 10
    max_lookback = 36             # 3-min warmup (covers SMA-20 vol context + 3-bar pattern)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("min_bar_ret", 0.0001, 0.00005, 0.0005),
            TunableParam("vol_expand_multiplier", 1.0, 0.8, 1.3),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Extract spot arrays (forward-fill NaN before numpy) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        min_bar_ret = params.get("min_bar_ret", 0.0001)
        vol_mult = params.get("vol_expand_multiplier", 1.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # --- Bar returns: (close - open) / open ---
        denom = np.where(open_ > 0, open_, 1.0)
        bar_ret = (close - open_) / denom

        # --- Bar range ---
        bar_range = high - low

        # --- Shifted arrays (bars i-1, i-2 relative to current i) ---
        bar_ret_1 = np.zeros(n)
        bar_ret_2 = np.zeros(n)
        vol_1 = np.zeros(n)
        vol_2 = np.zeros(n)
        range_1 = np.zeros(n)
        range_2 = np.zeros(n)

        bar_ret_1[1:] = bar_ret[:-1]
        bar_ret_2[2:] = bar_ret[:-2]
        vol_1[1:] = volume[:-1]
        vol_2[2:] = volume[:-2]
        range_1[1:] = bar_range[:-1]
        range_2[2:] = bar_range[:-2]

        # --- Three consecutive bullish bars (each close > open, above threshold) ---
        three_bull = (
            (bar_ret >= min_bar_ret) &
            (bar_ret_1 >= min_bar_ret) &
            (bar_ret_2 >= min_bar_ret)
        )

        # --- Three consecutive bearish bars ---
        three_bear = (
            (bar_ret <= -min_bar_ret) &
            (bar_ret_1 <= -min_bar_ret) &
            (bar_ret_2 <= -min_bar_ret)
        )

        # --- Monotonically expanding volume across 3 bars ---
        # vol[i] > vol[i-1] * mult AND vol[i-1] > vol[i-2] * mult
        # Guard against zero-volume bars
        safe_vol_1 = np.where(vol_1 > 0, vol_1, np.inf)
        safe_vol_2 = np.where(vol_2 > 0, vol_2, np.inf)
        vol_expand = (
            (volume > safe_vol_1 * vol_mult) &
            (vol_1 > safe_vol_2 * vol_mult) &
            (vol_2 > 0)
        )

        # --- Monotonically expanding bar range across 3 bars ---
        range_expand = (
            (bar_range > range_1) &
            (range_1 > range_2) &
            (range_2 > 0)
        )

        # --- Session filter ---
        in_session = (
            (time_min >= self.session_start_minutes) &
            (time_min < self.session_end_minutes)
        )

        # --- Warmup: need at least 3 bars ---
        warmup = np.zeros(n, dtype=bool)
        warmup[3:] = True

        # --- Final signals ---
        base_cond = in_session & warmup & vol_expand & range_expand
        buy_ce = base_cond & three_bull
        buy_pe = base_cond & three_bear

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
