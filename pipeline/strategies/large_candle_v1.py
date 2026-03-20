"""Large Candle Follow-Through: outsized 5s candle predicts next 3-6 bars.

A single 5-second candle with range far exceeding average signals aggressive
institutional flow. The close direction of this candle predicts follow-through.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "large_candle_v1"
    underlying = "NIFTY"
    session_start_minutes = 558
    session_end_minutes = 925
    max_trades_per_day = 12
    max_lookback = 72

    def tunable_params(self):
        return [
            TunableParam("range_mult", 3.0, 2.0, 5.0),
            TunableParam("body_ratio", 0.6, 0.4, 0.9),
            TunableParam("stop_pts", 4.0, 2.0, 6.0),
            TunableParam("target_pts", 8.0, 5.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        range_mult = params.get("range_mult", 3.0)
        body_ratio = params.get("body_ratio", 0.6)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # Bar range
        bar_range = high - low

        # Rolling average range (60 bars = 5 min)
        avg_range = np.zeros(n)
        lookback = 60
        for i in range(lookback, n):
            avg_range[i] = np.mean(bar_range[i - lookback:i])

        # Large candle detection
        is_large = np.zeros(n, dtype=bool)
        for i in range(lookback, n):
            if avg_range[i] > 0:
                is_large[i] = bar_range[i] > range_mult * avg_range[i]

        # Body ratio: |close - open| / (high - low)
        body = np.abs(close - open_)
        body_pct = np.zeros(n)
        for i in range(n):
            if bar_range[i] > 0:
                body_pct[i] = body[i] / bar_range[i]

        # Direction
        is_bullish = close > open_
        is_bearish = close < open_

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # Large bullish candle with strong body → BUY CE
        buy_ce = in_session & warmed & is_large & is_bullish & (body_pct > body_ratio)
        # Large bearish candle with strong body → BUY PE
        buy_pe = in_session & warmed & is_large & is_bearish & (body_pct > body_ratio)

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
