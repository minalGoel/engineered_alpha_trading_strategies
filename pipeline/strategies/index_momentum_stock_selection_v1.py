"""index_momentum_stock_selection_v1 — Morning regime momentum strategy.

Converted from equity beta-selection basket (Strategy_66.json) to NIFTY 5-second options.

Core thesis: When NIFTY has established >0.5% return from the day's open by 10:00 IST,
large institutional TWAP/VWAP algorithms remain active through the morning session,
creating a persistent directional bias. We trade micro-continuations within this regime,
confirmed by a 2-minute fast return in the same direction.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "index_momentum_stock_selection_v1"
    underlying = "NIFTY"
    session_start_minutes = 600   # 10:00 IST — morning regime established after 45 min
    session_end_minutes = 870     # 14:30 IST — match original exit time; avoid late-session reversals
    max_trades_per_day = 8
    max_lookback = 720            # 60 min warmup; covers pre-10:00 bars needed for day open calc

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("morning_threshold", 0.005, 0.002, 0.012),   # 0.5% default
            TunableParam("fast_ret_threshold", 0.0005, 0.0002, 0.002), # 2-min entry threshold
            TunableParam("vix_max", 22.0, 16.0, 28.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_price = spot_df["open"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        morning_threshold = params.get("morning_threshold", 0.005)
        fast_threshold = params.get("fast_ret_threshold", 0.0005)
        vix_max = params.get("vix_max", 22.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # VIX — align to spot bars via asof join
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # Day open price: use open of the first bar of each calendar day
        day_open_price = np.zeros(n)
        current_day = -1
        first_open = 0.0
        for i in range(n):
            if day_id[i] != current_day:
                current_day = day_id[i]
                first_open = open_price[i]
            day_open_price[i] = first_open

        # Morning return from day open — cumulative regime signal
        # Same concept as original's "nifty_return_from_open" at 10:00
        morning_return = np.where(
            day_open_price > 0,
            (close - day_open_price) / day_open_price,
            0.0,
        )

        # Fast 2-minute return (24 bars × 5s = 120s) — entry trigger
        # Confirms the regime is still active, filters brief reversals
        fast_ret = np.zeros(n)
        for i in range(24, n):
            if day_id[i] == day_id[i - 24] and close[i - 24] > 0:
                fast_ret[i] = (close[i] - close[i - 24]) / close[i - 24]

        # Session and VIX filters
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        vix_ok = vix_close < vix_max

        # Bullish regime: NIFTY up from open + 2-min continuation + low VIX
        buy_ce = (
            in_session
            & vix_ok
            & (morning_return > morning_threshold)
            & (fast_ret > fast_threshold)
        )

        # Bearish regime: NIFTY down from open + 2-min continuation + low VIX
        buy_pe = (
            in_session
            & vix_ok
            & (morning_return < -morning_threshold)
            & (fast_ret < -fast_threshold)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
