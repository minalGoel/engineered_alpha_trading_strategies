"""Bollinger Squeeze Breakout — NIFTY 5-second index options.

Mechanism: On NIFTY, micro-consolidation periods (2-minute Bollinger Band width at
session lows) reflect temporary equilibrium between institutional algorithms. When
price breaks above/below the squeeze boundary with a volume spike, accumulated resting
orders are overwhelmed, triggering stop cascades and VWAP-chasing algo entries that
produce a 10-20 spot point directional impulse in the subsequent 30-90 seconds.

Converted from: trading_strategies/unique_strategies_all/Strategy_345.json
Original: Bollinger Squeeze Breakout on nifty100 stocks, 1-min bars, 30-100 min hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "bollinger_squeeze_breakout"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min pre-open noise
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 120            # 120 bars = 10 min warmup
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("squeeze_threshold", 0.0015, 0.0005, 0.003),
            TunableParam("vol_ratio_threshold", 1.5, 1.1, 2.5),
            TunableParam("stop_pts", 5.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy() \
            if hasattr(spot_df["volume"], "cast") \
            else spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()

        squeeze_threshold = params.get("squeeze_threshold", 0.0015)
        vol_ratio_threshold = params.get("vol_ratio_threshold", 1.5)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        period = 24  # 2-minute Bollinger Band window

        # Rolling statistics — compute SMA, std, bb_width, vol_sma
        sma = np.full(n, np.nan)
        upper_bb = np.full(n, np.nan)
        lower_bb = np.full(n, np.nan)
        bb_width = np.full(n, np.nan)
        vol_sma = np.full(n, np.nan)

        for i in range(period - 1, n):
            c_win = close[i - period + 1: i + 1]
            m = np.mean(c_win)
            s = np.std(c_win)   # population std (ddof=0)
            sma[i] = m
            upper_bb[i] = m + 2.0 * s
            lower_bb[i] = m - 2.0 * s
            if m > 0.0:
                bb_width[i] = 4.0 * s / m  # (upper - lower) / mid

            v_win = volume[i - period + 1: i + 1]
            v_mean = np.mean(v_win)
            vol_sma[i] = v_mean if v_mean > 0.0 else 1.0

        # Volume ratio: current bar volume vs 2-minute rolling mean
        vol_ratio = np.zeros(n)
        valid_vol = vol_sma > 0.0
        vol_ratio[valid_vol] = volume[valid_vol] / vol_sma[valid_vol]

        # Squeeze condition: bb_width below threshold (micro-compression present)
        # At the first breakout bar the last 23 bars are still quiet, keeping bb_width low
        is_squeezed = ~np.isnan(bb_width) & (bb_width < squeeze_threshold)

        # Breakout conditions (close just crossed a Bollinger Band)
        above_upper = ~np.isnan(upper_bb) & (close > upper_bb)
        below_lower = ~np.isnan(lower_bb) & (close < lower_bb)

        # Volume must spike to confirm genuine order-flow imbalance
        high_vol = vol_ratio >= vol_ratio_threshold

        # Session filter
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        buy_ce = in_session & is_squeezed & above_upper & high_vol
        buy_pe = in_session & is_squeezed & below_lower & high_vol

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
