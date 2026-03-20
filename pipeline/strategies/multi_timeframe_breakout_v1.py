"""multi_timeframe_breakout_v1 — Dual-Timeframe Donchian Channel Breakout on NIFTY.

When NIFTY closes above both the 1-minute (12-bar) AND 5-minute (60-bar) Donchian upper
channel simultaneously, institutional supply has been absorbed at two timescales, producing
a high-probability short-term continuation. VWAP and volume ratio confirm direction and
participation. Symmetric logic applies for bearish breakdowns.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "multi_timeframe_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — allow 60-bar slow DC warmup
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 60             # 60 bars = 5-min slow DC warmup

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_threshold", 1.3, 1.0, 2.5),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        vol_threshold = params.get("vol_threshold", 1.3)

        # ── Raw arrays (forward-fill NaN before converting) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high  = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low   = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(strategy="forward").cast(pl.Float64).to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Donchian Channels (computed on previous N bars, no look-ahead) ──
        # Fast: 12 bars = 1 minute
        # Slow: 60 bars = 5 minutes
        dc_fast_upper = np.zeros(n)
        dc_fast_lower = np.full(n, 1e9)
        dc_slow_upper = np.zeros(n)
        dc_slow_lower = np.full(n, 1e9)

        for i in range(12, n):
            dc_fast_upper[i] = np.max(high[i - 12:i])
            dc_fast_lower[i] = np.min(low[i - 12:i])

        for i in range(60, n):
            dc_slow_upper[i] = np.max(high[i - 60:i])
            dc_slow_lower[i] = np.min(low[i - 60:i])

        # ── Session VWAP (resets each day) ──
        vwap = np.zeros(n)
        cum_pv = 0.0
        cum_v = 0.0
        prev_day = -1

        for i in range(n):
            if day_id[i] != prev_day:
                cum_pv = 0.0
                cum_v = 0.0
                prev_day = day_id[i]
            tp = (high[i] + low[i] + close[i]) / 3.0
            v = max(volume[i], 0.0)
            cum_pv += tp * v
            cum_v += v
            vwap[i] = cum_pv / cum_v if cum_v > 0 else close[i]

        # ── Volume Ratio: current / rolling 20-bar mean ──
        vol_ratio = np.ones(n)
        for i in range(20, n):
            avg_v = np.mean(volume[i - 20:i])
            if avg_v > 0:
                vol_ratio[i] = volume[i] / avg_v

        # ── Session filter ──
        in_session = (
            (time_min >= self.session_start_minutes) &
            (time_min < self.session_end_minutes)
        )

        # ── Warmup filter: need at least 60 bars for slow DC ──
        warmed_up = np.zeros(n, dtype=bool)
        warmed_up[60:] = True

        # ── Entry signals ──
        # buy_ce: dual-breakout upward + above VWAP + volume surge
        buy_ce = (
            in_session &
            warmed_up &
            (close > dc_fast_upper) &
            (close > dc_slow_upper) &
            (close > vwap) &
            (vol_ratio > vol_threshold)
        )

        # buy_pe: dual-breakdown downward + below VWAP + volume surge
        buy_pe = (
            in_session &
            warmed_up &
            (close < dc_fast_lower) &
            (close < dc_slow_lower) &
            (close < vwap) &
            (vol_ratio > vol_threshold)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),
            target_points=np.full(n, 7.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
