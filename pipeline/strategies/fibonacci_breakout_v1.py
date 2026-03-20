"""fibonacci_breakout_v1 — Fibonacci Retracement Bounce on NIFTY

Mechanism:
    On NIFTY, intraday swings of 30-80 spot points create strong institutional memory:
    VWAP-benchmarked desks and quant algos pre-place resting limit orders at 38.2%, 50%,
    and 61.8% retracement levels of the dominant 5-minute swing. When NIFTY pulls back
    into one of these precise Fibonacci zones, the concentrated passive order clusters
    absorb directional flow and flip short-term order imbalance within 2-4 bars (10-20s),
    producing a 15-30 spot-point reversal leg.

Converted from: trading_strategies/unique_strategies_all/Strategy_184.json
Original: Fibonacci retracement bounce on 1-min equity bars, hold 20-60 min.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "fibonacci_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min opening noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 60             # 5-minute warmup for swing detection

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("fib_tolerance", 0.003, 0.001, 0.008),
            TunableParam("min_range_pct", 0.0015, 0.0008, 0.003),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 15.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Extract arrays with forward-fill to eliminate NaN
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        fib_tol = params.get("fib_tolerance", 0.003)
        min_range_pct = params.get("min_range_pct", 0.0015)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # 5-minute rolling swing high/low (60 bars @ 5s each)
        # Compressed from original 10-min pivot because we're trading 30-90s reversal,
        # not the full 20-60 min recovery — need the RECENT micro-swing, not multi-hour
        hi_60 = (
            spot_df["high"]
            .fill_null(strategy="forward")
            .rolling_max(window_size=60, min_periods=1)
            .to_numpy()
        )
        lo_60 = (
            spot_df["low"]
            .fill_null(strategy="forward")
            .rolling_min(window_size=60, min_periods=1)
            .to_numpy()
        )

        range_60 = hi_60 - lo_60

        # --- Fibonacci retracement levels ---
        # Bullish: price pulled back FROM the swing high toward the swing low
        fib_382_bull = hi_60 - 0.382 * range_60
        fib_500_bull = hi_60 - 0.500 * range_60
        fib_618_bull = hi_60 - 0.618 * range_60

        # Bearish: price bounced UP from the swing low toward the swing high
        fib_382_bear = lo_60 + 0.382 * range_60
        fib_500_bear = lo_60 + 0.500 * range_60
        fib_618_bear = lo_60 + 0.618 * range_60

        # Tolerance band: within fib_tol * range_60 of any Fib level
        # Floor at 1 point to avoid degenerate zero-range bars
        tol = np.maximum(fib_tol * range_60, 1.0)

        # --- Near any Fibonacci level ---
        near_fib_bull = (
            (np.abs(close - fib_382_bull) <= tol)
            | (np.abs(close - fib_500_bull) <= tol)
            | (np.abs(close - fib_618_bull) <= tol)
        )
        near_fib_bear = (
            (np.abs(close - fib_382_bear) <= tol)
            | (np.abs(close - fib_500_bear) <= tol)
            | (np.abs(close - fib_618_bear) <= tol)
        )

        # --- In-retracement zone (price pulled back but thesis not invalidated) ---
        # Bull: price is below the high by at least 15% of range (actually retracing)
        #       AND still above the 61.8% level (not blown through support)
        in_retracement_bull = (
            (close < hi_60 - 0.15 * range_60)
            & (close >= fib_618_bull - tol)
        )
        # Bear: price is above the low by at least 15% of range (actually bounced up)
        #       AND still below the 61.8% level from the low (not broken through resistance)
        in_retracement_bear = (
            (close > lo_60 + 0.15 * range_60)
            & (close <= fib_618_bear + tol)
        )

        # --- Swing must be meaningful to generate real Fib clustering ---
        # min 0.15% of price ≈ ~36 pts on NIFTY24000 — enough for real order clusters
        meaningful_swing = range_60 > min_range_pct * close

        # --- 15-second momentum confirmation (3 bars @ 5s) ---
        # Replaces original's volume filter — index volume at 5s is too noisy
        mom_3 = np.zeros(n)
        mom_3[3:] = close[3:] - close[:-3]

        # --- Candle direction (reversal confirmation within the Fib zone) ---
        bull_candle = close > open_
        bear_candle = close < open_

        # --- Session and warmup filters ---
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        warmed_up = np.arange(n) >= 60

        # --- Final signals ---
        buy_ce = (
            in_session
            & warmed_up
            & meaningful_swing
            & near_fib_bull
            & in_retracement_bull
            & bull_candle
            & (mom_3 > 0)
        )

        buy_pe = (
            in_session
            & warmed_up
            & meaningful_swing
            & near_fib_bear
            & in_retracement_bear
            & bear_candle
            & (mom_3 < 0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
