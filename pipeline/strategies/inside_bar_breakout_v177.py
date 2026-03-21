"""Inside Bar Breakout — NIFTY 5s index options (v177).

Mechanism: Sharp 5-second directional bars (mother bars ≥5 pts range) represent
institutional momentum bursts. The subsequent inside bar is a micro-consolidation
as market makers reprice and limit orders absorb the thrust. When the next bar
closes beyond the inside bar's boundary in the mother bar's direction, trapped
opposing stops flush and momentum algos re-enter, driving a 10-20 spot pt
continuation. We enter at the moment directional consensus re-establishes.

Original: inside_bar_breakout_v177 (15-min equity stocks, inside bar + volume surge)
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "inside_bar_breakout_v177"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 10
    max_lookback = 48             # 4 min warmup (covers 36-bar trend window + 2 bars for pattern)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum mother bar range in NIFTY index points — filters out noise bars
            TunableParam("min_mother_range_pts", 5.0, 2.0, 15.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        min_mother_range = params.get("min_mother_range_pts", 5.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # ── Inside bar detection ──────────────────────────────────────────────
        # is_inside[i] = True if bar i is fully contained within bar i-1
        is_inside = np.zeros(n, dtype=bool)
        is_inside[1:] = (high[1:] < high[:-1]) & (low[1:] > low[:-1])

        # ── 3-minute trend (36 bars × 5s) ────────────────────────────────────
        # Context filter: breakout must align with prevailing 3-min trend
        trend_36 = np.zeros(n)
        trend_36[36:] = close[36:] - close[:-36]

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signal vectorisation ───────────────────────────────────────
        # Pattern at bar i:
        #   bar i-2: mother bar (large directional bar)
        #   bar i-1: inside bar (fully within mother bar's range)
        #   bar i:   breakout bar — close breaks beyond inside bar boundary

        # Shift is_inside back: prev_was_inside[i] = is_inside[i-1]
        prev_was_inside = np.zeros(n, dtype=bool)
        prev_was_inside[1:] = is_inside[:-1]

        # Mother bar range at i-2 (evaluated at bar i)
        mother_range = np.zeros(n)
        mother_range[2:] = high[:-2] - low[:-2]

        # Inside bar high/low (bar i-1, evaluated at bar i)
        inside_high = np.zeros(n)
        inside_low = np.zeros(n)
        inside_high[1:] = high[:-1]
        inside_low[1:] = low[:-1]

        # Base filter: previous bar was inside, mother bar was large enough, in session
        base_filter = (
            in_session
            & prev_was_inside
            & (mother_range >= min_mother_range)
        )

        # Buy CE: close breaks above inside bar high, trend is non-negative (bullish)
        buy_ce = (
            base_filter
            & (close > inside_high)
            & (trend_36 >= 0)
        )

        # Buy PE: close breaks below inside bar low, trend is non-positive (bearish)
        buy_pe = (
            base_filter
            & (close < inside_low)
            & (trend_36 <= 0)
        )

        # Guard: no signal on bars 0-1 where pattern can't be formed
        buy_ce[:2] = False
        buy_pe[:2] = False

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,          # 60 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
