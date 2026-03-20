"""consolidation_breakout_v1 — NIFTY 5-second index options strategy.

Thesis: After a directional 15-25 spot point NIFTY move, the index frequently
enters a 5-10 minute tight consolidation as institutional participants place
resting limit orders at the range boundaries, absorbing counter-flow. When
close crosses above/below the consolidation ceiling/floor with volume surge,
those resting orders are exhausted — momentum algos pile in for the next
10-20 spot point continuation over 30-90 seconds.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "consolidation_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min noise + warmup
    session_end_minutes = 885     # 14:45 IST — avoid EOD volatility
    max_trades_per_day = 5
    max_lookback = 360            # 30 min warmup (240 bars prior_move + 120 bars consol)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Range tightness threshold (fraction) — 0.001 = 0.1% of NIFTY price
            TunableParam("consol_range_threshold", 0.001, 0.0005, 0.002),
            # Prior directional move magnitude required before consolidation
            TunableParam("prior_move_threshold", 0.0015, 0.0008, 0.003),
            # Breakout bar volume must exceed consolidation avg by this factor
            TunableParam("vol_ratio_min", 1.5, 1.2, 2.5),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # --- Extract arrays (forward-fill NaN in Polars before numpy) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .cast(pl.Float64)
            .fill_null(0.0)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()

        # --- Parameters ---
        consol_range_threshold = params.get("consol_range_threshold", 0.001)
        prior_move_threshold = params.get("prior_move_threshold", 0.0015)
        vol_ratio_min = params.get("vol_ratio_min", 1.5)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        CONSOL_BARS = 120   # 10-min consolidation window
        PRIOR_BARS = 240    # 20-min lookback for the move before consolidation

        # --- Rolling 10-min high/low (consolidation boundaries) ---
        roll_high = np.empty(n)
        roll_low = np.empty(n)
        roll_high[:] = np.nan
        roll_low[:] = np.nan

        for i in range(CONSOL_BARS, n):
            roll_high[i] = np.max(high[i - CONSOL_BARS:i])
            roll_low[i] = np.min(low[i - CONSOL_BARS:i])

        # Replace NaN with close (neutral — no signal fires when range is unknown)
        roll_high = np.where(np.isnan(roll_high), close, roll_high)
        roll_low = np.where(np.isnan(roll_low), close, roll_low)

        # Consolidation range as fraction of price
        denom = np.where(roll_low > 0, roll_low, 1.0)
        consol_range_pct = (roll_high - roll_low) / denom
        is_consolidating = consol_range_pct < consol_range_threshold

        # --- Prior directional move: return from (i-240) to (i-120) ---
        # This is the move that preceded the consolidation window.
        prior_move = np.zeros(n)
        for i in range(PRIOR_BARS, n):
            ref = close[i - PRIOR_BARS]
            if ref > 0:
                prior_move[i] = (close[i - CONSOL_BARS] - ref) / ref

        # --- Rolling 10-min volume mean (baseline for volume confirmation) ---
        vol_sma = np.zeros(n)
        for i in range(CONSOL_BARS, n):
            v = volume[i - CONSOL_BARS:i]
            s = np.sum(v)
            vol_sma[i] = s / CONSOL_BARS if s > 0 else 1.0

        vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0.0)

        # --- Session filter ---
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # --- Entry signals ---
        # Bullish: breakout above consolidation ceiling + prior up-move + volume
        buy_ce = (
            in_session
            & is_consolidating
            & (close > roll_high)
            & (prior_move > prior_move_threshold)
            & (vol_ratio >= vol_ratio_min)
        )

        # Bearish: breakout below consolidation floor + prior down-move + volume
        buy_pe = (
            in_session
            & is_consolidating
            & (close < roll_low)
            & (prior_move < -prior_move_threshold)
            & (vol_ratio >= vol_ratio_min)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,            # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
