"""round_number_reversal_v1 — Psychological level reversal at NIFTY 100-point multiples.

Mechanism: When NIFTY approaches a 100-point level (22000, 22100, etc.), short option writers
delta-hedge aggressively — short put writers buy as NIFTY falls toward their strike (support),
short call writers sell as it rises (resistance). Combined with retail anchoring (clustered stops
and limits at round numbers), a 2-minute RSI exhaustion signal near the level predicts a 10-15
spot point reversal over the next 30-90 seconds.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "round_number_reversal_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup for RSI(24)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # How close to the 100-multiple to trigger (spot points)
            TunableParam("level_proximity", 12.0, 5.0, 25.0),
            # Minimum 1-min momentum magnitude to confirm directional approach
            TunableParam("momentum_threshold", 3.0, 1.0, 8.0),
            # RSI thresholds for exhaustion detection
            TunableParam("rsi_oversold", 35.0, 25.0, 45.0),
            TunableParam("rsi_overbought", 65.0, 55.0, 75.0),
            # Option premium stop/target in points
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        level_proximity = params.get("level_proximity", 12.0)
        mom_thresh = params.get("momentum_threshold", 3.0)
        rsi_oversold = params.get("rsi_oversold", 35.0)
        rsi_overbought = params.get("rsi_overbought", 65.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Nearest 100-multiple (major psychological level) ──────────────────
        # NIFTY options listed at 50-pt intervals; 100-multiples are max OI nodes
        nearest_100 = np.round(close / 100.0) * 100.0
        abs_dist = np.abs(close - nearest_100)
        near_level = abs_dist < level_proximity

        # ── 1-minute momentum (12 bars × 5s = 60s) ───────────────────────────
        # Confirms directional approach into the level
        mom_12 = np.zeros(n)
        if n > 12:
            mom_12[12:] = close[12:] - close[:-12]

        # ── 2-minute RSI (24 bars × 5s) ──────────────────────────────────────
        # Detects short-term exhaustion AT the level; initialises to 50 (neutral)
        rsi = _compute_rsi(close, 24)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: near level, falling into support (mom negative), RSI oversold
        buy_ce = (
            in_session
            & near_level
            & (mom_12 < -mom_thresh)
            & (rsi < rsi_oversold)
        )

        # buy_pe: near level, rising into resistance (mom positive), RSI overbought
        buy_pe = (
            in_session
            & near_level
            & (mom_12 > mom_thresh)
            & (rsi > rsi_overbought)
        )

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


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI initialised with simple average for the first window."""
    n = len(close)
    rsi = np.full(n, 50.0)

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = 0.0
    avg_loss = 0.0

    for i in range(1, n):
        if i < period:
            # Accumulate for SMA seed
            avg_gain += gains[i]
            avg_loss += losses[i]
        elif i == period:
            # First Wilder average = SMA of first `period` up/down moves
            avg_gain = (avg_gain + gains[i]) / period
            avg_loss = (avg_loss + losses[i]) / period
            if avg_loss == 0.0:
                rsi[i] = 100.0
            else:
                rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        else:
            # Wilder smoothing
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0.0:
                rsi[i] = 100.0
            else:
                rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    return rsi
