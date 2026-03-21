"""EMA Crossover v1 — NIFTY 5-second index options strategy.

Converted from: trading_strategies/unique_strategies_all/Strategy_7.json
Original: EMA(9)/EMA(21) crossover on nifty500 stocks, 1-min bars, hold 5-15 min.

Core thesis: When NIFTY's 1-minute EMA (12 bars) crosses above its 3-minute EMA (36 bars),
institutional TWAP/VWAP buy flow has tipped the order book and momentum algos pile in for
30-90 seconds of continuation. Entry at the cross bar, capturing the fast follow-through.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Compute exponential moving average. Returns array with NaN at start."""
    out = np.empty(len(arr), dtype=np.float64)
    k = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


class Strategy(BaseStrategy):
    name = "ema_crossover_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 72             # 6 min warmup — 2x the slow EMA period (36 bars)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Extract spot close; forward-fill any gaps before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        stop_pts = float(params.get("stop_pts", 3.0))
        target_pts = float(params.get("target_pts", 6.0))

        # ── Indicators ─────────────────────────────────────────────────────────
        # EMA12 = 1-minute fast trend (12 × 5s = 60s)
        # EMA36 = 3-minute slow trend context (36 × 5s = 180s)
        # Lookback deliberately NOT 12x original (which would give 9-min/21-min EMAs
        # too slow to cross within a 30-90s hold window).
        ema12 = _ema(close, 12)
        ema36 = _ema(close, 36)

        # ── Crossover detection ─────────────────────────────────────────────────
        # Shift by 1 to get previous bar values; guard index 0 with current value.
        prev_ema12 = np.empty(n, dtype=np.float64)
        prev_ema36 = np.empty(n, dtype=np.float64)
        prev_ema12[0] = ema12[0]
        prev_ema36[0] = ema36[0]
        prev_ema12[1:] = ema12[:-1]
        prev_ema36[1:] = ema36[:-1]

        # Golden cross: fast EMA just crossed above slow EMA
        golden_cross = (prev_ema12 <= prev_ema36) & (ema12 > ema36)

        # Death cross: fast EMA just crossed below slow EMA
        death_cross = (prev_ema12 >= prev_ema36) & (ema12 < ema36)

        # ── Session filter ──────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ───────────────────────────────────────────────────────
        # buy_ce: golden cross + close above fast EMA (confirms upward momentum)
        buy_ce = in_session & golden_cross & (close > ema12)

        # buy_pe: death cross + close below fast EMA (confirms downward momentum)
        buy_pe = in_session & death_cross & (close < ema12)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts, dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
