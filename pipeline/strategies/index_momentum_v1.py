"""index_momentum_v1 — NIFTY 5-second index momentum continuation strategy.

Original: Strategy_10.json — NIFTY50 1-min momentum (5-min return > 0.2%)
Conversion: Two-layer signal at 5s — 5-min trend context (ret_60) + 1-min
acceleration confirmation (ret_12). Targets the first 5 option pts of a
sustained institutional momentum burst (30-90s hold).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "index_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 10
    max_lookback = 60             # 60 bars × 5s = 5 min warmup (matches ret_60 lookback)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("momentum_threshold", 0.002, 0.001, 0.005),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        threshold = params.get("momentum_threshold", 0.002)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # ── 5-minute return (60 bars × 5s = 5 min) ──
        # Context filter: confirms sustained directional move matching the
        # original 5-min lookback window.
        ret_60 = np.zeros(n)
        for i in range(60, n):
            base = close[i - 60]
            if base > 0:
                ret_60[i] = (close[i] - base) / base

        # ── 1-minute return (12 bars × 5s = 1 min) ──
        # Acceleration confirmation: detects that the burst is actively
        # continuing NOW, not just that a 5-min trend exists.
        ret_12 = np.zeros(n)
        for i in range(12, n):
            base = close[i - 12]
            if base > 0:
                ret_12[i] = (close[i] - base) / base

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Signals ──
        # buy_ce: 5-min trend bullish AND 1-min acceleration confirming up
        buy_ce = in_session & (ret_60 > threshold) & (ret_12 > 0)

        # buy_pe: 5-min trend bearish AND 1-min acceleration confirming down
        buy_pe = in_session & (ret_60 < -threshold) & (ret_12 < 0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,        # 90 seconds — momentum either extends or fails quickly
            max_trades_per_day=self.max_trades_per_day,
        )
