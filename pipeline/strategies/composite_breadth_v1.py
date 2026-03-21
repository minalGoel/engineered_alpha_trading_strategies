"""composite_breadth_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_302): Market breadth — the proportion of stocks advancing
vs declining — provides information about the strength or weakness of an index
move that price alone cannot convey. A NIFTY rally driven by few stocks
(narrow breadth) is fragile; one with broad participation is sustainable.

Adaptation: Without individual stock data, we proxy breadth using the internal
consistency of NIFTY's bar-level price action. "Internal breadth" = the
proportion of recent bars (in a lookback window) that closed in the same
direction as the current bar. When >70% of recent bars agree with the
current bar direction, it signals broad participation (strong breadth).
When <30% agree, it signals narrow/divergent price action.

Source: trading_strategies/unique_strategies_all/Strategy_302.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "composite_breadth_v1"
    underlying = "NIFTY"
    session_start_minutes = 575
    session_end_minutes = 900
    max_trades_per_day = 6
    max_lookback = 120

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("breadth_window", 60.0, 30.0, 120.0),
            TunableParam("breadth_threshold", 0.65, 0.55, 0.80),
            TunableParam("momentum_bars", 12.0, 6.0, 24.0),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 9.0, 5.0, 16.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        breadth_window = int(params.get("breadth_window", 60))
        breadth_thresh = float(params.get("breadth_threshold", 0.65))
        mom_bars = int(params.get("momentum_bars", 12))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 9.0))

        # ── Bar direction: +1 if close > prev close, -1 if below, 0 if flat ──
        bar_dir = np.zeros(n)
        for i in range(1, n):
            if day_id[i] == day_id[i - 1]:
                if close[i] > close[i - 1]:
                    bar_dir[i] = 1
                elif close[i] < close[i - 1]:
                    bar_dir[i] = -1

        # ── Internal breadth: fraction of recent bars in same direction ───────
        breadth_up = np.zeros(n)   # fraction of up bars in window
        breadth_down = np.zeros(n)
        for i in range(breadth_window, n):
            seg = bar_dir[i - breadth_window:i]
            total = len(seg)
            if total > 0:
                breadth_up[i] = np.sum(seg > 0) / total
                breadth_down[i] = np.sum(seg < 0) / total

        # ── Momentum in current direction ─────────────────────────────────────
        roc = np.zeros(n)
        for i in range(mom_bars, n):
            if day_id[i] == day_id[i - mom_bars] and close[i - mom_bars] > 0:
                roc[i] = (close[i] - close[i - mom_bars]) / close[i - mom_bars]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= breadth_window

        # Broad upward breadth + positive momentum → buy CE
        buy_ce = in_session & warmed & (breadth_up > breadth_thresh) & (roc > 0)
        # Broad downward breadth + negative momentum → buy PE
        buy_pe = in_session & warmed & (breadth_down > breadth_thresh) & (roc < 0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=36,
            max_trades_per_day=self.max_trades_per_day,
        )
