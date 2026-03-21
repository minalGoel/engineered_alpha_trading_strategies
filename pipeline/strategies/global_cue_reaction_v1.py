"""global_cue_reaction_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_262): Indian markets open at 09:15 IST. By this time, SGX
NIFTY futures have already priced in overnight US, European, and Asian market
moves. The opening gap in NIFTY (today's open vs yesterday's close) provides
a directional signal: when NIFTY opens with a clear directional gap (>0.3%),
global cues have delivered a consensus direction that tends to persist for
20-40 minutes as domestic institutions align their books.

Since SGX NIFTY data is unavailable, we use the overnight gap (today's open
vs yesterday's close) as the proxy for global cue direction. Large gap-ups
lead to continuation in the first 30 minutes; large gap-downs similarly.

Source: trading_strategies/unique_strategies_all/Strategy_262.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "global_cue_reaction_v1"
    underlying = "NIFTY"
    session_start_minutes = 555   # 09:15 IST — enter at open
    session_end_minutes = 590     # 09:50 IST — global cue trades resolve quickly
    max_trades_per_day = 2
    max_lookback = 12

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.003, 0.002, 0.008),
            TunableParam("momentum_confirm_bars", 3.0, 2.0, 8.0),
            TunableParam("stop_pts", 6.0, 3.0, 12.0),
            TunableParam("target_pts", 10.0, 6.0, 20.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_arr = spot_df["open"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        gap_threshold = float(params.get("gap_threshold", 0.003))
        confirm_bars = int(params.get("momentum_confirm_bars", 3))
        stop_pts = float(params.get("stop_pts", 6.0))
        target_pts = float(params.get("target_pts", 10.0))

        # ── Day open and previous close ───────────────────────────────────────
        day_open_map: dict[int, float] = {}
        day_last_close_map: dict[int, float] = {}
        for i in range(n):
            d = int(day_id[i])
            if d not in day_open_map:
                day_open_map[d] = float(open_arr[i])
            day_last_close_map[d] = float(close[i])

        unique_days = sorted(day_open_map.keys())
        gap_pct = np.zeros(n)
        for i in range(n):
            d = int(day_id[i])
            day_open_val = day_open_map[d]
            idx = unique_days.index(d)
            if idx > 0:
                prev_close = day_last_close_map[unique_days[idx - 1]]
                if prev_close > 0:
                    gap_pct[i] = (day_open_val - prev_close) / prev_close

        # ── Momentum in gap direction: close moving in same direction as gap ──
        mom_with_gap_up = np.zeros(n, dtype=bool)
        mom_with_gap_down = np.zeros(n, dtype=bool)
        cb = max(1, confirm_bars)
        for i in range(cb, n):
            if day_id[i] == day_id[i - cb]:
                mom_with_gap_up[i] = close[i] > close[i - cb]
                mom_with_gap_down[i] = close[i] < close[i - cb]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # Gap-up + price continuing up → buy CE (bullish global cues)
        buy_ce = in_session & (gap_pct > gap_threshold) & mom_with_gap_up
        # Gap-down + price continuing down → buy PE (bearish global cues)
        buy_pe = in_session & (gap_pct < -gap_threshold) & mom_with_gap_down

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
