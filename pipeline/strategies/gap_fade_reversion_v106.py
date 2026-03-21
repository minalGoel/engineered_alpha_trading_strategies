"""gap_fade_reversion_v106 — NIFTY 5s index options strategy.

Thesis (Strategy_115): Overnight gaps revert as institutional value-area
participants fade overextended retail gaps. This variant trades the SECOND
phase of the fade — after price has already started reverting, it enters when
the close exceeds the midpoint of the gap range (half-gap fill) indicating
genuine institutional accumulation/distribution.

Differentiated from v104: v104 enters on first reversal bar; v106 waits for
the price to reach the half-gap fill level before entering, catching the
acceleration leg of the reversion.

Source: trading_strategies/unique_strategies_all/Strategy_115.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "gap_fade_reversion_v106"
    underlying = "NIFTY"
    session_start_minutes = 555   # 09:15 IST
    session_end_minutes = 590     # 09:50 IST
    max_trades_per_day = 2
    max_lookback = 12

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.005, 0.003, 0.012),
            TunableParam("fill_ratio", 0.40, 0.25, 0.65),
            TunableParam("stop_pts", 6.0, 3.0, 10.0),
            TunableParam("target_pts", 10.0, 6.0, 18.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_arr = spot_df["open"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        gap_threshold = float(params.get("gap_threshold", 0.005))
        fill_ratio = float(params.get("fill_ratio", 0.40))
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
        day_open = np.zeros(n)
        prev_day_close = np.zeros(n)
        for i in range(n):
            d = int(day_id[i])
            day_open[i] = day_open_map[d]
            idx = unique_days.index(d)
            if idx > 0:
                prev_day_close[i] = day_last_close_map[unique_days[idx - 1]]

        with np.errstate(invalid="ignore", divide="ignore"):
            gap_pct = np.where(
                prev_day_close > 0,
                (day_open - prev_day_close) / prev_day_close,
                0.0,
            )

        # Half-gap fill level: the point that's fill_ratio of the way from open
        # back toward prev_close. For a gap-down: half_fill < day_open (price
        # must rally). For a gap-up: half_fill > day_open (price must fall).
        half_fill = day_open + fill_ratio * (prev_day_close - day_open)

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        gap_down = gap_pct < -gap_threshold
        gap_up = gap_pct > gap_threshold

        # buy_ce: gap-down day AND price has crossed above half-fill level
        buy_ce = in_session & gap_down & (close > half_fill)

        # buy_pe: gap-up day AND price has crossed below half-fill level
        buy_pe = in_session & gap_up & (close < half_fill)

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
