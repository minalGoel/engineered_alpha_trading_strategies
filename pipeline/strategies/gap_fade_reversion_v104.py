"""gap_fade_reversion_v104 — NIFTY 5s index options strategy.

Thesis (Strategy_114): Overnight gaps driven by retail overreaction are faded
by institutional participants at the open. This variant enters on the FIRST
bar showing a directional close back toward the gap fill level, using a simple
gap threshold and immediate momentum reversal signal (no reversal candle wait).

Differentiated from v116: v116 waits for a confirmed 1-min reversal candle;
v104 enters earlier on first 3-bar reversal momentum within the first 10 minutes.

Source: trading_strategies/unique_strategies_all/Strategy_114.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "gap_fade_reversion_v104"
    underlying = "NIFTY"
    session_start_minutes = 555   # 09:15 IST — enter immediately at open
    session_end_minutes = 580     # 09:40 IST — gap fades must complete early
    max_trades_per_day = 2
    max_lookback = 12

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.004, 0.002, 0.010),
            TunableParam("momentum_bars", 3.0, 2.0, 6.0),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_arr = spot_df["open"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        gap_threshold = float(params.get("gap_threshold", 0.004))
        momentum_bars = int(params.get("momentum_bars", 3))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

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

        # ── N-bar momentum: close[i] > close[i - momentum_bars] for fade ─────
        # Gap-down fade → need close moving UP (buy_ce)
        # Gap-up fade   → need close moving DOWN (buy_pe)
        fading_up = np.zeros(n, dtype=bool)
        fading_down = np.zeros(n, dtype=bool)
        mb = max(1, momentum_bars)
        for i in range(mb, n):
            if day_id[i] == day_id[i - mb]:
                fading_up[i] = close[i] > close[i - mb]
                fading_down[i] = close[i] < close[i - mb]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        buy_ce = in_session & (gap_pct < -gap_threshold) & fading_up
        buy_pe = in_session & (gap_pct > gap_threshold) & fading_down

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
