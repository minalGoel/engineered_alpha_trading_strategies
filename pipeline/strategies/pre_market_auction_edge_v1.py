"""pre_market_auction_edge_v1 — Opening gap continuation strategy for NIFTY.

Thesis: NIFTY's pre-opening session (09:00-09:08) embeds overnight institutional
positioning into the opening price. When the opening gap exceeds 0.2% from the
previous close AND the first 2 minutes of continuous trading confirm the direction,
algo momentum strategies and delta-hedging market makers amplify the move over the
next 30-90 seconds. Enter at 5-second resolution as soon as confirmation completes.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "pre_market_auction_edge_v1"
    underlying = "NIFTY"
    session_start_minutes = 557   # 09:17 IST — after first 2-min opening noise
    session_end_minutes = 570     # 09:30 IST — entry window effectively closed by bar_in_day filter
    max_lookback = 24             # 2-min warmup (24 × 5s bars)
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.002, 0.001, 0.005),
            TunableParam("confirm_threshold", 0.0005, 0.0002, 0.002),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        gap_threshold = params.get("gap_threshold", 0.002)
        confirm_threshold = params.get("confirm_threshold", 0.0005)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # Pass 1: collect per-day first open, last close, and bar index within day
        bar_in_day = np.zeros(n, dtype=np.int32)
        day_first_open: dict[int, float] = {}
        day_last_close: dict[int, float] = {}
        day_bar_count: dict[int, int] = {}

        for i in range(n):
            d = int(day_id[i])
            if d not in day_first_open:
                day_first_open[d] = float(open_[i])
                day_bar_count[d] = 0
            bar_in_day[i] = day_bar_count[d]
            day_bar_count[d] += 1
            day_last_close[d] = float(close[i])

        # Pass 2: compute per-day opening gap vs previous day's close
        unique_days = sorted(day_first_open.keys())
        day_gap: dict[int, float] = {}
        for idx, d in enumerate(unique_days):
            if idx > 0:
                prev_d = unique_days[idx - 1]
                prev_close = day_last_close[prev_d]
                day_gap[d] = (day_first_open[d] - prev_close) / prev_close if prev_close > 0.0 else 0.0
            else:
                day_gap[d] = 0.0  # first day in dataset: no prior close, no signal

        # Pass 3: build per-bar signal arrays
        gap_pct = np.zeros(n)
        day_open_arr = np.zeros(n)
        for i in range(n):
            d = int(day_id[i])
            gap_pct[i] = day_gap.get(d, 0.0)
            day_open_arr[i] = day_first_open.get(d, float(close[i]))

        # Intraday return from day's first bar open (2-min confirmation signal)
        intraday_ret = np.where(
            day_open_arr > 0.0,
            (close - day_open_arr) / day_open_arr,
            0.0,
        )

        # Entry conditions
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        # bar_in_day 24-72: 09:17 to ~09:21 — confirmation window complete, signal not yet stale
        in_window = (bar_in_day >= 24) & (bar_in_day <= 72)

        # Gap up + first 2 min confirm bullish → buy CE
        buy_ce = (
            in_session
            & in_window
            & (gap_pct > gap_threshold)
            & (intraday_ret > confirm_threshold)
        )

        # Gap down + first 2 min confirm bearish → buy PE
        buy_pe = (
            in_session
            & in_window
            & (gap_pct < -gap_threshold)
            & (intraday_ret < -confirm_threshold)
        )

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
