"""factor_momentum_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_293): Academic factor investing (value + momentum) works at
monthly frequencies but is under-explored intraday. Stocks that are relatively
cheap AND showing strong intraday momentum tend to outperform in the subsequent
30-60 minutes.

Adaptation: Without individual stock P/E ratios, we construct a single-index
proxy. "Value" factor on index: price is below rolling median (cheap relative
to recent history). "Momentum" factor: price has strong positive ROC over the
last 60 bars. When both align (value dip + momentum resurgence), it signals
a factor momentum entry.

Source: trading_strategies/unique_strategies_all/Strategy_293.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "factor_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 575
    session_end_minutes = 870
    max_trades_per_day = 6
    max_lookback = 180

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("value_window", 120.0, 60.0, 240.0),
            TunableParam("momentum_window", 60.0, 30.0, 120.0),
            TunableParam("momentum_threshold", 0.0008, 0.0003, 0.002),
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            TunableParam("target_pts", 9.0, 5.0, 16.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        val_window = int(params.get("value_window", 120))
        mom_window = int(params.get("momentum_window", 60))
        mom_thresh = float(params.get("momentum_threshold", 0.0008))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 9.0))

        # ── "Value" factor: price below rolling median (cheap) ────────────────
        rolling_median = np.zeros(n)
        for i in range(val_window, n):
            rolling_median[i] = np.median(close[i - val_window:i])
        for i in range(min(val_window, n)):
            rolling_median[i] = close[i]

        below_median = close < rolling_median  # value dip (cheap)
        above_median = close > rolling_median  # expensive

        # ── "Momentum" factor: strong ROC over momentum_window ────────────────
        roc = np.zeros(n)
        for i in range(mom_window, n):
            if day_id[i] == day_id[i - mom_window] and close[i - mom_window] > 0:
                roc[i] = (close[i] - close[i - mom_window]) / close[i - mom_window]

        strong_up = roc > mom_thresh
        strong_down = roc < -mom_thresh

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= val_window

        # Value (cheap) + momentum resuming upward → buy CE
        buy_ce = in_session & warmed & below_median & strong_up
        # "Expensive" + momentum declining → buy PE
        buy_pe = in_session & warmed & above_median & strong_down

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
