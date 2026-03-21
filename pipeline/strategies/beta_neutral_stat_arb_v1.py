"""beta_neutral_stat_arb_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_195): High-beta and low-beta stocks exhibit intraday relative
value dislocations when market-wide moves create beta-proportional price
changes, but idiosyncratic order flow creates residual mispricing.

Adaptation: Without individual stock data, we construct a beta-adjusted
mean-reversion signal on NIFTY itself. We compute the rolling beta of NIFTY's
returns versus its own longer-term trend, then trade when the "beta-adjusted"
return (actual return / expected beta-scaled return) deviates significantly.
This captures the same intraday mean-reversion that the multi-stock version
captures: when realized return deviates from its beta-predicted fair value.

Source: trading_strategies/unique_strategies_all/Strategy_195.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "beta_neutral_stat_arb_v1"
    underlying = "NIFTY"
    session_start_minutes = 575
    session_end_minutes = 900
    max_trades_per_day = 8
    max_lookback = 240

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("beta_window", 120.0, 60.0, 240.0),
            TunableParam("residual_zscore", 1.8, 1.2, 3.0),
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        beta_window = int(params.get("beta_window", 120))
        resid_thresh = float(params.get("residual_zscore", 1.8))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # ── Bar returns ───────────────────────────────────────────────────────
        returns = np.zeros(n)
        for i in range(1, n):
            if day_id[i] == day_id[i - 1] and close[i - 1] > 0:
                returns[i] = (close[i] - close[i - 1]) / close[i - 1]

        # ── Rolling mean return (proxy for "beta × market return") ───────────
        rolling_mean = np.zeros(n)
        rolling_std = np.full(n, 1e-8)
        for i in range(beta_window, n):
            seg = returns[i - beta_window:i]
            rolling_mean[i] = np.mean(seg)
            rolling_std[i] = max(np.std(seg), 1e-8)

        # ── Residual z-score: how far is current return from rolling mean ─────
        residual_z = np.zeros(n)
        for i in range(beta_window, n):
            residual_z[i] = (returns[i] - rolling_mean[i]) / rolling_std[i]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= beta_window

        # Residual too positive (overshot) → revert down → buy PE
        buy_pe = in_session & warmed & (residual_z > resid_thresh)
        # Residual too negative (undershot) → revert up → buy CE
        buy_ce = in_session & warmed & (residual_z < -resid_thresh)

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
