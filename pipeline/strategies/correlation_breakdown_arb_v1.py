"""correlation_breakdown_arb_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_197): Highly correlated NIFTY stocks occasionally experience
intraday correlation breakdowns where one stock moves sharply while correlated
peers stay flat. These breakdowns are typically caused by single large block
trades or news misattribution, and typically revert within 15-30 minutes.

Adaptation: Without multi-stock data, we proxy the correlation breakdown using
a dual-timeframe divergence on NIFTY itself: when the short-term return
(30-bar / 2.5-min) diverges sharply from the medium-term trend (120-bar / 10-min),
it signals a single-instrument correlation breakdown — a short burst that
hasn't yet propagated to the broader trend, which tends to revert.

Source: trading_strategies/unique_strategies_all/Strategy_197.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "correlation_breakdown_arb_v1"
    underlying = "NIFTY"
    session_start_minutes = 575
    session_end_minutes = 900
    max_trades_per_day = 8
    max_lookback = 150

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("short_window", 24.0, 12.0, 48.0),
            TunableParam("long_window", 120.0, 60.0, 240.0),
            TunableParam("divergence_sigma", 1.5, 1.0, 2.5),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 8.0, 5.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        short_w = int(params.get("short_window", 24))
        long_w = int(params.get("long_window", 120))
        div_sigma = float(params.get("divergence_sigma", 1.5))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # ── Short and long returns ─────────────────────────────────────────────
        roc_short = np.zeros(n)
        roc_long = np.zeros(n)
        for i in range(long_w, n):
            if day_id[i] == day_id[i - short_w] and close[i - short_w] > 0:
                roc_short[i] = (close[i] - close[i - short_w]) / close[i - short_w]
            if day_id[i] == day_id[i - long_w] and close[i - long_w] > 0:
                roc_long[i] = (close[i] - close[i - long_w]) / close[i - long_w]

        # ── Divergence: short return vs long return ───────────────────────────
        divergence = roc_short - roc_long

        # Rolling z-score of divergence
        div_zscore = np.zeros(n)
        for i in range(long_w, n):
            seg = divergence[i - long_w:i]
            std = np.std(seg)
            if std > 0:
                div_zscore[i] = (divergence[i] - np.mean(seg)) / std

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= long_w

        # Short burst too positive vs long trend → revert → buy PE
        buy_pe = in_session & warmed & (div_zscore > div_sigma)
        # Short burst too negative vs long trend → revert → buy CE
        buy_ce = in_session & warmed & (div_zscore < -div_sigma)

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
