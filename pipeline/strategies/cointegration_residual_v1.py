"""cointegration_residual_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_323): Cointegrated stock pairs share a long-run equilibrium.
When their spread deviates from equilibrium, it provides mean-reversion
opportunities. The Johansen cointegration test identifies stable residuals.

Adaptation: Without a cointegrated pair, we implement single-instrument
cointegration by constructing NIFTY's own "equilibrium" via a linear trend
in the intraday price path (using ordinary least squares). The residual from
this intraday trend is the cointegration-like spread. When the residual
deviates beyond a threshold, it tends to revert back to the trend.

This captures the same mean-reversion dynamic: systematic deviation from
an equilibrium relationship (here: intraday trend) predicts reversion.

Source: trading_strategies/unique_strategies_all/Strategy_323.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "cointegration_residual_v1"
    underlying = "NIFTY"
    session_start_minutes = 575
    session_end_minutes = 900
    max_trades_per_day = 8
    max_lookback = 180

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("trend_window", 120.0, 60.0, 240.0),
            TunableParam("residual_sigma", 1.5, 1.0, 2.5),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 8.0, 5.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        trend_w = int(params.get("trend_window", 120))
        resid_sigma = float(params.get("residual_sigma", 1.5))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # ── Rolling OLS residual from linear trend ────────────────────────────
        # For each bar i, fit a linear trend to the past trend_w bars and
        # compute the residual (actual - predicted) at bar i.
        residual = np.zeros(n)
        for i in range(trend_w, n):
            y = close[i - trend_w:i + 1]  # include current bar
            x = np.arange(len(y), dtype=np.float64)
            # OLS: y = a + b*x
            x_mean = np.mean(x)
            y_mean = np.mean(y[:-1])  # exclude current for unbiased prediction
            num = np.sum((x[:-1] - x_mean) * (y[:-1] - y_mean))
            den = np.sum((x[:-1] - x_mean) ** 2)
            if den > 0:
                b = num / den
                a = y_mean - b * x_mean
                predicted = a + b * x[-1]
                residual[i] = y[-1] - predicted

        # ── Rolling z-score of residual ───────────────────────────────────────
        resid_zscore = np.zeros(n)
        for i in range(trend_w * 2, n):
            seg = residual[i - trend_w:i]
            std = np.std(seg)
            if std > 0:
                resid_zscore[i] = (residual[i] - np.mean(seg)) / std

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= trend_w * 2

        # Positive residual (above trend) → revert down → buy PE
        buy_pe = in_session & warmed & (resid_zscore > resid_sigma)
        # Negative residual (below trend) → revert up → buy CE
        buy_ce = in_session & warmed & (resid_zscore < -resid_sigma)

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
