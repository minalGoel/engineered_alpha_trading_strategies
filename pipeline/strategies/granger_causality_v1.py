"""granger_causality_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_324): Certain stocks lead others in intraday price discovery.
Large-cap liquid names tend to move before smaller peers. By computing rolling
Granger causality tests on pairs, we find predictive lead-lag relationships.

Adaptation: On a single time series, Granger causality reduces to testing
whether lagged values of the series predict its current value beyond its own
AR process. When the autocorrelation of NIFTY returns at lag-L is
significantly positive (lags predict future returns), we have momentum
(Granger-causal relationship from past to future). When significantly
negative, we have mean reversion.

We implement a rolling lagged-correlation test: if corr(ret[t-L], ret[t])
is significantly positive over the last N bars, take a momentum trade.
If significantly negative, take a mean-reversion trade.

Source: trading_strategies/unique_strategies_all/Strategy_324.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "granger_causality_v1"
    underlying = "NIFTY"
    session_start_minutes = 580
    session_end_minutes = 900
    max_trades_per_day = 6
    max_lookback = 240

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("lag_bars", 6.0, 3.0, 18.0),
            TunableParam("corr_window", 120.0, 60.0, 240.0),
            TunableParam("corr_threshold", 0.20, 0.10, 0.40),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 8.0, 5.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        lag = int(params.get("lag_bars", 6))
        corr_w = int(params.get("corr_window", 120))
        corr_thresh = float(params.get("corr_threshold", 0.20))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # ── Bar returns ───────────────────────────────────────────────────────
        returns = np.zeros(n)
        for i in range(1, n):
            if day_id[i] == day_id[i - 1] and close[i - 1] > 0:
                returns[i] = (close[i] - close[i - 1]) / close[i - 1]

        # ── Rolling lagged autocorrelation (Granger proxy) ───────────────────
        # corr(ret[t-lag:t], ret[t-lag+lag:t+lag]) over corr_w window
        lagged_corr = np.zeros(n)
        req = corr_w + lag
        for i in range(req, n):
            y = returns[i - corr_w:i]           # current returns window
            x = returns[i - corr_w - lag:i - lag]  # lagged returns window
            if np.std(x) > 0 and np.std(y) > 0:
                lagged_corr[i] = np.corrcoef(x, y)[0, 1]

        # Current bar momentum (for signal direction)
        cur_roc = np.zeros(n)
        for i in range(lag, n):
            if day_id[i] == day_id[i - lag] and close[i - lag] > 0:
                cur_roc[i] = (close[i] - close[i - lag]) / close[i - lag]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= req

        # Positive lagged autocorr (momentum regime) + upward current move → buy CE
        buy_ce = in_session & warmed & (lagged_corr > corr_thresh) & (cur_roc > 0)
        # Positive lagged autocorr + downward current move → buy PE
        buy_pe = in_session & warmed & (lagged_corr > corr_thresh) & (cur_roc < 0)

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
