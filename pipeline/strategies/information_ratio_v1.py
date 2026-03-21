"""information_ratio_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_325): The information ratio (IR = alpha / tracking_error)
computed on rolling intraday data identifies which stocks are currently
generating the best risk-adjusted alpha. Stocks in the top IR quartile,
entered in their current momentum direction, outperform over 30-60 minutes.

Adaptation: On NIFTY as a single instrument, IR reduces to the Sharpe ratio
of recent returns. We compute the rolling Sharpe (mean return / std return)
over a short window. When the rolling Sharpe is significantly positive or
negative, it signals a regime of strong risk-adjusted trend in that direction —
enter in the Sharpe-indicated direction with a time stop.

Source: trading_strategies/unique_strategies_all/Strategy_325.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "information_ratio_v1"
    underlying = "NIFTY"
    session_start_minutes = 575
    session_end_minutes = 900
    max_trades_per_day = 8
    max_lookback = 120

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ir_window", 60.0, 30.0, 120.0),
            TunableParam("ir_threshold", 0.15, 0.08, 0.35),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        ir_w = int(params.get("ir_window", 60))
        ir_thresh = float(params.get("ir_threshold", 0.15))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # ── Bar returns ───────────────────────────────────────────────────────
        returns = np.zeros(n)
        for i in range(1, n):
            if day_id[i] == day_id[i - 1] and close[i - 1] > 0:
                returns[i] = (close[i] - close[i - 1]) / close[i - 1]

        # ── Rolling Information Ratio (Sharpe proxy) ──────────────────────────
        rolling_ir = np.zeros(n)
        for i in range(ir_w, n):
            seg = returns[i - ir_w:i]
            std = np.std(seg)
            if std > 0:
                rolling_ir[i] = np.mean(seg) / std

        # ── VIX filter (avoid trading in extreme vol regimes) ──────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        vix_ok = (vix_close >= 10.0) & (vix_close <= 30.0)
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= ir_w

        # High positive IR (consistent upward returns) → buy CE
        buy_ce = in_session & warmed & vix_ok & (rolling_ir > ir_thresh)
        # High negative IR (consistent downward returns) → buy PE
        buy_pe = in_session & warmed & vix_ok & (rolling_ir < -ir_thresh)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=30,
            max_trades_per_day=self.max_trades_per_day,
        )
