"""
volume_momentum_v1 — Volume-Weighted Momentum on NIFTY 5-second bars.

Thesis: When >62% of the last 100 seconds of NIFTY volume lands on up-bars
(5s candles closing above open) AND spot is above session VWAP AND 60-second
ROC is positive, institutional TWAP order flow is driving the move and will
continue for another 30-90 seconds. Buy ATM CE. Symmetrical for PE.

Original: trading_strategies/unique_strategies_all/Strategy_166.json
         Equity 1-min strategy, hold 15-40 min, NIFTY200 universe.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "volume_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — let opening volume settle
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10-minute warmup (120 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_ratio_bull_thresh", 0.62, 0.55, 0.72),
            TunableParam("vol_ratio_bear_thresh", 0.38, 0.28, 0.45),
            TunableParam("roc_thresh", 0.10, 0.05, 0.25),
            TunableParam("stop_pts", 5.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill prices, zero-fill volume) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ──
        vol_ratio_bull = params.get("vol_ratio_bull_thresh", 0.62)
        vol_ratio_bear = params.get("vol_ratio_bear_thresh", 0.38)
        roc_thresh = params.get("roc_thresh", 0.10)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── Session VWAP — cumulative from 09:15, reset each day ──
        typical = (high + low + close) / 3.0
        vwap = np.zeros(n)
        cum_vol = 0.0
        cum_tp_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_vol = 0.0
                cum_tp_vol = 0.0
                prev_day = day_id[i]
            cum_vol += volume[i]
            cum_tp_vol += typical[i] * volume[i]
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0 else close[i]

        # ── Up-bar classification ──
        up_vol = np.where(close > open_, volume, 0.0)

        # ── Rolling 20-bar (100s) up-volume ratio — day-boundary aware ──
        W_ratio = 20
        up_vol_sum = np.zeros(n)
        total_vol_sum = np.zeros(n)
        for i in range(n):
            j = i
            s_up = 0.0
            s_total = 0.0
            count = 0
            while j >= 0 and count < W_ratio and day_id[j] == day_id[i]:
                s_up += up_vol[j]
                s_total += volume[j]
                j -= 1
                count += 1
            up_vol_sum[i] = s_up
            total_vol_sum[i] = s_total

        vol_ratio = np.where(total_vol_sum > 0, up_vol_sum / total_vol_sum, 0.5)

        # ── 1-minute ROC (12 bars = 60s) — day-boundary aware ──
        ROC_W = 12
        roc_12 = np.zeros(n)
        for i in range(n):
            j = i - ROC_W
            if j >= 0 and day_id[j] == day_id[i] and close[j] > 0:
                roc_12[i] = (close[i] - close[j]) / close[j] * 100.0

        # ── Rolling 2-minute volume average (24 prior bars, excluding current) ──
        vol_sma_24 = np.zeros(n)
        for i in range(n):
            j = i - 1
            s = 0.0
            count = 0
            while j >= 0 and count < 24 and day_id[j] == day_id[i]:
                s += volume[j]
                j -= 1
                count += 1
            vol_sma_24[i] = s / count if count > 0 else volume[i]

        # ── Volume ratio trend over last 30s (6 bars) ──
        vol_ratio_trend = np.zeros(n)
        for i in range(n):
            j = i - 6
            if j >= 0 and day_id[j] == day_id[i]:
                vol_ratio_trend[i] = vol_ratio[i] - vol_ratio[j]

        # ── History guard: need at least W_ratio bars in current day ──
        bars_in_day = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                bars_in_day[i] = 1
            else:
                bars_in_day[i] = bars_in_day[i - 1] + 1
        has_history = bars_in_day >= W_ratio

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry signals ──
        buy_ce = (
            in_session
            & has_history
            & (vol_ratio > vol_ratio_bull)
            & (roc_12 > roc_thresh)
            & (close > vwap)
            & (volume > vol_sma_24)
            & (vol_ratio_trend > 0)
        )

        buy_pe = (
            in_session
            & has_history
            & (vol_ratio < vol_ratio_bear)
            & (roc_12 < -roc_thresh)
            & (close < vwap)
            & (volume > vol_sma_24)
            & (vol_ratio_trend < 0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
