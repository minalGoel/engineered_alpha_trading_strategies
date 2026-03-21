"""cross_listed_arb_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_198): Stocks listed on both NSE and BSE can exhibit transient
price divergences of 2-10 bps due to differential order flow and exchange
matching engine latency. Professional arbitrageurs close these spreads quickly.

Adaptation: Since we don't have dual-exchange data, we proxy the cross-listed
arb opportunity using intraday tick-level momentum bursts on NIFTY. When a
sharp single-bar return (> N standard deviations of recent bar returns) occurs
without corresponding volume support, it resembles a cross-exchange
mismatch that self-corrects. We trade the reversion of these spike bars.

Source: trading_strategies/unique_strategies_all/Strategy_198.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "cross_listed_arb_v1"
    underlying = "NIFTY"
    session_start_minutes = 570
    session_end_minutes = 900
    max_trades_per_day = 10
    max_lookback = 120

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("spike_sigma", 2.5, 1.5, 4.0),
            TunableParam("vol_ratio_max", 0.8, 0.4, 1.2),
            TunableParam("lookback_bars", 60.0, 30.0, 120.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        spike_sigma = float(params.get("spike_sigma", 2.5))
        vol_ratio_max = float(params.get("vol_ratio_max", 0.8))
        lb = int(params.get("lookback_bars", 60))
        stop_pts = float(params.get("stop_pts", 4.0))
        target_pts = float(params.get("target_pts", 6.0))

        # ── Bar returns ───────────────────────────────────────────────────────
        bar_ret = np.zeros(n)
        for i in range(1, n):
            if day_id[i] == day_id[i - 1] and close[i - 1] > 0:
                bar_ret[i] = (close[i] - close[i - 1]) / close[i - 1]

        # ── Rolling return z-score ────────────────────────────────────────────
        ret_zscore = np.zeros(n)
        for i in range(lb, n):
            seg = bar_ret[i - lb:i]
            std = np.std(seg)
            if std > 0:
                ret_zscore[i] = (bar_ret[i] - np.mean(seg)) / std

        # ── Rolling volume ratio (current / avg) ──────────────────────────────
        vol_avg = np.ones(n)
        for i in range(1, n):
            start = max(0, i - lb)
            seg = volume[start:i]
            pos = seg[seg > 0]
            vol_avg[i] = np.mean(pos) if len(pos) > 0 else 1.0

        with np.errstate(invalid="ignore", divide="ignore"):
            vol_ratio = np.where(vol_avg > 0, volume / vol_avg, 1.0)

        # ── Spike: high return z-score but LOW volume (not a real order flow) ─
        spike_up = (ret_zscore > spike_sigma) & (vol_ratio < vol_ratio_max)
        spike_down = (ret_zscore < -spike_sigma) & (vol_ratio < vol_ratio_max)

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= lb

        # Upward spike with no volume support → revert → buy PE
        buy_pe = in_session & warmed & spike_up
        # Downward spike with no volume support → revert → buy CE
        buy_ce = in_session & warmed & spike_down

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,
            max_trades_per_day=self.max_trades_per_day,
        )
