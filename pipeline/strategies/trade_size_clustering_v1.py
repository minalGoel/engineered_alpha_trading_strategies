"""trade_size_clustering_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_240): Institutional orders in Indian equities are often split
into round-lot sizes via TWAP/VWAP algorithms. Clusters of consistent volume
in a rolling window signal institutional accumulation/distribution.

Adaptation: Without individual trade data, we use 5-second bar volumes as the
proxy for trade clustering. When NIFTY exhibits repeated volume bars in a
similar range (coefficient of variation < threshold) over a short window while
price drifts consistently in one direction, it signals systematic institutional
TWAP/VWAP execution — predictable continuation.

Source: trading_strategies/unique_strategies_all/Strategy_240.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "trade_size_clustering_v1"
    underlying = "NIFTY"
    session_start_minutes = 570
    session_end_minutes = 900
    max_trades_per_day = 6
    max_lookback = 120

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("cluster_window", 12.0, 6.0, 24.0),
            TunableParam("vol_cv_max", 0.35, 0.20, 0.60),
            TunableParam("price_drift_bars", 6.0, 3.0, 12.0),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 8.0, 5.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        cluster_w = int(params.get("cluster_window", 12))
        vol_cv_max = float(params.get("vol_cv_max", 0.35))
        drift_bars = int(params.get("price_drift_bars", 6))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # ── Volume coefficient of variation (low CV = consistent/clustered vol) ─
        vol_cv = np.full(n, 1.0)
        for i in range(cluster_w, n):
            seg = volume[i - cluster_w:i]
            pos = seg[seg > 0]
            if len(pos) >= 3:
                mean_v = np.mean(pos)
                std_v = np.std(pos)
                if mean_v > 0:
                    vol_cv[i] = std_v / mean_v

        clustered_vol = vol_cv < vol_cv_max

        # ── Consistent price drift: close monotonically drifting in one direction ─
        drifting_up = np.zeros(n, dtype=bool)
        drifting_down = np.zeros(n, dtype=bool)
        db = max(2, drift_bars)
        for i in range(db, n):
            if day_id[i] == day_id[i - db]:
                # Check if each bar moved in the same direction
                up_count = sum(1 for j in range(1, db) if close[i - db + j] > close[i - db + j - 1])
                dn_count = sum(1 for j in range(1, db) if close[i - db + j] < close[i - db + j - 1])
                drifting_up[i] = up_count >= (db - 1)
                drifting_down[i] = dn_count >= (db - 1)

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= cluster_w

        # Institutional buying (consistent vol + steady upward drift) → buy CE
        buy_ce = in_session & warmed & clustered_vol & drifting_up
        # Institutional selling (consistent vol + steady downward drift) → buy PE
        buy_pe = in_session & warmed & clustered_vol & drifting_down

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
