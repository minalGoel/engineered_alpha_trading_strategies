"""index_arb_nifty_components_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_192): The NIFTY 50 index occasionally diverges from its
synthetic fair value (sum of weighted components) due to asynchronous price
updates and ETF creation/redemption lags. Adapted: since component data is
unavailable, we proxy the fair-value divergence by comparing NIFTY's current
price to a dual-timeframe EMA fair value (fast EMA = market price, slow EMA =
fair value). When the fast price diverges significantly from the slow EMA
(the "fair value" proxy) with volume confirmation, the divergence tends to
revert as arbitrageurs realign the index.

Source: trading_strategies/unique_strategies_all/Strategy_192.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "index_arb_nifty_components_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST
    session_end_minutes = 900     # 15:00 IST
    max_trades_per_day = 8
    max_lookback = 240

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("fast_ema", 12.0, 6.0, 24.0),
            TunableParam("slow_ema", 120.0, 60.0, 240.0),
            TunableParam("divergence_threshold", 0.0015, 0.0008, 0.004),
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        fast_span = int(params.get("fast_ema", 12))
        slow_span = int(params.get("slow_ema", 120))
        div_thresh = float(params.get("divergence_threshold", 0.0015))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # ── EMA computation ───────────────────────────────────────────────────
        def ema(arr, span):
            alpha = 2.0 / (span + 1)
            out = np.zeros(len(arr))
            out[0] = arr[0]
            for i in range(1, len(arr)):
                out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
            return out

        fast_ema = ema(close, fast_span)
        slow_ema = ema(close, slow_span)

        # ── Divergence of fast from slow (proxy for index-component gap) ─────
        with np.errstate(invalid="ignore", divide="ignore"):
            divergence = np.where(slow_ema > 0, (fast_ema - slow_ema) / slow_ema, 0.0)

        # ── Volume above 60-bar average ───────────────────────────────────────
        vol_avg = np.ones(n)
        for i in range(1, n):
            start = max(0, i - 60)
            seg = volume[start:i]
            pos = seg[seg > 0]
            vol_avg[i] = np.mean(pos) if len(pos) > 0 else 1.0
        vol_above = volume > 1.2 * vol_avg

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # Price above fair value → mean revert down → buy PE
        buy_pe = in_session & (divergence > div_thresh) & vol_above
        # Price below fair value → mean revert up → buy CE
        buy_ce = in_session & (divergence < -div_thresh) & vol_above

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
