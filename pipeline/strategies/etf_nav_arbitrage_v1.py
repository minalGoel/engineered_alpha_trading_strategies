"""etf_nav_arbitrage_v1 — NIFTY 5s index options strategy.

Thesis (Strategy_193): NIFTY 50 ETFs frequently trade at a premium/discount to
their real-time iNAV due to stale limit orders and low market-maker activity
during volatile periods. This mismatch creates a predictable mean-reversion.

Adaptation: Without ETF/iNAV data, we proxy the NAV divergence using the
Bollinger Band z-score of NIFTY's VWAP deviation. When price deviates beyond
1.5σ from intraday VWAP (analogous to an ETF premium/discount), it tends to
revert as market-making activity brings it back toward fair value. The signal
is similar to ETF-NAV arb but operates on the index itself.

Source: trading_strategies/unique_strategies_all/Strategy_193.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "etf_nav_arbitrage_v1"
    underlying = "NIFTY"
    session_start_minutes = 570
    session_end_minutes = 900
    max_trades_per_day = 8
    max_lookback = 180

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bb_window", 120.0, 60.0, 240.0),
            TunableParam("bb_sigma", 1.5, 1.0, 2.5),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        bb_window = int(params.get("bb_window", 120))
        bb_sigma = float(params.get("bb_sigma", 1.5))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 7.0))

        # ── Intraday VWAP (proxy for "fair NAV") ─────────────────────────────
        typical_price = (high + low + close) / 3.0
        vwap = np.zeros(n)
        for d in np.unique(day_id):
            day_mask = day_id == d
            day_idx = np.where(day_mask)[0]
            vol_day = np.where(volume[day_idx] > 0, volume[day_idx], 1.0)
            vwap[day_idx] = np.cumsum(typical_price[day_idx] * vol_day) / np.cumsum(vol_day)

        # ── Rolling Bollinger Bands on VWAP deviation ─────────────────────────
        vwap_dev = close - vwap
        bb_upper = np.zeros(n)
        bb_lower = np.zeros(n)
        for i in range(bb_window, n):
            seg = vwap_dev[i - bb_window:i]
            mean = np.mean(seg)
            std = np.std(seg)
            bb_upper[i] = mean + bb_sigma * std
            bb_lower[i] = mean - bb_sigma * std

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= bb_window

        # Price premium (above upper BB) → revert down → buy PE
        buy_pe = in_session & warmed & (vwap_dev > bb_upper)
        # Price discount (below lower BB) → revert up → buy CE
        buy_ce = in_session & warmed & (vwap_dev < bb_lower)

        # Exit when deviation returns to zero
        sell_ce = vwap_dev >= 0.0
        sell_pe = vwap_dev <= 0.0

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=sell_ce,
            sell_pe=sell_pe,
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=48,
            max_trades_per_day=self.max_trades_per_day,
        )
