"""nifty_banknifty_spread_v1 — BANKNIFTY 5s index options strategy.

Thesis (Strategy_255): BANK NIFTY carries ~33% weight in NIFTY 50, creating a
tight but imperfect correlation. The intraday return spread between BANK NIFTY
and NIFTY 50 exhibits mean-reverting behavior. When BANK NIFTY's intraday
return deviates significantly from its historical beta-adjusted relationship
with NIFTY, it tends to revert.

Adaptation: Since `compute()` receives only one spot_df (BANKNIFTY), we proxy
the NIFTY-BANKNIFTY spread divergence using BANKNIFTY's deviation from its own
rolling VWAP z-score, calibrated to the typical spread reversion magnitude.
We trade mean reversion of BANKNIFTY's z-score excess relative to its own
intraday trend. This captures the same inefficiency: when BANKNIFTY over-extends
relative to fair value, it mean-reverts.

Source: trading_strategies/unique_strategies_all/Strategy_255.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "nifty_banknifty_spread_v1"
    underlying = "BANKNIFTY"
    session_start_minutes = 570   # 09:30 IST — need warmup for VWAP/z-score
    session_end_minutes = 900     # 15:00 IST
    max_trades_per_day = 8
    max_lookback = 180

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", 1.8, 1.2, 3.0),
            TunableParam("zscore_exit", 0.3, 0.1, 0.8),
            TunableParam("lookback_bars", 120.0, 60.0, 240.0),
            TunableParam("stop_pts", 10.0, 6.0, 18.0),
            TunableParam("target_pts", 14.0, 8.0, 25.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        zscore_entry = float(params.get("zscore_entry", 1.8))
        zscore_exit = float(params.get("zscore_exit", 0.3))
        lookback = int(params.get("lookback_bars", 120))
        stop_pts = float(params.get("stop_pts", 10.0))
        target_pts = float(params.get("target_pts", 14.0))

        # ── Intraday VWAP ────────────────────────────────────────────────────
        typical_price = (high + low + close) / 3.0
        vwap = np.zeros(n)
        for d in np.unique(day_id):
            day_mask = day_id == d
            day_idx = np.where(day_mask)[0]
            vol_day = np.where(volume[day_idx] > 0, volume[day_idx], 1.0)
            vwap[day_idx] = np.cumsum(typical_price[day_idx] * vol_day) / np.cumsum(vol_day)

        # ── Spread from VWAP ─────────────────────────────────────────────────
        spread = close - vwap

        # ── Rolling z-score of spread ────────────────────────────────────────
        zscore = np.zeros(n)
        for i in range(lookback, n):
            seg = spread[i - lookback:i]
            std = np.std(seg)
            if std > 0:
                zscore[i] = (spread[i] - np.mean(seg)) / std

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # When BANKNIFTY is far above its VWAP (z-score > entry), buy PE (mean revert down)
        buy_pe = in_session & (zscore > zscore_entry)
        # When BANKNIFTY is far below its VWAP (z-score < -entry), buy CE (mean revert up)
        buy_ce = in_session & (zscore < -zscore_entry)

        # Signal-based exit: z-score reverts toward 0
        sell_ce = zscore > -zscore_exit
        sell_pe = zscore < zscore_exit

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=sell_ce,
            sell_pe=sell_pe,
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=72,
            max_trades_per_day=self.max_trades_per_day,
        )
