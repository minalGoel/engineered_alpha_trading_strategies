"""Opening Range Breakout strategy for NIFTY index options.

Thesis: The first 15 minutes of the NSE session (09:15-09:30) form the price
discovery window as overnight auction information is absorbed. A breakout above
or below this range, confirmed by volume, signals committed institutional
directional flow.

Source: trading_strategies/unique_strategies_all/Strategy_11.json
Adapted: 15-min ORB on NIFTY index, 5-second bars, option entry on breakout.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "orb_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — after 15-min opening range forms
    session_end_minutes = 870      # 14:30 IST
    max_trades_per_day = 3
    max_lookback = 180             # 15-min warmup (180 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 10.0, 6.0, 18.0),
            TunableParam("volume_ratio", 1.3, 0.9, 2.5),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 10.0))
        vol_ratio_thresh = float(params.get("volume_ratio", 1.3))

        # ── 15-min Opening Range (09:15–09:30, bars 555–570) ─────────────────
        orb_high = np.full(n, np.nan)
        orb_low = np.full(n, np.nan)

        for d in np.unique(day_id):
            day_mask = day_id == d
            day_idx = np.where(day_mask)[0]
            orb_mask = day_mask & (time_min >= 555) & (time_min < 570)
            orb_idx = np.where(orb_mask)[0]
            if len(orb_idx) == 0:
                continue
            orb_high[day_idx] = np.nanmax(high[orb_idx])
            orb_low[day_idx] = np.nanmin(low[orb_idx])

        # Forward-fill NaN
        last_h = close[0] if not np.isnan(close[0]) else 0.0
        last_l = last_h
        for i in range(n):
            if not np.isnan(orb_high[i]):
                last_h = orb_high[i]
                last_l = orb_low[i]
            else:
                orb_high[i] = last_h
                orb_low[i] = last_l

        # ── Rolling 5-min (60-bar) volume average ────────────────────────────
        vol_avg = np.ones(n)
        for i in range(1, n):
            start = max(0, i - 60)
            seg = volume[start:i]
            pos = seg[seg > 0]
            vol_avg[i] = np.mean(pos) if len(pos) > 0 else 1.0
        vol_above = volume > (vol_ratio_thresh * vol_avg)

        # ── 3-bar persistence filter ──────────────────────────────────────────
        above_orb = close > orb_high
        below_orb = close < orb_low
        persist_above = np.zeros(n, dtype=bool)
        persist_below = np.zeros(n, dtype=bool)
        for i in range(2, n):
            if day_id[i] == day_id[i - 1] == day_id[i - 2]:
                persist_above[i] = above_orb[i] and above_orb[i - 1] and above_orb[i - 2]
                persist_below[i] = below_orb[i] and below_orb[i - 1] and below_orb[i - 2]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        buy_ce = in_session & persist_above & vol_above
        buy_pe = in_session & persist_below & vol_above

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
