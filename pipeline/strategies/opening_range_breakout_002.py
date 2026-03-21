"""opening_range_breakout_002 — NIFTY 5s index options strategy.

Thesis (Strategy_131): Delayed institutional continuation after the opening
range resolves. Price discovery is fragmented between the auction, cash open,
and slower institutional execution schedules. After the ORB resolves at 09:30,
institutions enter on pullbacks toward the ORB level with trend alignment
(price must be on the breakout side of VWAP). Trades 09:30-11:00 window.

Differentiated from orb_breakout_v1: orb_breakout_v1 enters on the initial
breakout bar; this strategy trades pullback-and-continuation entries after the
breakout is established (price pulls back toward ORB edge, then resumes).

Source: trading_strategies/unique_strategies_all/Strategy_131.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "opening_range_breakout_002"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — after ORB forms
    session_end_minutes = 660      # 11:00 IST — institutional continuation window
    max_trades_per_day = 4
    max_lookback = 200

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 9.0, 5.0, 16.0),
            TunableParam("pullback_pct", 0.30, 0.15, 0.55),
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
        target_pts = float(params.get("target_pts", 9.0))
        pullback_pct = float(params.get("pullback_pct", 0.30))

        # ── 15-min ORB (09:15–09:30) ──────────────────────────────────────────
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

        last_h = close[0] if not np.isnan(close[0]) else 0.0
        last_l = last_h
        for i in range(n):
            if not np.isnan(orb_high[i]):
                last_h = orb_high[i]
                last_l = orb_low[i]
            else:
                orb_high[i] = last_h
                orb_low[i] = last_l

        orb_range = orb_high - orb_low  # can be 0 on early bars

        # ── Intraday VWAP ────────────────────────────────────────────────────
        typical_price = (high + low + close) / 3.0
        vwap = np.zeros(n)
        for d in np.unique(day_id):
            day_mask = day_id == d
            day_idx = np.where(day_mask)[0]
            vol_day = np.where(volume[day_idx] > 0, volume[day_idx], 1.0)
            vwap[day_idx] = np.cumsum(typical_price[day_idx] * vol_day) / np.cumsum(vol_day)

        # ── Pullback-continuation signal ────────────────────────────────────
        # Bull: price broke above ORB, pulled back to within pullback_pct of
        #       the ORB high, then resumed above ORB high, and close > VWAP.
        # Bear: symmetric.
        near_orb_high = (close >= orb_high - pullback_pct * orb_range) & (close <= orb_high + pullback_pct * orb_range)
        near_orb_low  = (close >= orb_low  - pullback_pct * orb_range) & (close <= orb_low  + pullback_pct * orb_range)

        # Track whether we've seen a confirmed breakout above/below ORB this day
        broke_above = np.zeros(n, dtype=bool)
        broke_below = np.zeros(n, dtype=bool)
        day_broke_above: dict[int, bool] = {}
        day_broke_below: dict[int, bool] = {}

        for i in range(n):
            d = int(day_id[i])
            if d not in day_broke_above:
                day_broke_above[d] = False
                day_broke_below[d] = False
            if close[i] > orb_high[i] + 0.5:
                day_broke_above[d] = True
            if close[i] < orb_low[i] - 0.5:
                day_broke_below[d] = True
            broke_above[i] = day_broke_above[d]
            broke_below[i] = day_broke_below[d]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # Buy CE: breakout above ORB confirmed, current bar near ORB high (pullback), close > VWAP
        buy_ce = in_session & broke_above & near_orb_high & (close > vwap)
        # Buy PE: breakout below ORB confirmed, current bar near ORB low (pullback), close < VWAP
        buy_pe = in_session & broke_below & near_orb_low & (close < vwap)

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
