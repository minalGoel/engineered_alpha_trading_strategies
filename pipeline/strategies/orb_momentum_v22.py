"""Opening Range Breakout momentum strategy for NIFTY index options.

Thesis: NIFTY's first 20 minutes (09:15-09:35) is the overnight information
absorption window. When price closes above/below this range with VWAP and
volume confirmation, institutional order flow has committed a direction,
producing 20-40 spot point continuation moves.

Converted from: trading_strategies/unique_strategies_all/Strategy_102.json
Original: ORB on Nifty200 stocks, 1-min bars, 10-30 min hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "orb_momentum_v22"
    underlying = "NIFTY"
    session_start_minutes = 575    # 09:35 IST — after 20-min opening range forms
    session_end_minutes = 870      # 14:30 IST — avoid low-volume late-day ORBs
    max_trades_per_day = 4
    max_lookback = 240             # 20-min warmup for opening range (240 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
            TunableParam("volume_ratio", 1.2, 0.8, 2.5),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays ──────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))
        vol_ratio_thresh = float(params.get("volume_ratio", 1.2))

        # ── Opening Range High/Low (fixed 09:15-09:35 window per day) ───────
        # Same 20-minute real-time window regardless of bar size; 240 bars at 5s.
        orb_high = np.full(n, np.nan)
        orb_low = np.full(n, np.nan)

        for d in np.unique(day_id):
            day_mask = day_id == d
            day_idx = np.where(day_mask)[0]

            # Bars within opening range window: 09:15 (555) to 09:35 (575)
            orb_mask = day_mask & (time_min >= 555) & (time_min < 575)
            orb_idx = np.where(orb_mask)[0]

            if len(orb_idx) == 0:
                continue

            day_orb_high = np.nanmax(high[orb_idx])
            day_orb_low = np.nanmin(low[orb_idx])
            orb_high[day_idx] = day_orb_high
            orb_low[day_idx] = day_orb_low

        # Forward-fill NaN (start of day before range forms)
        last_h = close[0] if not np.isnan(close[0]) else 0.0
        last_l = close[0] if not np.isnan(close[0]) else 0.0
        for i in range(n):
            if not np.isnan(orb_high[i]):
                last_h = orb_high[i]
                last_l = orb_low[i]
            else:
                orb_high[i] = last_h
                orb_low[i] = last_l

        # ── VWAP (cumulative per day, no scaling needed) ─────────────────────
        typical_price = (high + low + close) / 3.0
        vwap = np.zeros(n)

        for d in np.unique(day_id):
            day_mask = day_id == d
            day_idx = np.where(day_mask)[0]

            tp_day = typical_price[day_idx]
            vol_day = np.where(volume[day_idx] > 0, volume[day_idx], 1.0)

            cum_tpv = np.cumsum(tp_day * vol_day)
            cum_vol = np.cumsum(vol_day)
            vwap[day_idx] = cum_tpv / cum_vol

        # ── Rolling volume ratio (10-min / 120-bar window) ───────────────────
        # Adapted from original's SMA(volume, 20) on 1-min = 20-min window.
        # Compressed to 10-min (120 bars) for 5s resolution.
        vol_mean_120 = np.zeros(n)
        for i in range(1, n):
            start = max(0, i - 120)
            seg = volume[start:i]
            seg_pos = seg[seg > 0]
            vol_mean_120[i] = np.mean(seg_pos) if len(seg_pos) > 0 else 1.0

        vol_mean_120[0] = max(volume[0], 1.0)
        vol_above = volume > (vol_ratio_thresh * vol_mean_120)

        # ── Breakout + 3-bar persistence filter ─────────────────────────────
        # 3-bar persistence is MANDATORY at 5s to filter sub-15s spike noise
        # that would be invisible in the original's 1-min bars.
        above_orb = close > orb_high
        below_orb = close < orb_low

        persist_above = np.zeros(n, dtype=bool)
        persist_below = np.zeros(n, dtype=bool)
        for i in range(2, n):
            if day_id[i] == day_id[i - 1] == day_id[i - 2]:
                persist_above[i] = above_orb[i] and above_orb[i - 1] and above_orb[i - 2]
                persist_below[i] = below_orb[i] and below_orb[i - 1] and below_orb[i - 2]

        # ── Session and VWAP filters ─────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        above_vwap = close > vwap
        below_vwap = close < vwap

        # ── Entry signals ────────────────────────────────────────────────────
        # Bullish: 3-bar persist above ORB + VWAP above + volume surge
        buy_ce = in_session & persist_above & above_vwap & vol_above

        # Bearish: 3-bar persist below ORB + VWAP below + volume surge
        buy_pe = in_session & persist_below & below_vwap & vol_above

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
