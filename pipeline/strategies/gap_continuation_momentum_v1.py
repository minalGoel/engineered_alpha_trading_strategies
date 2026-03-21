"""
gap_continuation_momentum_v1 — NIFTY Gap + ORB Continuation

Mechanism: When NIFTY gaps up > 0.4% at open (FII/global-cue driven), the first 15
minutes (09:15-09:30) form the opening range as the morning order queue absorbs. When
NIFTY breaks above this ORB high (with VWAP and volume confirmation), VWAP-benchmarked
algorithms with unfinished buy programs trigger simultaneously, creating a 30-90 second
directional burst. We enter at the exact 5-second ORB breakout bar, not a 1-min
confirmation, capturing the first institutional flow resumption.

Stop: 4 option pts (~8 NIFTY spot pts) — reversal back inside ORB invalidates thesis.
Target: 7 option pts (~14 NIFTY spot pts) — lower end of typical 15-20 pt post-ORB burst.
Hold: up to 18 bars (90 seconds).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "gap_continuation_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 555   # 09:15 — need full ORB window from market open
    session_end_minutes = 925     # 15:25 — EOD flatten
    max_trades_per_day = 4
    max_lookback = 180            # 15 min warmup (180 bars × 5s) for ORB formation

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.4, 0.2, 0.8),   # % gap to qualify
            TunableParam("vol_ratio", 1.2, 1.0, 2.0),        # volume multiplier for confirmation
            TunableParam("stop_pts", 4.0, 2.0, 7.0),
            TunableParam("target_pts", 8.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays ──────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ──────────────────────────────────────────────────────
        gap_threshold = params.get("gap_threshold", 0.4)
        vol_ratio = params.get("vol_ratio", 1.2)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # ── Gap % per day ───────────────────────────────────────────────────
        # gap_pct[i] = (day_open - prev_day_close) / prev_day_close * 100
        gap_pct = np.zeros(n)
        days = np.unique(day_id)
        day_last_close: dict[int, float] = {}

        for idx_d, d in enumerate(days):
            mask = day_id == d
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            day_open_price = open_[idxs[0]]
            if idx_d > 0:
                prev_d = days[idx_d - 1]
                prev_close = day_last_close.get(int(prev_d), day_open_price)
                if prev_close != 0:
                    gap_pct[idxs] = (day_open_price - prev_close) / prev_close * 100.0
            day_last_close[int(d)] = close[idxs[-1]]

        # ── ORB (09:15-09:30 = time_minutes in [555, 570)) ─────────────────
        orb_high = np.full(n, np.inf)
        orb_low = np.full(n, -np.inf)
        orb_formed = np.zeros(n, dtype=bool)

        for d in days:
            mask_d = day_id == d
            orb_mask = mask_d & (time_min < 570)          # 09:15-09:30 ORB window
            post_mask = mask_d & (time_min >= 570)        # after ORB is complete

            orb_idxs = np.where(orb_mask)[0]
            post_idxs = np.where(post_mask)[0]

            if len(orb_idxs) == 0:
                continue

            orb_h = np.max(high[orb_idxs])
            orb_l = np.min(low[orb_idxs])

            if len(post_idxs) > 0:
                orb_high[post_idxs] = orb_h
                orb_low[post_idxs] = orb_l
                orb_formed[post_idxs] = True

        # ── Session VWAP (cumulative from 09:15 each day) ──────────────────
        vwap = np.zeros(n)

        for d in days:
            mask_d = day_id == d
            idxs = np.where(mask_d)[0]
            if len(idxs) == 0:
                continue
            tp = (high[idxs] + low[idxs] + close[idxs]) / 3.0
            vol = volume[idxs]
            cum_tp_vol = np.cumsum(tp * vol)
            cum_vol = np.cumsum(vol)
            with np.errstate(divide="ignore", invalid="ignore"):
                vwap[idxs] = np.where(cum_vol > 0, cum_tp_vol / cum_vol, close[idxs])

        # ── Rolling volume SMA (60 bars = 5 min) ───────────────────────────
        vol_sma = np.zeros(n)
        for i in range(60, n):
            vol_sma[i] = np.mean(volume[i - 60 : i])
        # For first 60 bars, use cumulative mean as best estimate
        for i in range(1, min(60, n)):
            vol_sma[i] = np.mean(volume[: i + 1])
        if n > 0:
            vol_sma[0] = volume[0] if volume[0] > 0 else 1.0

        vol_sma = np.where(vol_sma <= 0, 1.0, vol_sma)   # avoid divide-by-zero

        # ── Entry time filter: only 09:30-10:30 ────────────────────────────
        # time_min 570 = 09:30, 630 = 10:30
        in_entry_window = (time_min >= 570) & (time_min < 630)

        # ── Signals ────────────────────────────────────────────────────────
        # Bull: gap-up day, close breaks above ORB high, above VWAP, volume surge
        buy_ce = (
            in_entry_window
            & orb_formed
            & (gap_pct > gap_threshold)
            & (close > orb_high)
            & (close > vwap)
            & (volume > vol_sma * vol_ratio)
        )

        # Bear: gap-down day, close breaks below ORB low, below VWAP, volume surge
        buy_pe = (
            in_entry_window
            & orb_formed
            & (gap_pct < -gap_threshold)
            & (close < orb_low)
            & (close < vwap)
            & (volume > vol_sma * vol_ratio)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
