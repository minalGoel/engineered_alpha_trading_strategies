"""orb_gap_combo_v1 — ORB + Gap Dual Alignment on NIFTY

Thesis: When NIFTY gaps overnight AND the 15-min opening range breakout aligns with
that gap direction, institutional consensus is confirmed at two timescales.  This
dual alignment triggers cascading limit-order consumption above/below the ORB
boundary for 30-90 seconds.  Entry window: 09:30–10:40 IST only.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "orb_gap_combo_v1"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — ORB fully formed
    session_end_minutes = 640      # 10:40 IST — entry window closes
    max_trades_per_day = 3
    max_lookback = 240             # 20 min (240 × 5s) warmup for volume SMA

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold_pct", 0.15, 0.08, 0.35),
            TunableParam("vol_ratio_threshold", 1.2, 0.9, 2.0),
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
            TunableParam("vix_max", 22.0, 18.0, 30.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        open_arr = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .cast(pl.Float64)
            .fill_null(0.0)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        gap_threshold = params.get("gap_threshold_pct", 0.15)
        vol_ratio_thr = params.get("vol_ratio_threshold", 1.2)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)
        vix_max = params.get("vix_max", 22.0)

        # ── 1. Per-day gap_pct and ORB high/low ──
        # Build last-close and first-open per day in a single pass
        day_last_close: dict[int, float] = {}
        day_first_open: dict[int, float] = {}
        for i in range(n):
            d = int(day_id[i])
            day_last_close[d] = float(close[i])   # last write = last bar's close
            if d not in day_first_open:
                day_first_open[d] = float(open_arr[i])

        sorted_days = sorted(day_last_close.keys())

        # Gap pct: (day_open - prev_day_close) / prev_day_close * 100
        day_gap_pct: dict[int, float] = {}
        for idx, d in enumerate(sorted_days):
            if idx == 0:
                day_gap_pct[d] = 0.0        # no prev day available
            else:
                prev_d = sorted_days[idx - 1]
                prev_c = day_last_close.get(prev_d, 0.0)
                curr_o = day_first_open.get(d, 0.0)
                if prev_c != 0.0:
                    day_gap_pct[d] = (curr_o - prev_c) / prev_c * 100.0
                else:
                    day_gap_pct[d] = 0.0

        # ORB: per-day max(high) and min(low) for bars with time_min in [555, 570)
        unique_days = np.unique(day_id)
        day_orb_high: dict[int, float] = {}
        day_orb_low: dict[int, float] = {}
        for d in unique_days:
            d_int = int(d)
            mask = (day_id == d) & (time_min >= 555) & (time_min < 570)
            if mask.any():
                day_orb_high[d_int] = float(high[mask].max())
                day_orb_low[d_int] = float(low[mask].min())
            else:
                day_orb_high[d_int] = 0.0
                day_orb_low[d_int] = 1e9

        # Fill per-bar arrays
        gap_pct_arr = np.zeros(n)
        orb_high_arr = np.zeros(n)
        orb_low_arr = np.full(n, 1e9)
        for i in range(n):
            d = int(day_id[i])
            gap_pct_arr[i] = day_gap_pct.get(d, 0.0)
            orb_high_arr[i] = day_orb_high.get(d, 0.0)
            orb_low_arr[i] = day_orb_low.get(d, 1e9)

        # ── 2. Rolling 240-bar volume SMA (20 min) ──
        vol_sma = np.zeros(n)
        running_sum = 0.0
        window = 240
        for i in range(n):
            running_sum += volume[i]
            if i >= window:
                running_sum -= volume[i - window]
                vol_sma[i] = running_sum / window
            else:
                vol_sma[i] = running_sum / (i + 1)

        vol_ratio = np.where(vol_sma > 0.0, volume / vol_sma, 0.0)

        # ── 3. VIX filter ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        vix_ok = (vix_close >= 12.0) & (vix_close <= vix_max)

        # ── 4. Signal conditions ──
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # Gap direction
        gap_up = gap_pct_arr > gap_threshold
        gap_down = gap_pct_arr < -gap_threshold

        # ORB breakout (valid only when ORB levels are computed)
        orb_break_up = (close > orb_high_arr) & (orb_high_arr > 0.0)
        orb_break_down = (close < orb_low_arr) & (orb_low_arr < 1e8)

        # 2-bar persistence filter (10 seconds) — eliminates 5s spike noise
        persist_up = np.zeros(n, dtype=bool)
        persist_down = np.zeros(n, dtype=bool)
        for i in range(1, n):
            persist_up[i] = bool(orb_break_up[i]) and bool(orb_break_up[i - 1])
            persist_down[i] = bool(orb_break_down[i]) and bool(orb_break_down[i - 1])

        # Volume confirmation
        vol_confirm = vol_ratio >= vol_ratio_thr

        # Combined: gap direction must match ORB breakout direction
        buy_ce = in_session & gap_up & persist_up & vol_confirm & vix_ok
        buy_pe = in_session & gap_down & persist_down & vol_confirm & vix_ok

        # Mutual exclusion (should not occur given gap_up / gap_down are exclusive)
        both = buy_ce & buy_pe
        buy_ce = buy_ce & ~both
        buy_pe = buy_pe & ~both

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,            # 90 seconds = first impulse leg of gap+ORB continuation
            max_trades_per_day=self.max_trades_per_day,
        )
