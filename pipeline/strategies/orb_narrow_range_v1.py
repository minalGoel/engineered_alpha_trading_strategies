"""orb_narrow_range_v1 — NR4 Opening Range Breakout on NIFTY.

Fires only on sessions where NIFTY's 15-minute opening range (09:15-09:30)
is the narrowest in 4 consecutive prior sessions (NR4 condition). The
cross-session compression signals withheld institutional aggression; the
subsequent breakout faces a thin order book at OR boundaries and tends to
extend 20-35 NIFTY spot points on the initial leg.

Entry: close breaks above OR_high (buy CE) or below OR_low (buy PE) with
bar body in upper/lower 25% and volume elevated vs OR baseline.
Hold: up to 120 seconds (24 bars). Active 09:30-12:00 IST.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "orb_narrow_range_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — after OR is established
    session_end_minutes = 720     # 12:00 IST — ORB momentum front-loaded
    max_trades_per_day = 3
    max_lookback = 240            # 20 min warmup for volume baseline

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", 1.3, 0.8, 2.0),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 9.0, 6.0, 15.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .cast(pl.Float64)
            .fill_null(0.0)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 9.0)
        vol_mult = params.get("vol_mult", 1.3)

        # ── VIX filter ────────────────────────────────────────────────────────
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

        # ── Per-day opening range (09:15-09:29:55 = time_min 555-569) ────────
        OR_OPEN = 555   # 09:15 IST
        OR_END = 570    # 09:30 IST (exclusive)

        sorted_days = sorted(set(day_id.tolist()))

        day_or_high: dict[int, float] = {}
        day_or_low: dict[int, float] = {}
        day_or_range: dict[int, float] = {}
        day_or_vol: dict[int, float] = {}

        for d in sorted_days:
            mask = (day_id == d) & (time_min >= OR_OPEN) & (time_min < OR_END)
            if mask.any():
                day_or_high[d] = float(np.max(high[mask]))
                day_or_low[d] = float(np.min(low[mask]))
                day_or_range[d] = day_or_high[d] - day_or_low[d]
                day_or_vol[d] = float(np.mean(volume[mask]))
            else:
                day_or_high[d] = 0.0
                day_or_low[d] = 0.0
                day_or_range[d] = 0.0
                day_or_vol[d] = 0.0

        # ── NR4: today's OR range < min of prior 4 sessions' OR ranges ───────
        day_is_nr4: dict[int, bool] = {}
        for i, d in enumerate(sorted_days):
            if i < 4:
                day_is_nr4[d] = False
                continue
            today_range = day_or_range[d]
            if today_range <= 0:
                day_is_nr4[d] = False
                continue
            prior_ranges = [
                day_or_range[sorted_days[j]]
                for j in range(i - 4, i)
                if day_or_range[sorted_days[j]] > 0
            ]
            if len(prior_ranges) == 4:
                day_is_nr4[d] = today_range < min(prior_ranges)
            else:
                day_is_nr4[d] = False

        # ── Build per-bar arrays ──────────────────────────────────────────────
        bar_or_high = np.zeros(n)
        bar_or_low = np.zeros(n)
        bar_or_vol = np.zeros(n)
        bar_is_nr4 = np.zeros(n, dtype=bool)

        for i in range(n):
            d = int(day_id[i])
            bar_or_high[i] = day_or_high.get(d, 0.0)
            bar_or_low[i] = day_or_low.get(d, 0.0)
            bar_or_vol[i] = day_or_vol.get(d, 0.0)
            bar_is_nr4[i] = day_is_nr4.get(d, False)

        # ── 3-bar rolling volume (15-second window) ───────────────────────────
        vol_3 = np.zeros(n)
        for i in range(3, n):
            vol_3[i] = np.mean(volume[i - 3:i])

        # ── Close position within bar (0 = at low, 1 = at high) ──────────────
        bar_range = high - low
        close_pct = np.where(bar_range > 0.0, (close - low) / bar_range, 0.5)

        # ── OR validity checks ────────────────────────────────────────────────
        orb_valid = (bar_or_high > 0.0) & (bar_or_low > 0.0) & (bar_or_high > bar_or_low)
        or_range_pct = np.where(
            bar_or_low > 0.0,
            (bar_or_high - bar_or_low) / bar_or_low * 100.0,
            0.0,
        )
        # OR range must be at least 0.05% and at most 2.0% of OR_low
        orb_reasonable = (or_range_pct >= 0.05) & (or_range_pct <= 2.0)

        # ── Filters ───────────────────────────────────────────────────────────
        vix_ok = vix_close < 25.0
        vol_elevated = (bar_or_vol > 0.0) & (vol_3 > vol_mult * bar_or_vol)
        entry_window = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        # Bullish: NR4 day, close breaks above OR_high, bar body in upper 25%
        buy_ce = (
            entry_window
            & bar_is_nr4
            & orb_valid
            & orb_reasonable
            & (close > bar_or_high)
            & (close_pct > 0.75)
            & vol_elevated
            & vix_ok
        )

        # Bearish: NR4 day, close breaks below OR_low, bar body in lower 25%
        buy_pe = (
            entry_window
            & bar_is_nr4
            & orb_valid
            & orb_reasonable
            & (close < bar_or_low)
            & (close_pct < 0.25)
            & vol_elevated
            & vix_ok
        )

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
