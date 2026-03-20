"""PDH/PDL Fakeout Strategy — v141

On NIFTY, Previous Day High/Low are the most-referenced technical levels. When the
index probes above PDH (or below PDL) and closes back within range on a 5-second bar,
it signals a failed liquidity hunt. Option market makers short gamma near these levels
delta-hedge aggressively on reversal, amplifying the move back through PDH/PDL within
30-90 seconds.

buy_pe: high > PDH, close < PDH, breach < max_breach_pts, VIX < vix_max
buy_ce: low < PDL, close > PDL, breach < max_breach_pts, VIX < vix_max
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "pdh_pdl_fakeout_v141"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip pre-open noise
    session_end_minutes = 920     # 15:20 IST — flatten before EOD
    max_trades_per_day = 6
    # Need ~1 full previous trading day of data to compute PDH/PDL
    # 375 min × 12 bars/min = 4500 bars; set 4800 for buffer
    max_lookback = 4800

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("max_breach_pts", 15.0, 5.0, 30.0),
            TunableParam("vix_max", 22.0, 15.0, 28.0),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        max_breach = params.get("max_breach_pts", 15.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── VIX filter ──────────────────────────────────────────────────────────
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

        # ── Compute PDH / PDL from previous trading day ─────────────────────────
        # Build a map: day_id → (day_high, day_low) using all bars of that day
        unique_days = np.unique(day_id)
        day_high_map: dict[int, float] = {}
        day_low_map: dict[int, float] = {}
        for d in unique_days:
            mask = day_id == d
            day_high_map[int(d)] = float(np.max(high[mask]))
            day_low_map[int(d)] = float(np.min(low[mask]))

        # For each bar, look up the PREVIOUS day's high/low
        pdh = np.full(n, np.nan)
        pdl = np.full(n, np.nan)
        for i in range(n):
            prev = int(day_id[i]) - 1
            if prev in day_high_map:
                pdh[i] = day_high_map[prev]
                pdl[i] = day_low_map[prev]

        # Valid mask — only trade on days where we have previous-day reference
        pdh_valid = np.isfinite(pdh)
        pdl_valid = np.isfinite(pdl)

        # Replace NaN with sentinel values that prevent false triggers
        pdh_safe = np.where(pdh_valid, pdh, np.inf)
        pdl_safe = np.where(pdl_valid, pdl, -np.inf)

        # ── Session and VIX filters ─────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        vix_ok = vix_close < vix_max

        # ── PDH fakeout → buy PE (bearish reversal) ─────────────────────────────
        # Bar probed above PDH but closed back below — rejection confirmed
        breach_up = np.maximum(high - pdh_safe, 0.0)
        pdh_fakeout = (
            (high > pdh_safe)
            & (close < pdh_safe)
            & (breach_up < max_breach)
        )

        # ── PDL fakeout → buy CE (bullish reversal) ─────────────────────────────
        # Bar probed below PDL but closed back above — rejection confirmed
        breach_dn = np.maximum(pdl_safe - low, 0.0)
        pdl_fakeout = (
            (low < pdl_safe)
            & (close > pdl_safe)
            & (breach_dn < max_breach)
        )

        # ── Final signals ───────────────────────────────────────────────────────
        buy_pe = in_session & vix_ok & pdh_fakeout & pdh_valid
        buy_ce = in_session & vix_ok & pdl_fakeout & pdl_valid

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
