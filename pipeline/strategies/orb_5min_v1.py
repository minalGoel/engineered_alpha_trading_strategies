"""orb_5min_v1 — 5-Minute Opening Range Breakout on NIFTY Index Options.

Mechanism:
    NIFTY's 5-minute opening range (09:15-09:20, 60 bars at 5s) is formed by the most
    aggressive overnight-positioned participants with strong pre-open directional convictions.
    When NIFTY's close breaks above/below this tight range with volume > 1.5x the recent
    session pace and VWAP confirmation, it signals that aggressive institutional flow has
    overwhelmed the early opening supply/demand, triggering a 10-20 spot point continuation
    burst within 30-90 seconds before the full 20-minute institutional window takes over.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "orb_5min_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — breakout window opens after range forms
    session_end_minutes = 925     # 15:25 IST — EOD flatten (signal filtered to 09:20-09:35)
    max_trades_per_day = 4
    max_lookback = 120            # 10 min warmup (120 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("orb_range_max", 80.0, 40.0, 150.0),      # max ORB range in NIFTY pts
            TunableParam("volume_ratio_min", 1.5, 1.0, 3.0),        # min vol ratio for breakout
            TunableParam("stop_pts", 3.0, 2.0, 7.0),                # stop in option premium pts
            TunableParam("target_pts", 6.0, 4.0, 12.0),             # target in option premium pts
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Forward-fill NaN in Polars before converting to numpy ---
        spot_filled = spot_df.with_columns([
            pl.col("close").fill_null(strategy="forward"),
            pl.col("high").fill_null(strategy="forward"),
            pl.col("low").fill_null(strategy="forward"),
            pl.col("volume").fill_null(0).cast(pl.Float64),
        ])

        close = spot_filled["close"].to_numpy()
        volume = spot_filled["volume"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # --- Parameters ---
        orb_range_max = params.get("orb_range_max", 80.0)
        volume_ratio_min = params.get("volume_ratio_min", 1.5)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # --- ORB: compute per-day high/low from 09:15-09:20 (time_minutes 555-559) ---
        orb_agg = (
            spot_df
            .filter(
                (pl.col("time_minutes") >= 555) & (pl.col("time_minutes") < 560)
            )
            .with_columns([
                pl.col("high").fill_null(strategy="forward"),
                pl.col("low").fill_null(strategy="forward"),
            ])
            .group_by("day_id")
            .agg([
                pl.col("high").max().alias("orb_high"),
                pl.col("low").min().alias("orb_low"),
            ])
        )

        # Join ORB back to full dataframe (preserves row order of left frame)
        spot_with_orb = spot_df.select(["day_id"]).join(orb_agg, on="day_id", how="left")
        orb_high = spot_with_orb["orb_high"].fill_null(0.0).to_numpy()
        orb_low = spot_with_orb["orb_low"].fill_null(0.0).to_numpy()
        orb_range = orb_high - orb_low

        # --- VWAP: cumulative intraday (09:15 onward, grouped by day_id) ---
        vwap_arr = (
            spot_filled
            .with_columns([
                (pl.col("close") * pl.col("volume")).alias("_pv"),
            ])
            .with_columns([
                pl.col("_pv").cum_sum().over("day_id").alias("_cum_pv"),
                pl.col("volume").cum_sum().over("day_id").alias("_cum_vol"),
            ])
            .with_columns([
                (pl.col("_cum_pv") / pl.col("_cum_vol").clip(lower_bound=1.0)).alias("_vwap"),
            ])
        )["_vwap"].fill_null(strategy="forward").fill_null(0.0).to_numpy()

        # --- Volume ratio: current bar vs 60-bar rolling mean (5-minute session pace) ---
        vol_ma60 = np.zeros(n)
        for i in range(n):
            start = max(0, i - 60)
            count = i - start
            vol_ma60[i] = np.mean(volume[start:i]) if count > 0 else (volume[i] if volume[i] > 0 else 1.0)
        vol_ratio = np.where(vol_ma60 > 0.0, volume / np.maximum(vol_ma60, 1.0), 1.0)

        # --- Signal conditions ---
        # Breakout window: 09:20-09:35 only (560-575 minutes) — signal decays after 09:35
        in_window = (time_min >= 560) & (time_min < 575)

        # ORB validity: range formed, not too large, not too small
        orb_valid = (orb_high > 0) & (orb_low > 0) & (orb_range <= orb_range_max) & (orb_range > 5.0)

        # Volume confirmation
        vol_ok = vol_ratio > volume_ratio_min

        # Buy CE: close breaks above ORB_high, above VWAP, volume confirmed
        buy_ce = in_window & orb_valid & (close > orb_high) & (close > vwap_arr) & vol_ok

        # Buy PE: close breaks below ORB_low, below VWAP, volume confirmed
        buy_pe = in_window & orb_valid & (close < orb_low) & (close < vwap_arr) & vol_ok

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
