"""Bollinger Squeeze Reversion — NIFTY 5-second index options strategy.

When NIFTY bandwidth compresses to a 10-minute session low, HFT algorithms
probe the band edges. Probes that close outside the band without institutional
follow-through snap back toward the 20-bar SMA midline. Enter on the probe
bar itself; hold 30-90 seconds for the reversion.

Converted from: trading_strategies/unique_strategies_all/Strategy_21.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "bollinger_squeeze_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min opening noise
    session_end_minutes = 920     # 15:20 IST — flatten before illiquid close
    max_trades_per_day = 8
    max_lookback = 144            # 144 bars × 5s = 720s = 12 min (covers 120-bar pctile + 20-bar BB warmup + buffer)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bw_pctile_threshold", 20.0, 5.0, 35.0),
            TunableParam("vix_max", 22.0, 16.0, 28.0),
            TunableParam("stop_pts", 3.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        bw_pctile_threshold = params.get("bw_pctile_threshold", 20.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # ── VIX filter ──────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Bollinger Bands (20-bar = 100 seconds) ─────────────────────────
        bb_period = 20
        bb_mid = np.full(n, np.nan)
        bb_upper = np.full(n, np.nan)
        bb_lower = np.full(n, np.nan)
        bb_bandwidth = np.full(n, np.nan)

        for i in range(bb_period - 1, n):
            window = close[i - bb_period + 1: i + 1]
            mid = np.mean(window)
            std = np.std(window)
            bb_mid[i] = mid
            bb_upper[i] = mid + 2.0 * std
            bb_lower[i] = mid - 2.0 * std
            if mid > 0:
                bb_bandwidth[i] = (bb_upper[i] - bb_lower[i]) / mid

        # ── Bandwidth percentile rank (120-bar = 10 minutes) ───────────────
        pctile_period = 120
        bb_bw_pctile = np.full(n, 50.0)  # neutral default — not in squeeze

        for i in range(pctile_period - 1, n):
            window = bb_bandwidth[i - pctile_period + 1: i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) > 1:
                current = bb_bandwidth[i]
                if not np.isnan(current):
                    bb_bw_pctile[i] = float(np.mean(valid <= current) * 100.0)

        # ── Replace NaN band values with extreme sentinels (no false signals) ──
        # If bands are NaN (warmup), set them to values that cannot be triggered
        bb_upper_safe = np.where(np.isnan(bb_upper), close + 1e6, bb_upper)
        bb_lower_safe = np.where(np.isnan(bb_lower), close - 1e6, bb_lower)

        # ── Session and regime filters ──────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        squeeze_active = bb_bw_pctile < bw_pctile_threshold
        low_vix = vix_close < vix_max

        # ── Entry signals ───────────────────────────────────────────────────
        # buy_ce: close below lower band during squeeze → failed bearish probe → buy call (bullish reversion)
        buy_ce = in_session & squeeze_active & low_vix & (close < bb_lower_safe)

        # buy_pe: close above upper band during squeeze → failed bullish probe → buy put (bearish reversion)
        buy_pe = in_session & squeeze_active & low_vix & (close > bb_upper_safe)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,         # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
