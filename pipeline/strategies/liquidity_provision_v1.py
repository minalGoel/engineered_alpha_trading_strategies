"""Liquidity Provision via Tight VWAP Band Reversion on NIFTY.

Adapted from: trading_strategies/unique_strategies_all/Strategy_246.json
Original thesis: market-making around VWAP bands on NIFTY50 stocks.

Directional kernel: when NIFTY briefly dips below its tight VWAP lower band
(0.5σ, computed over 5 min), institutional VWAP algorithms accelerate buying
(fills better than their benchmark), driving a 15-90s reversion back toward VWAP.
Reverse at upper band. VIX < 18 and normal volume required.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "liquidity_provision_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip 10 min for VWAP stabilisation
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 15
    max_lookback = 120            # 10 min warmup (120 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("band_width", 0.5, 0.3, 0.9),
            TunableParam("dev_threshold_bps", 5.0, 2.0, 10.0),
            TunableParam("trend_abort_bps", 20.0, 12.0, 35.0),
            TunableParam("vol_ratio_min", 0.6, 0.4, 0.9),
            TunableParam("stop_pts", 3.0, 2.0, 5.0),
            TunableParam("target_pts", 5.0, 3.0, 8.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract numpy arrays (forward-fill in Polars first) ──────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ───────────────────────────────────────────────────────
        band_width = float(params.get("band_width", 0.5))
        dev_thresh = float(params.get("dev_threshold_bps", 5.0))
        trend_abort = float(params.get("trend_abort_bps", 20.0))
        vol_min = float(params.get("vol_ratio_min", 0.6))
        stop_pts = float(params.get("stop_pts", 3.0))
        target_pts = float(params.get("target_pts", 5.0))

        # ── Session VWAP (cumulative, resets each day) ────────────────────────
        vwap = np.zeros(n)
        cum_pv = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_pv = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            v = max(volume[i], 0.0)
            cum_pv += close[i] * v
            cum_vol += v
            vwap[i] = cum_pv / cum_vol if cum_vol > 0 else close[i]

        # ── VWAP deviation in bps ─────────────────────────────────────────────
        safe_vwap = np.where(vwap > 0, vwap, close)
        dev_bps = (close - safe_vwap) / safe_vwap * 10000.0

        # ── Rolling 60-bar (5-min) std of dev_bps ────────────────────────────
        std_window = 60
        dev_std = np.zeros(n)
        for i in range(std_window, n):
            dev_std[i] = float(np.std(dev_bps[i - std_window:i]))
        # Fill early bars with the first available std (or fallback)
        first_std = dev_std[std_window] if n > std_window else 5.0
        dev_std[:std_window] = first_std if first_std > 0.0 else 5.0

        # ── Volume ratio: 1-min rolling avg / 5-min rolling avg ──────────────
        win_fast, win_slow = 12, 60
        vol_fast = np.zeros(n)
        vol_slow = np.zeros(n)
        for i in range(win_fast, n):
            vol_fast[i] = float(np.mean(volume[i - win_fast:i]))
        for i in range(win_slow, n):
            vol_slow[i] = float(np.mean(volume[i - win_slow:i]))
        # For early bars default to 1.0 (neutral — assume normal participation)
        vol_fast[:win_fast] = 1.0
        vol_slow[:win_slow] = 1.0
        vol_ratio = np.where(vol_slow > 0, vol_fast / vol_slow, 1.0)

        # ── VIX close aligned to spot bars ───────────────────────────────────
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

        # ── Filters ───────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        vix_ok = vix_close < 18.0
        vol_ok = vol_ratio >= vol_min
        # Not in a strong intraday trend — within trend_abort bps of VWAP
        abs_dev = np.abs(dev_bps)
        not_trending = abs_dev < trend_abort

        base = in_session & vix_ok & vol_ok & not_trending

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: price below tight VWAP lower band → VWAP algos will buy back
        lower_band = -(band_width * dev_std)
        buy_ce = base & (dev_bps < lower_band) & (dev_bps < -dev_thresh)

        # buy_pe: price above tight VWAP upper band → VWAP sell algos dominate
        upper_band = band_width * dev_std
        buy_pe = base & (dev_bps > upper_band) & (dev_bps > dev_thresh)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,        # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
