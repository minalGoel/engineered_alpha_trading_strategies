"""microstructure_volume_reversion_v1 — NIFTY index volume absorption signal.

On NIFTY, a 5-second bar with volume >3x the 5-minute rolling average AND range
<0.03% of spot price signals institutional order absorption (large VWAP/TWAP program
orders matched without moving price). The bar's direction (close vs open) reveals
which side is winning. Once the program completes, market makers delta-hedge and
momentum algos pile in, generating a 6-12 spot point directional move in 30-90 seconds.

Original: Strategy_27.json — volume spike absorption on individual NIFTY50 stocks
(1-min bars, 5x spike, <0.15% range, hold 5-20 min).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "microstructure_volume_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min (opening volume noise)
    session_end_minutes = 920     # 15:20 IST — flatten before EOD
    max_trades_per_day = 8
    max_lookback = 60             # 5-minute warmup for volume baseline

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Volume spike multiplier threshold (original used 5x on 1-min; 3x at 5s is equivalent)
            TunableParam("vol_threshold", 3.0, 2.0, 5.0),
            # Max bar range as % of open to qualify as absorption (not a genuine directional move)
            TunableParam("range_pct_threshold", 0.03, 0.01, 0.08),
            # VIX ceiling — above this, absorption signals are noise in macro panic
            TunableParam("vix_ceiling", 25.0, 18.0, 35.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .cast(pl.Float64)
            .fill_null(0.0)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──
        vol_threshold = params.get("vol_threshold", 3.0)
        range_pct_threshold = params.get("range_pct_threshold", 0.03)
        vix_ceiling = params.get("vix_ceiling", 25.0)

        # ── Rolling 60-bar (5-min) mean volume ──
        # 5-min window captures local activity level; shorter than original's 20-min
        # because 5s volume regimes shift faster than 1-min regimes.
        vol_mean_60 = np.full(n, np.nan)
        for i in range(60, n):
            vol_mean_60[i] = np.mean(volume[i - 60:i])

        # Fill early bars with session mean to avoid NaN signals during warmup
        overall_mean = float(np.mean(volume)) if np.any(volume > 0) else 1.0
        vol_mean_60 = np.where(np.isnan(vol_mean_60), overall_mean, vol_mean_60)

        # Guard against zero-volume baselines
        safe_mean = np.where(vol_mean_60 <= 0.0, 1.0, vol_mean_60)
        vol_ratio = volume / safe_mean

        # ── Bar range as % of open ──
        safe_open = np.where(np.abs(open_) < 1.0, 1.0, open_)
        bar_range_pct = np.abs(close - open_) / safe_open * 100.0

        # ── Bar direction: which side absorbed the spike ──
        bar_dir = np.sign(close - open_)  # +1 bullish, -1 bearish, 0 flat (exclude flat)

        # ── VIX filter: skip extreme volatility regimes ──
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

        # ── Signal construction ──
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        vix_ok = vix_close < vix_ceiling

        # Core absorption condition: high volume, tight range
        absorption = (vol_ratio > vol_threshold) & (bar_range_pct < range_pct_threshold)

        # Bullish absorption: buy-side absorbed supply (close > open)
        buy_ce = in_session & vix_ok & absorption & (bar_dir > 0)

        # Bearish absorption: sell-side absorbed demand (close < open)
        buy_pe = in_session & vix_ok & absorption & (bar_dir < 0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 3 pts (≈6 NIFTY spot pts at delta 0.5) — if resolution goes against us
            # by this much, the absorption thesis is broken.
            stop_points=np.full(n, 3.0),
            # Target: 5 pts (≈10 NIFTY spot pts) — post-absorption first-leg resolution
            # typically extends 8-14 spot pts in 30-90 seconds. 1:1.67 R:R.
            target_points=np.full(n, 5.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,            # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
