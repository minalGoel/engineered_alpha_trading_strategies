"""open_high_low_v1 — One-Sided Bar Momentum on NIFTY

Mechanism:
    On NIFTY, a 5-second bar where open equals low (every tick at or above open)
    reveals aggressive institutional buying that absorbed all resting sell orders at
    that price level. When accompanied by elevated relative volume (>1.5x the 5-minute
    rolling mean), it confirms genuine demand — not thin-market drift. Institutional
    TWAP algorithms executing buy orders in NIFTY futures create these patterns as they
    systematically lift offers, driving the index 8-15 spot points over 30-90 seconds.
    The complementary open=high bar signals the equivalent sell-side imbalance.

Converted from: trading_strategies/unique_strategies_all/Strategy_13.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "open_high_low_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 10
    max_lookback = 60             # 60 bars = 5 min warmup for rolling volume mean

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rel_volume_threshold", 1.5, 1.0, 3.0),
            TunableParam("vix_max", 21.0, 15.0, 28.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Extract spot columns (forward-fill NaN before converting) ---
        open_p = spot_df["open"].fill_null(strategy="forward").fill_null(0.0).to_numpy()
        high_p = spot_df["high"].fill_null(strategy="forward").fill_null(0.0).to_numpy()
        low_p = spot_df["low"].fill_null(strategy="forward").fill_null(0.0).to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        # --- VIX (aligned to spot bars via asof join) ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- Rolling 60-bar mean of volume (O(n) via cumsum) ---
        vol_cumsum = np.zeros(n + 1)
        vol_cumsum[1:] = np.cumsum(volume)

        vol_sma = np.zeros(n)
        for i in range(1, n):
            start = max(0, i - 60)
            count = i - start
            vol_sma[i] = (vol_cumsum[i] - vol_cumsum[start]) / count

        # Relative volume (guard division by zero)
        rel_vol = np.zeros(n)
        positive = vol_sma > 0
        rel_vol[positive] = volume[positive] / vol_sma[positive]

        # --- Parameters ---
        rel_vol_thresh = params.get("rel_volume_threshold", 1.5)
        vix_max = params.get("vix_max", 21.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # --- One-sided bar detection (tolerance 0.1 pts for float precision) ---
        tol = 0.1
        open_eq_low = np.abs(open_p - low_p) <= tol    # bullish: never traded below open
        open_eq_high = np.abs(open_p - high_p) <= tol  # bearish: never traded above open

        # Doji (flat bar): open=high=low — exclude because both flags fire simultaneously
        doji = open_eq_low & open_eq_high

        # --- Filters ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < vix_max
        vol_ok = rel_vol >= rel_vol_thresh

        # --- Signals ---
        buy_ce = in_session & vix_ok & vol_ok & open_eq_low & ~doji
        buy_pe = in_session & vix_ok & vol_ok & open_eq_high & ~doji

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds = 18 bars × 5s
            max_trades_per_day=self.max_trades_per_day,
        )
