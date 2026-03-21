"""Volume Breakout — NIFTY 5-second index options.

Thesis: When NIFTY price breaks above/below the 1-minute range high/low AND volume
spikes >2x the 5-minute baseline, directional institutional program order flow is
driving the move. These breakouts sustain for 30-90 seconds as the program continues
filling, creating a predictable continuation window for ATM call/put entries.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "volume_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min of opening noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 72             # 60 bars (5 min vol avg) + 12 bars range = 72 bars warmup

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_multiplier", 2.0, 1.2, 3.5),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays ──────────────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──────────────────────────────────────────────────────────
        vol_mult = params.get("vol_multiplier", 2.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Rolling 1-minute range (12 bars × 5s, exclusive of current bar) ────
        # range_high[i] = max(high[i-12 : i])  — no look-ahead
        range_high = np.full(n, np.nan)
        range_low = np.full(n, np.nan)
        for i in range(12, n):
            range_high[i] = np.max(high[i - 12:i])
            range_low[i] = np.min(low[i - 12:i])

        # ── 5-minute volume baseline (60 bars × 5s) ──────────────────────────
        avgvol = np.full(n, np.nan)
        for i in range(60, n):
            avgvol[i] = np.mean(volume[i - 60:i])

        # ── Session filter ───────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Volume surge filter ──────────────────────────────────────────────
        valid = ~np.isnan(range_high) & ~np.isnan(avgvol) & (avgvol > 0)
        vol_surge = valid & (volume > vol_mult * avgvol)

        # ── Breakout signals ─────────────────────────────────────────────────
        buy_ce = in_session & vol_surge & (close > range_high)
        buy_pe = in_session & vol_surge & (close < range_low)

        # Mutual exclusion: if both fire (unlikely but possible at boundaries), skip
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
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
