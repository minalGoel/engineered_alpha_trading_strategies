"""gap_continuation_v1 — NIFTY Gap Continuation Breakout

Original: equity gap continuation on NIFTY200 stocks (1-min bars, 60-240 min hold).
Converted: NIFTY index gap continuation at 5-second resolution, 30-90 second hold.

Mechanism: When NIFTY gaps >0.7% (driven by SGX Nifty/global macro) and holds the
gap un-filled for the first 30 minutes, a breakout above (gap-up) or below (gap-down)
the first 30-min range at 09:45+ triggers simultaneous VWAP-algorithm entries and
creates a 30-90 second momentum burst capturable at 5s resolution.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "gap_continuation_v1"
    underlying = "NIFTY"
    session_start_minutes = 585   # 09:45 IST — after opening range forms
    session_end_minutes = 660     # 11:00 IST — gap continuation window
    max_trades_per_day = 3
    max_lookback = 360            # 30-min warmup for opening range (360 x 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.7, 0.3, 1.5),   # min gap % to trade
            TunableParam("vol_mult", 1.2, 1.0, 2.0),         # volume surge multiplier
            TunableParam("vix_max", 25.0, 18.0, 30.0),       # max VIX to trade
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .fill_null(0)
            .cast(pl.Float64)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_ids = spot_df["day_id"].to_numpy()

        gap_threshold = params.get("gap_threshold", 0.7)
        vol_mult = params.get("vol_mult", 1.2)
        vix_max = params.get("vix_max", 25.0)

        # ── VIX filter ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")])
                .sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Per-day opening-range stats ──
        gap_pct_arr = np.zeros(n)
        gap_filled_arr = np.ones(n, dtype=bool)   # default: filled → no trade
        f30_high_arr = np.full(n, np.inf)
        f30_low_arr = np.full(n, -np.inf)

        unique_days = np.unique(day_ids)

        for i, d in enumerate(unique_days):
            day_mask = day_ids == d
            day_idx = np.where(day_mask)[0]

            # Session open (first bar's open)
            day_open = open_[day_idx[0]]

            # Previous day close
            if i > 0:
                prev_d = unique_days[i - 1]
                prev_idx = np.where(day_ids == prev_d)[0]
                prev_close_val = close[prev_idx[-1]]
            else:
                prev_close_val = day_open  # no gap info on first day

            # Gap %
            if prev_close_val != 0:
                gap = (day_open - prev_close_val) / prev_close_val * 100.0
            else:
                gap = 0.0

            # First 30-min window: 09:15-09:44 IST = minutes 555-584
            f30_mask = day_mask & (time_min >= 555) & (time_min < 585)
            f30_idx = np.where(f30_mask)[0]

            if len(f30_idx) > 0:
                f30h = float(np.max(high[f30_idx]))
                f30l = float(np.min(low[f30_idx]))
                # Gap fill check
                if gap > 0:
                    filled = bool(np.any(low[f30_idx] <= prev_close_val))
                elif gap < 0:
                    filled = bool(np.any(high[f30_idx] >= prev_close_val))
                else:
                    filled = True
            else:
                f30h = day_open
                f30l = day_open
                filled = True

            gap_pct_arr[day_idx] = gap
            gap_filled_arr[day_idx] = filled
            f30_high_arr[day_idx] = f30h
            f30_low_arr[day_idx] = f30l

        # ── Rolling 20-min volume SMA (240 bars x 5s = 20 min) ──
        vol_window = 240
        vol_sma = np.zeros(n)
        for i in range(1, n):
            start = max(0, i - vol_window)
            vol_sma[i] = np.mean(volume[start:i]) if i > 0 else volume[0]

        # ── Signal conditions ──
        in_entry_window = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        large_gap_up = gap_pct_arr > gap_threshold
        large_gap_dn = gap_pct_arr < -gap_threshold
        gap_intact = ~gap_filled_arr
        vol_surge = (vol_sma > 0) & (volume > vol_sma * vol_mult)
        vix_ok = vix_close < vix_max

        # Gap must be < 2% to avoid circuit-breaker territory
        gap_not_extreme = np.abs(gap_pct_arr) < 2.0

        breakout_up = close > f30_high_arr
        breakdown_dn = close < f30_low_arr

        buy_ce = (
            in_entry_window
            & large_gap_up
            & gap_intact
            & gap_not_extreme
            & breakout_up
            & vol_surge
            & vix_ok
        )
        buy_pe = (
            in_entry_window
            & large_gap_dn
            & gap_intact
            & gap_not_extreme
            & breakdown_dn
            & vol_surge
            & vix_ok
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4),
            target_points=np.full(n, 8),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,        # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
