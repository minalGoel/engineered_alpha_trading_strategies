"""15-Minute Opening Range Breakout with Width Filter — orb_15min_v1

Converts: trading_strategies/unique_strategies_all/Strategy_69.json
Original: 15-min ORB on NIFTY 50 FNO stocks, 1-min bars, width/volume/VIX filters.

Mechanism: On NIFTY, when the 09:15-09:29 range is tight (0.3-1.5% width), it signals
genuine coiled supply/demand equilibrium. A close above the ORB high with a volume surge
(>1.5x 20-min rolling average) means institutional buy programs have cleared the supply
ceiling, triggering momentum continuation at 5-second resolution. Key differentiator from
orb_momentum_v22: the width filter (0.3-1.5%) ensures we only trade on 'coiled spring'
sessions, not wide-gap or indecisive-range days. VIX < 25 guard and 2-bar persistence
prevent false triggers.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "orb_15min_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — entry only after ORB forms
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 4
    max_lookback = 240            # 20-min warmup for volume SMA (240 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", 1.5, 1.0, 2.5),
            TunableParam("orb_width_min_pct", 0.3, 0.1, 0.6),
            TunableParam("orb_width_max_pct", 1.5, 0.8, 2.5),
            TunableParam("vix_max", 25.0, 18.0, 32.0),
            TunableParam("stop_pts", 3.0, 1.5, 6.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy()
        day_id = spot_df["day_id"].fill_null(0).to_numpy()

        vol_mult = params.get("vol_mult", 1.5)
        orb_width_min = params.get("orb_width_min_pct", 0.3)
        orb_width_max = params.get("orb_width_max_pct", 1.5)
        vix_max = params.get("vix_max", 25.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # --- ORB: fixed 09:15-09:29 window ---
        # time_minutes: 555 = 09:15, 569 = 09:29, 570 = 09:30
        ORB_START = 555  # 09:15 IST (inclusive)
        ORB_END = 570    # 09:30 IST (exclusive)

        orb_high = np.full(n, np.nan)
        orb_low = np.full(n, np.nan)
        orb_width_pct = np.full(n, np.nan)

        for d in np.unique(day_id):
            mask_day = day_id == d
            mask_orb = mask_day & (time_min >= ORB_START) & (time_min < ORB_END)
            if not mask_orb.any():
                continue
            day_orb_high = float(np.nanmax(high[mask_orb]))
            day_orb_low = float(np.nanmin(low[mask_orb]))
            if day_orb_low <= 0.0:
                continue
            day_width_pct = (day_orb_high - day_orb_low) / day_orb_low * 100.0
            # Carry ORB levels forward for all post-ORB bars on this day
            mask_post = mask_day & (time_min >= ORB_END)
            orb_high[mask_post] = day_orb_high
            orb_low[mask_post] = day_orb_low
            orb_width_pct[mask_post] = day_width_pct

        # --- Rolling 20-min volume SMA (240 bars) ---
        # Uses cumulative sum for O(n) computation
        vol_sma = np.zeros(n)
        vol_window = 240
        cumvol = np.concatenate([[0.0], np.cumsum(volume)])
        for i in range(1, n):
            start = max(0, i - vol_window)
            count = i - start
            vol_sma[i] = (cumvol[i] - cumvol[start]) / count if count > 0 else 0.0

        # --- VIX filter: align VIX close to spot bars ---
        vix_close = np.full(n, 15.0)  # neutral default (pass-through)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime", strategy="backward"
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- 2-bar persistence filter ---
        # close must be above orb_high for both current AND previous bar (10 seconds)
        prev_close = np.empty(n)
        prev_close[0] = close[0]
        prev_close[1:] = close[:-1]

        # --- ORB validity masks ---
        has_orb = ~np.isnan(orb_high) & ~np.isnan(orb_low)
        width_ok = (
            has_orb
            & (orb_width_pct >= orb_width_min)
            & (orb_width_pct <= orb_width_max)
        )

        # --- Volume surge ---
        vol_ok = (vol_sma > 0) & (volume > vol_mult * vol_sma)

        # --- VIX regime ---
        vix_ok = vix_close < vix_max

        # --- Time: entry window 09:30 (570) to 12:00 (720) ---
        ENTRY_END = 720  # 12:00 IST
        entry_window = (time_min >= self.session_start_minutes) & (time_min < ENTRY_END)

        # --- Entry signals ---
        # buy_ce: 2-bar persistence above ORB high + volume + width + VIX
        above_orb_high = close > orb_high
        prev_above_orb_high = prev_close > orb_high
        buy_ce = (
            entry_window
            & width_ok
            & vol_ok
            & vix_ok
            & above_orb_high
            & prev_above_orb_high
        )

        # buy_pe: 2-bar persistence below ORB low + volume + width + VIX
        below_orb_low = close < orb_low
        prev_below_orb_low = prev_close < orb_low
        buy_pe = (
            entry_window
            & width_ok
            & vol_ok
            & vix_ok
            & below_orb_low
            & prev_below_orb_low
        )

        # Ensure no NaN in signal arrays
        buy_ce = np.where(np.isnan(close), False, buy_ce)
        buy_pe = np.where(np.isnan(close), False, buy_pe)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds — tight ORB breakouts resolve quickly
            max_trades_per_day=self.max_trades_per_day,
        )
