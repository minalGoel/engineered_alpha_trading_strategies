"""ORB 30-Minute Volume Confirmation — orb_30min_volume_v1

NIFTY's 30-minute opening range (09:15-09:45) represents the full initial
price discovery period. Breakouts above/below this range confirmed by volume
surge (>1.2x recent 5-min average) and VWAP alignment signal institutional
order flow overwhelming the range boundary, creating 20-40 spot point momentum
pushes in the first 60-120 seconds post-breakout. 3-bar (15s) persistence
filter eliminates false 5-second spikes.

Entry window: 09:45-11:00 IST only (within 75 min of range completion).
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "orb_30min_volume_v1"
    underlying = "NIFTY"
    session_start_minutes = 585   # 09:45 IST — after 30-min range completes
    session_end_minutes = 660     # 11:00 IST — ORB thesis expires after 75 min
    max_trades_per_day = 3
    max_lookback = 360            # 30 min warmup for opening range (360 × 5s bars)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_surge_threshold", 1.2, 0.8, 2.0),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        vol_surge_threshold = params.get("vol_surge_threshold", 1.2)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── Per-day Opening Range (09:15-09:44 IST, time_minutes 555-584) ──
        # Fixed time window: 30 calendar minutes = 360 bars at 5s
        orb_high = np.full(n, np.nan)
        orb_low = np.full(n, np.nan)

        unique_days = np.unique(day_id)
        day_orb_high = {}
        day_orb_low = {}

        for d in unique_days:
            range_mask = (day_id == d) & (time_min >= 555) & (time_min < 585)
            if np.any(range_mask):
                day_orb_high[d] = float(np.nanmax(high[range_mask]))
                day_orb_low[d] = float(np.nanmin(low[range_mask]))
            else:
                day_orb_high[d] = np.nan
                day_orb_low[d] = np.nan

        for i in range(n):
            d = day_id[i]
            orb_high[i] = day_orb_high.get(d, np.nan)
            orb_low[i] = day_orb_low.get(d, np.nan)

        # ── Intraday VWAP (cumulative from 09:15 per day) ──
        vwap = np.full(n, np.nan)
        for d in unique_days:
            day_mask = (day_id == d) & (time_min >= 555)
            idxs = np.where(day_mask)[0]
            if len(idxs) == 0:
                continue
            cum_vol = 0.0
            cum_tpv = 0.0
            for i in idxs:
                bar_vol = volume[i]
                cum_vol += bar_vol
                cum_tpv += close[i] * bar_vol
                vwap[i] = cum_tpv / cum_vol if cum_vol > 0 else close[i]

        # ── Rolling 5-minute (60-bar) volume average ──
        vol_avg_60 = np.full(n, np.nan)
        for i in range(60, n):
            vol_avg_60[i] = np.mean(volume[i - 60:i])

        # ── 3-bar (15s) persistence filter ──
        # All 3 bars must close above ORB high (or below ORB low) for a valid breakout
        above_orb_3bar = np.zeros(n, dtype=bool)
        below_orb_3bar = np.zeros(n, dtype=bool)
        for i in range(2, n):
            if not np.isnan(orb_high[i]) and not np.isnan(orb_low[i]):
                above_orb_3bar[i] = (
                    close[i] > orb_high[i]
                    and close[i - 1] > orb_high[i]
                    and close[i - 2] > orb_high[i]
                )
                below_orb_3bar[i] = (
                    close[i] < orb_low[i]
                    and close[i - 1] < orb_low[i]
                    and close[i - 2] < orb_low[i]
                )

        # ── Volume surge: current bar volume > threshold × recent 5-min avg ──
        vol_surge = np.zeros(n, dtype=bool)
        for i in range(60, n):
            avg = vol_avg_60[i]
            if avg > 0:
                vol_surge[i] = volume[i] >= vol_surge_threshold * avg

        # ── Safe arrays for comparisons (replace NaN with neutral) ──
        vwap_safe = np.where(np.isnan(vwap), close, vwap)
        orb_high_safe = np.where(np.isnan(orb_high), np.inf, orb_high)
        orb_low_safe = np.where(np.isnan(orb_low), -np.inf, orb_low)

        # ── Entry window: 09:45-11:00 IST ──
        in_entry_window = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ──
        # Bullish: price has been above ORB high for 3 bars, volume surge, above VWAP
        buy_ce = (
            in_entry_window
            & above_orb_3bar
            & vol_surge
            & (close > vwap_safe)
        )

        # Bearish: price has been below ORB low for 3 bars, volume surge, below VWAP
        buy_pe = (
            in_entry_window
            & below_orb_3bar
            & vol_surge
            & (close < vwap_safe)
        )

        # Mutual exclusion — should not occur given ORB structure, but guard anyway
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
            time_stop_bars=24,           # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
