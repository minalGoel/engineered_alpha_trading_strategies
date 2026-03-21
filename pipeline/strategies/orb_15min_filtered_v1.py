"""
orb_15min_filtered_v1 — 15-Minute Opening Range Breakout with VIX Filter on NIFTY

Mechanism:
  NIFTY's first 15 minutes (09:15-09:30) concentrates overnight gap resolution and
  institutional limit-book repricing. When India VIX sits in the 13-20 Goldilocks zone,
  breakouts from this range trigger VWAP-benchmarked programs and CTA momentum algos that
  sustain the first impulse for 30-90 seconds. Below VIX 13, ranges are too compressed and
  market makers fade breakouts trivially. Above VIX 20, institutional flow is erratic and
  gap risk dominates. A 2-bar (10-second) persistence filter at 5s resolution replaces the
  original 2-minute confirmation, preserving first-mover advantage while filtering noise spikes.
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "orb_15min_filtered_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — after 15-min range forms
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 5
    max_lookback = 240            # 20-min warmup covers range formation + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_low", 13.0, 11.0, 16.0),
            TunableParam("vix_high", 20.0, 17.0, 25.0),
            TunableParam("vol_ratio_threshold", 1.2, 0.8, 2.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Extract arrays ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        vix_low = params.get("vix_low", 13.0)
        vix_high = params.get("vix_high", 20.0)
        vol_ratio_thresh = params.get("vol_ratio_threshold", 1.2)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # --- VIX at 09:30 IST per day (day-level filter) ---
        vix_930 = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_raw = vix_joined["vix_close"].fill_null(15.0).to_numpy()
            # Snapshot VIX at first bar where time_minutes >= 570 per day (09:30)
            day_vix: dict[int, float] = {}
            for i in range(n):
                d = int(day_id[i])
                if d not in day_vix and time_min[i] >= 570:
                    day_vix[d] = float(vix_raw[i])
            for i in range(n):
                d = int(day_id[i])
                if d in day_vix:
                    vix_930[i] = day_vix[d]

        # --- ORB high/low: 09:15-09:30 (time_minutes 555 to <570) per day ---
        # 180 bars at 5s = 15 minutes (fixed-window, no scaling)
        day_orb_high: dict[int, float] = {}
        day_orb_low: dict[int, float] = {}
        for i in range(n):
            if 555 <= time_min[i] < 570:
                d = int(day_id[i])
                if d not in day_orb_high:
                    day_orb_high[d] = float(high[i])
                    day_orb_low[d] = float(low[i])
                else:
                    if high[i] > day_orb_high[d]:
                        day_orb_high[d] = float(high[i])
                    if low[i] < day_orb_low[d]:
                        day_orb_low[d] = float(low[i])

        orb_high = np.full(n, np.nan)
        orb_low = np.full(n, np.nan)
        for i in range(n):
            d = int(day_id[i])
            if time_min[i] >= 570 and d in day_orb_high:
                orb_high[i] = day_orb_high[d]
                orb_low[i] = day_orb_low[d]

        # --- Session VWAP: cumulative (typical_price * volume) / cumulative_volume per day ---
        vwap = np.full(n, np.nan)
        cum_tp_vol: dict[int, float] = {}
        cum_vol: dict[int, float] = {}
        for i in range(n):
            d = int(day_id[i])
            tp = (high[i] + low[i] + close[i]) / 3.0
            vol = volume[i]
            if d not in cum_tp_vol:
                cum_tp_vol[d] = 0.0
                cum_vol[d] = 0.0
            cum_tp_vol[d] += tp * vol
            cum_vol[d] += vol
            if cum_vol[d] > 0.0:
                vwap[i] = cum_tp_vol[d] / cum_vol[d]
            else:
                vwap[i] = close[i]

        # --- Volume ratio: rolling 12-bar SMA (1 minute at 5s bars) ---
        vol_sma = np.full(n, 1.0)
        for i in range(12, n):
            s = float(np.mean(volume[i - 12:i]))
            vol_sma[i] = s if s > 0.0 else 1.0
        vol_ratio = np.where(vol_sma > 0.0, volume / vol_sma, 1.0)

        # --- Signal construction ---
        # Entry window: 09:30-10:30 IST (570-630 min) — ORB continuation drops sharply after 10:30
        in_entry_window = (time_min >= 570) & (time_min < 630)
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        vix_ok = (vix_930 >= vix_low) & (vix_930 <= vix_high)
        orb_valid = ~np.isnan(orb_high) & ~np.isnan(orb_low)
        vwap_valid = ~np.isnan(vwap)
        vol_ok = vol_ratio >= vol_ratio_thresh

        # Raw single-bar breakout signals
        raw_ce = orb_valid & vwap_valid & (close > orb_high) & (close > vwap)
        raw_pe = orb_valid & vwap_valid & (close < orb_low) & (close < vwap)

        # 2-bar persistence (10s): two consecutive closes outside ORB on the same day
        persist_ce = np.zeros(n, dtype=bool)
        persist_pe = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if day_id[i] == day_id[i - 1]:
                persist_ce[i] = raw_ce[i] & raw_ce[i - 1]
                persist_pe[i] = raw_pe[i] & raw_pe[i - 1]

        buy_ce = in_session & in_entry_window & vix_ok & vol_ok & persist_ce
        buy_pe = in_session & in_entry_window & vix_ok & vol_ok & persist_pe

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
