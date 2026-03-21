"""Opening Range Breakout — 5-second NIFTY index options strategy.

Mechanism:
  On NIFTY, the 09:15-09:30 window is the price discovery phase where FII positioning,
  SGX cues, and gap reactions are absorbed. When NIFTY breaches OR_high with session VWAP
  already above the OR midpoint, TWAP/VWAP algorithms release queued directional orders,
  triggering a 30-90 second cascade of 15-40 spot points. The VWAP confirmation is the
  critical filter: VWAP-above breakouts have genuine institutional buy-side bias; VWAP-below
  breakouts are suspect retail spikes that fade.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "opening_range_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — after OR is established
    session_end_minutes = 720     # 12:00 IST — ORB edge dissipates by midday
    max_trades_per_day = 4
    max_lookback = 200            # covers 15-min OR (180 bars at 5s) + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", 1.5, 1.0, 3.0),    # volume surge multiplier vs OR avg
            TunableParam("vix_max", 22.0, 16.0, 28.0),  # max India VIX for valid ORB
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill then extract arrays
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        vol_mult = params.get("vol_mult", 1.5)
        vix_max = params.get("vix_max", 22.0)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── VIX filter ─────────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Session VWAP — cumulative per day from 09:15 ────────────────────────
        # VWAP is a cumulative measure; no lookback scaling needed.
        vwap = np.zeros(n)
        running_pv = 0.0
        running_vol = 0.0
        current_day = -1
        for i in range(n):
            d = day_id[i]
            if d != current_day:
                current_day = d
                running_pv = 0.0
                running_vol = 0.0
            vol_i = volume[i] if volume[i] > 0 else 1.0
            running_pv += close[i] * vol_i
            running_vol += vol_i
            vwap[i] = running_pv / running_vol

        # ── Per-day Opening Range (09:15-09:30, time_minutes 555-569) ──────────
        # Fixed time window — 15 minutes = 180 bars at 5s. Not scaled.
        OR_START = 555   # 09:15 IST in minutes from midnight
        OR_END = 570     # 09:30 IST (exclusive)

        or_high = np.zeros(n)
        or_low = np.zeros(n)
        or_vol_avg = np.zeros(n)

        for d in np.unique(day_id):
            day_mask = day_id == d
            day_indices = np.where(day_mask)[0]

            or_bar_mask = day_mask & (time_min >= OR_START) & (time_min < OR_END)
            or_indices = np.where(or_bar_mask)[0]

            if len(or_indices) == 0:
                continue

            day_or_high = np.max(high[or_indices])
            day_or_low = np.min(low[or_indices])
            day_or_vol_avg = np.mean(volume[or_indices])

            or_high[day_indices] = day_or_high
            or_low[day_indices] = day_or_low
            or_vol_avg[day_indices] = day_or_vol_avg

        # ── OR range validity (0.1% to 2.0% of OR low) ─────────────────────────
        or_range_pct = np.where(or_low > 0, (or_high - or_low) / or_low * 100.0, 0.0)
        or_range_ok = (or_range_pct > 0.1) & (or_range_pct < 2.0)

        # ── Rolling 3-bar (15s) volume average ─────────────────────────────────
        vol_3 = np.zeros(n)
        for i in range(3, n):
            vol_3[i] = np.mean(volume[i - 3:i])

        # ── 3-bar persistence filter (replaces 1-min body check from original) ─
        above_or = close > or_high
        below_or = close < or_low

        persist_above = np.zeros(n, dtype=bool)
        persist_below = np.zeros(n, dtype=bool)
        for i in range(3, n):
            persist_above[i] = above_or[i] and above_or[i - 1] and above_or[i - 2]
            persist_below[i] = below_or[i] and below_or[i - 1] and below_or[i - 2]

        # ── Volume confirmation ─────────────────────────────────────────────────
        vol_confirm = (vol_3 > vol_mult * or_vol_avg) & (or_vol_avg > 0)

        # ── VWAP confirmation (key differentiator from other ORB variants) ──────
        # Original requires close > vwap for long, close < vwap for short.
        vwap_bullish = close > vwap
        vwap_bearish = close < vwap

        # ── Combined filters ───────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < vix_max

        buy_ce = in_session & persist_above & vol_confirm & vwap_bullish & vix_ok & or_range_ok
        buy_pe = in_session & persist_below & vol_confirm & vwap_bearish & vix_ok & or_range_ok

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
