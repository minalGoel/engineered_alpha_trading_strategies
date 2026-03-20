"""Opening Range Breakout — VIX-Conditioned 30-Minute Range (orb_momentum_v29)

Mechanism: NIFTY's 30-minute opening range (09:15-09:45) is used as the institutional
calibration window. When VIX > 15, option market-makers reduce gamma selling exposure,
removing the mean-reversion counterforce that normally dampens ORB breakouts. Institutional
directional TWAP algos running partially filled orders must continue through the supply
ceiling with less resistance, producing 25-45 spot point continuation moves. The VIX
condition is the critical differentiator from orb_momentum_v22 (20-min ORB, no VIX filter).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "orb_momentum_v29"
    underlying = "NIFTY"
    session_start_minutes = 585   # 09:45 IST — after 30-min ORB fully forms
    session_end_minutes = 870     # 14:30 IST — cut off before late-session thin liquidity
    max_trades_per_day = 4
    max_lookback = 360            # 30 min warmup (360 bars × 5s = 1800s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_ratio_threshold", 1.2, 1.0, 2.0),
            TunableParam("vix_min", 15.0, 12.0, 20.0),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 9.0, 5.0, 15.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before numpy) ──────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        vol_ratio_threshold = params.get("vol_ratio_threshold", 1.2)
        vix_min = params.get("vix_min", 15.0)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 9.0)

        # ── VIX aligned to spot bars ──────────────────────────────────────────
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

        # ── 30-minute Opening Range High/Low (09:15-09:45) ───────────────────
        # Fixed time window: 09:15 = 555 min, 09:45 = 585 min.
        # Accumulate during the window, lock at 09:45, hold constant all day.
        orb_high = np.zeros(n)
        orb_low = np.zeros(n)

        current_day = -1
        day_orb_high = -np.inf
        day_orb_low = np.inf
        locked_high = 0.0
        locked_low = 0.0
        day_locked = False

        for i in range(n):
            d = day_id[i]
            t = time_min[i]

            if d != current_day:
                current_day = d
                day_orb_high = -np.inf
                day_orb_low = np.inf
                locked_high = 0.0
                locked_low = 0.0
                day_locked = False

            if not day_locked:
                if 555 <= t < 585:
                    # Within the 30-min opening range window — accumulate
                    if high[i] > day_orb_high:
                        day_orb_high = high[i]
                    if low[i] < day_orb_low:
                        day_orb_low = low[i]
                elif t >= 585:
                    # Lock the range at 09:45
                    locked_high = day_orb_high if day_orb_high > -np.inf else close[i]
                    locked_low = day_orb_low if day_orb_low < np.inf else close[i]
                    day_locked = True

            orb_high[i] = locked_high if day_locked else (
                day_orb_high if day_orb_high > -np.inf else close[i]
            )
            orb_low[i] = locked_low if day_locked else (
                day_orb_low if day_orb_low < np.inf else close[i]
            )

        # ── Session VWAP (cumulative, reset per day) ─────────────────────────
        vwap = np.zeros(n)
        current_day = -1
        running_tp_vol = 0.0
        running_vol = 0.0

        for i in range(n):
            d = day_id[i]
            if d != current_day:
                current_day = d
                running_tp_vol = 0.0
                running_vol = 0.0
            tp = (high[i] + low[i] + close[i]) / 3.0
            running_tp_vol += tp * volume[i]
            running_vol += volume[i]
            vwap[i] = (running_tp_vol / running_vol) if running_vol > 0 else close[i]

        # ── Rolling 15-minute volume mean (180 bars) ─────────────────────────
        vol_mean = np.zeros(n)
        for i in range(n):
            start = max(0, i - 179)
            vol_mean[i] = np.mean(volume[start : i + 1])

        vol_ratio = np.zeros(n)
        valid_vol = vol_mean > 0
        vol_ratio[valid_vol] = volume[valid_vol] / vol_mean[valid_vol]

        # ── 3-bar persistence filter ──────────────────────────────────────────
        # close must exceed ORB level for 3 consecutive 5s bars (15 seconds)
        above_orb = close > orb_high
        below_orb = close < orb_low

        persist_above = np.zeros(n, dtype=bool)
        persist_below = np.zeros(n, dtype=bool)
        for i in range(2, n):
            persist_above[i] = above_orb[i] and above_orb[i - 1] and above_orb[i - 2]
            persist_below[i] = below_orb[i] and below_orb[i - 1] and below_orb[i - 2]

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── VIX regime filter ─────────────────────────────────────────────────
        vix_elevated = vix_close > vix_min

        # ── ORB validity (range must have been computed) ──────────────────────
        orb_valid = (orb_high > 0) & (orb_low > 0) & (orb_high > orb_low)

        # ── Entry signals ─────────────────────────────────────────────────────
        buy_ce = (
            in_session
            & vix_elevated
            & persist_above
            & (close > vwap)
            & (vol_ratio > vol_ratio_threshold)
            & orb_valid
        )

        buy_pe = (
            in_session
            & vix_elevated
            & persist_below
            & (close < vwap)
            & (vol_ratio > vol_ratio_threshold)
            & orb_valid
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
