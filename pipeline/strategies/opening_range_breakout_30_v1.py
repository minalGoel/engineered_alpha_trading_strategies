"""Opening Range Breakout 30-minute — NIFTY/BANKNIFTY 5-second options.

Mechanism:
    NIFTY and BANKNIFTY spend the first 30 minutes (09:15-09:45 IST) absorbing
    overnight positioning, gap adjustments, and initial institutional TWAP/VWAP flow.
    When 3 consecutive 5-second bars close above the OR30 high with above-average volume
    and price above session VWAP, the supply ceiling has been systematically consumed by
    institutional buy-side flow. We enter within 15 seconds of confirmed breakout, capturing
    the initial 15-25 spot point directional thrust.

Original: opening_range_breakout_30_v1 (Strategy_175.json)
  - 1-min bars, 2-bar confirmation, hold 30-120 min on NIFTY100 stocks
  - Adapted to: 5s bars, 3-bar persistence (15s), hold 30-90s on NIFTY/BANKNIFTY index
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "opening_range_breakout_30_v1"
    underlying = "BOTH"
    session_start_minutes = 585   # 09:45 IST — after 30-min OR is fully established
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 4
    max_lookback = 360            # 30 min warmup for OR formation (360 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", 2.0, 1.0, 4.0),      # volume confirmation multiplier
            TunableParam("stop_pts", 5.0, 3.0, 9.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
            TunableParam("vix_max", 28.0, 18.0, 35.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # --- Extract spot arrays (forward-fill before to_numpy) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        vol_mult = params.get("vol_mult", 2.0)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)
        vix_max = params.get("vix_max", 28.0)

        # --- VIX: align to spot bars ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- Per-day 30-minute Opening Range ---
        # OR window: 09:15-09:44:55 IST (time_min 555 to 584 inclusive)
        OR_HIGH = np.full(n, np.nan)
        OR_LOW = np.full(n, np.nan)
        OR_AVG_VOL = np.full(n, np.nan)

        unique_days = np.unique(day_id)
        for d in unique_days:
            day_mask = day_id == d
            in_or = day_mask & (time_min >= 555) & (time_min < 585)
            if np.sum(in_or) == 0:
                continue
            or_high_val = np.max(high[in_or])
            or_low_val = np.min(low[in_or])
            or_avg_vol = np.mean(volume[in_or])
            OR_HIGH[day_mask] = or_high_val
            OR_LOW[day_mask] = or_low_val
            OR_AVG_VOL[day_mask] = or_avg_vol

        # OR range as percentage of OR_LOW
        with np.errstate(invalid="ignore", divide="ignore"):
            OR_RANGE_PCT = np.where(OR_LOW > 0, (OR_HIGH - OR_LOW) / OR_LOW * 100.0, np.nan)

        # --- Session VWAP (cumulative per day from session open) ---
        vwap = np.full(n, np.nan)
        for d in unique_days:
            idx = np.where(day_id == d)[0]
            if len(idx) == 0:
                continue
            c = close[idx]
            v = volume[idx]
            cum_cv = np.cumsum(c * v)
            cum_v = np.cumsum(v)
            with np.errstate(invalid="ignore", divide="ignore"):
                vwap[idx] = np.where(cum_v > 0, cum_cv / cum_v, c)

        # Session filter (time-based, after OR is established)
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # VIX regime filter
        vix_ok = (vix_close >= 12.0) & (vix_close <= vix_max)

        # OR range validity filter: 0.2% to 3.0%
        range_ok = (OR_RANGE_PCT >= 0.2) & (OR_RANGE_PCT <= 3.0)

        # --- Generate breakout signals (3-bar persistence, first occurrence per day) ---
        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        for d in unique_days:
            day_idx = np.where(day_id == d)[0]
            if len(day_idx) < 3:
                continue
            ce_triggered = False
            pe_triggered = False

            for k in range(2, len(day_idx)):
                i = day_idx[k]
                i1 = day_idx[k - 1]
                i2 = day_idx[k - 2]

                if not in_session[i]:
                    continue
                if np.isnan(OR_HIGH[i]) or np.isnan(OR_LOW[i]) or np.isnan(vwap[i]):
                    continue
                if not (range_ok[i] and vix_ok[i]):
                    continue

                or_h = OR_HIGH[i]
                or_l = OR_LOW[i]
                vol_threshold = vol_mult * OR_AVG_VOL[i]

                # Bullish: 3 consecutive bars above OR30_high + VWAP + volume
                if (
                    not ce_triggered
                    and close[i] > or_h
                    and close[i1] > or_h
                    and close[i2] > or_h
                    and close[i] > vwap[i]
                    and volume[i] > vol_threshold
                ):
                    buy_ce[i] = True
                    ce_triggered = True

                # Bearish: 3 consecutive bars below OR30_low + VWAP + volume
                if (
                    not pe_triggered
                    and close[i] < or_l
                    and close[i1] < or_l
                    and close[i2] < or_l
                    and close[i] < vwap[i]
                    and volume[i] > vol_threshold
                ):
                    buy_pe[i] = True
                    pe_triggered = True

        # No simultaneous CE and PE signals
        both = buy_ce & buy_pe
        buy_ce[both] = False
        buy_pe[both] = False

        any_signal = buy_ce | buy_pe
        stop_arr = np.where(any_signal, stop_pts, 0.0)
        target_arr = np.where(any_signal, target_pts, 0.0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_arr.astype(np.float64),
            target_points=target_arr.astype(np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
