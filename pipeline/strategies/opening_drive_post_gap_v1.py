"""opening_drive_post_gap_v1 — Gap-Aligned Opening Drive Momentum on NIFTY

Mechanism:
    On NIFTY, overnight gaps >0.3% reflect FII/macro repositioning. The first 5 minutes
    (09:15–09:20) resolve pre-open auction imbalances. When the first-5-min candle closes
    in the gap direction with a strong body (>55% body/range) and above-average volume,
    institutional VWAP/TWAP algorithms have committed to directional orders that are only
    20–30% filled at 09:20. Residual unfilled demand continues pushing NIFTY 10–20 spot
    points in the same direction for 30–120 seconds. We buy ATM CE (gap up + drive up)
    or ATM PE (gap down + drive down) at exactly 09:20 IST.

Signal fires once per day at 09:20. Max one trade per day.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "opening_drive_post_gap_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — signal fires at first 5-min close
    session_end_minutes = 562     # 09:22 IST — signal fires once at 09:20; no entries after 09:22
    max_lookback = 60             # 5-minute opening drive window (60 × 5s bars)
    max_trades_per_day = 1        # one entry per day at the open

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold",    0.003, 0.001, 0.008),  # min gap % (0.3% default)
            TunableParam("body_ratio_min",   0.55,  0.40,  0.75),   # candle body/range min
            TunableParam("vol_surge_ratio",  1.5,   1.0,   3.0),    # vs 5-day avg first-5-min vol
            TunableParam("stop_pts",         4.0,   2.0,   8.0),    # option premium points
            TunableParam("target_pts",       8.0,   4.0,   12.0),   # option premium points
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close  = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_  = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high   = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low    = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        gap_threshold  = params.get("gap_threshold",   0.003)
        body_ratio_min = params.get("body_ratio_min",  0.55)
        vol_surge_min  = params.get("vol_surge_ratio", 1.5)
        stop_pts       = params.get("stop_pts",        4.0)
        target_pts     = params.get("target_pts",      8.0)

        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        # ── Build ordered list of unique trading days ──────────────────────────
        unique_days: list[int] = []
        seen: set[int] = set()
        for d in day_id:
            if d not in seen:
                seen.add(d)
                unique_days.append(int(d))

        # ── Per-day bar ranges ──────────────────────────────────────────────────
        day_start: dict[int, int] = {}  # day_id -> first bar index
        day_end:   dict[int, int] = {}  # day_id -> last bar index
        for i, d in enumerate(day_id):
            d = int(d)
            if d not in day_start:
                day_start[d] = i
            day_end[d] = i

        # ── Collect per-day stats for first 5-min window ────────────────────────
        day_stats: dict[int, dict] = {}
        first5_vol_by_day: dict[int, float] = {}

        for d in unique_days:
            s = day_start[d]
            e = day_end[d]
            local_time = time_min[s: e + 1]

            # First 5-min bars: 09:15:00 (time_min=555) to 09:19:55 (time_min=559)
            mask5 = (local_time >= 555) & (local_time < 560)
            idx5 = np.where(mask5)[0] + s

            if len(idx5) == 0:
                continue  # missing opening data, skip day

            f5_open  = open_[idx5[0]]
            f5_close = close[idx5[-1]]
            f5_high  = float(np.max(high[idx5]))
            f5_low   = float(np.min(low[idx5]))
            f5_vol   = float(np.sum(volume[idx5]))

            # Day open = first bar's open (used for gap computation)
            day_open_price = open_[s]
            # Last bar close (used as prev_close for next day's gap)
            day_last_close = close[e]

            day_stats[d] = {
                "day_open":      day_open_price,
                "f5_open":       f5_open,
                "f5_close":      f5_close,
                "f5_high":       f5_high,
                "f5_low":        f5_low,
                "f5_vol":        f5_vol,
                "last_close":    day_last_close,
            }
            first5_vol_by_day[d] = f5_vol

        # ── Generate signals: one per day at 09:20 ─────────────────────────────
        for i_day, d in enumerate(unique_days):
            if d not in day_stats:
                continue
            stats = day_stats[d]

            # Need previous day's close for gap computation
            if i_day == 0:
                continue
            prev_d = unique_days[i_day - 1]
            if prev_d not in day_stats:
                continue
            prev_close = day_stats[prev_d]["last_close"]
            if prev_close <= 0:
                continue

            # ── Gap check ─────────────────────────────────────────────────────
            gap = (stats["day_open"] - prev_close) / prev_close
            if abs(gap) < gap_threshold:
                continue
            gap_dir = 1 if gap > 0 else -1

            # ── First 5-min drive direction ────────────────────────────────────
            f5_open  = stats["f5_open"]
            f5_close = stats["f5_close"]
            if f5_close > f5_open:
                drive_dir = 1
            elif f5_close < f5_open:
                drive_dir = -1
            else:
                continue  # doji, no drive

            if drive_dir != gap_dir:
                continue  # drive counter to gap — thesis does not hold

            # ── Candle body strength ───────────────────────────────────────────
            body   = abs(f5_close - f5_open)
            c_range = stats["f5_high"] - stats["f5_low"] + 0.01
            if body / c_range < body_ratio_min:
                continue  # weak, unresolved price discovery

            # ── Volume surge vs rolling 5-day average ─────────────────────────
            prev_vols = [
                first5_vol_by_day[unique_days[j]]
                for j in range(max(0, i_day - 5), i_day)
                if unique_days[j] in first5_vol_by_day
            ]
            if len(prev_vols) > 0:
                avg_vol = float(np.mean(prev_vols))
                if avg_vol > 0 and stats["f5_vol"] / avg_vol < vol_surge_min:
                    continue  # volume not elevated enough

            # ── Find entry bar: first bar at time_minutes == 560 (09:20:00) ───
            s = day_start[d]
            e = day_end[d]
            local_time = time_min[s: e + 1]

            entry_mask = local_time == 560
            entry_local = np.where(entry_mask)[0]
            if len(entry_local) == 0:
                # Fallback: first bar in [560, 561) window
                entry_mask = (local_time >= 560) & (local_time < 561)
                entry_local = np.where(entry_mask)[0]
            if len(entry_local) == 0:
                continue

            entry_bar = int(entry_local[0]) + s

            if gap_dir == 1:
                buy_ce[entry_bar] = True
            else:
                buy_pe[entry_bar] = True

        # ── Session filter (signals must be in active session) ─────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        buy_ce = buy_ce & in_session
        buy_pe = buy_pe & in_session

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,       # 120 seconds max hold (24 × 5s bars)
            max_trades_per_day=1,    # one shot per day at the open
        )
