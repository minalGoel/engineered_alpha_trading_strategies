"""
first_hour_range_breakout_v1 — 5-second NIFTY index options strategy.

Mechanism:
    NIFTY's first trading hour (09:15-10:15) forms a well-anchored high-low range as
    overnight information, FII positioning, and auction imbalances are absorbed. Institutional
    bracket orders and TWAP algorithms accumulate stop-losses just outside this range. When
    NIFTY closes above the first-hour high (or below the low) with elevated volume post-10:15,
    it triggers a cascade of stops and directional algorithms, producing a 20-35 spot point
    momentum burst in the first 30-90 seconds. We enter on 2-bar persistence above/below the
    range level (filtering spikes) and ride the initial cascade.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "first_hour_range_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 615   # 10:15 IST — entries only after FHR complete
    session_end_minutes = 900     # 15:00 IST
    max_trades_per_day = 5
    max_lookback = 720            # 720 bars = 60 min (first-hour window)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("fhr_width_min_pct", 0.30, 0.10, 0.70),
            TunableParam("fhr_width_max_pct", 1.50, 0.80, 2.50),
            TunableParam("vol_ratio_threshold", 1.30, 1.00, 2.00),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ──
        fhr_width_min_pct = params.get("fhr_width_min_pct", 0.30)
        fhr_width_max_pct = params.get("fhr_width_max_pct", 1.50)
        vol_ratio_threshold = params.get("vol_ratio_threshold", 1.30)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── Session VWAP (cumulative, reset per day) ──
        typical = (high + low + close) / 3.0
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol_s = 0.0
        prev_day = day_id[0] - 1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_tp_vol = 0.0
                cum_vol_s = 0.0
                prev_day = day_id[i]
            v = volume[i]
            cum_tp_vol += typical[i] * v
            cum_vol_s += v
            vwap[i] = cum_tp_vol / cum_vol_s if cum_vol_s > 0.0 else close[i]

        # ── First Hour Range (09:15–10:14:55 = time_minutes 555–614) ──
        # First pass: accumulate max high / min low per day during first hour
        day_fhr_high: dict[int, float] = {}
        day_fhr_low: dict[int, float] = {}
        for i in range(n):
            d = int(day_id[i])
            t = int(time_min[i])
            if 555 <= t <= 614:
                if d not in day_fhr_high:
                    day_fhr_high[d] = high[i]
                    day_fhr_low[d] = low[i]
                else:
                    if high[i] > day_fhr_high[d]:
                        day_fhr_high[d] = high[i]
                    if low[i] < day_fhr_low[d]:
                        day_fhr_low[d] = low[i]

        # Second pass: assign FHR levels to bars after 10:15 (time_minutes >= 615)
        fhr_high_arr = np.zeros(n)
        fhr_low_arr = np.zeros(n)
        fhr_valid = np.zeros(n, dtype=bool)
        for i in range(n):
            d = int(day_id[i])
            if d in day_fhr_high and int(time_min[i]) >= 615:
                fhr_high_arr[i] = day_fhr_high[d]
                fhr_low_arr[i] = day_fhr_low[d]
                fhr_valid[i] = True

        # ── FHR width filter ──
        fhr_width = fhr_high_arr - fhr_low_arr
        fhr_width_pct = np.where(
            fhr_low_arr > 0.0,
            fhr_width / fhr_low_arr * 100.0,
            0.0
        )
        width_ok = (fhr_width_pct >= fhr_width_min_pct) & (fhr_width_pct <= fhr_width_max_pct)

        # ── Volume ratio vs 5-min rolling mean (60 bars at 5s) ──
        vol_mean_60 = np.zeros(n)
        for i in range(60, n):
            vol_mean_60[i] = np.mean(volume[i - 60:i])
        vol_ratio = np.where(vol_mean_60 > 0.0, volume / vol_mean_60, 1.0)

        # ── 2-bar persistence: both current and previous bar above/below level ──
        # (filters single-5s-bar spikes; same-day FHR level used)
        above_fhr = np.zeros(n, dtype=bool)
        below_fhr = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if fhr_valid[i] and fhr_valid[i - 1] and day_id[i] == day_id[i - 1]:
                lvl_h = fhr_high_arr[i]
                lvl_l = fhr_low_arr[i]
                above_fhr[i] = (close[i] > lvl_h) and (close[i - 1] > lvl_h)
                below_fhr[i] = (close[i] < lvl_l) and (close[i - 1] < lvl_l)

        # ── VIX filter (12–25) ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()
        vix_ok = (vix_close >= 12.0) & (vix_close <= 25.0)

        # ── Session time filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry signals ──
        buy_ce = (
            in_session
            & fhr_valid
            & width_ok
            & above_fhr
            & (close > vwap)
            & (vol_ratio >= vol_ratio_threshold)
            & vix_ok
        )

        buy_pe = (
            in_session
            & fhr_valid
            & width_ok
            & below_fhr
            & (close < vwap)
            & (vol_ratio >= vol_ratio_threshold)
            & vix_ok
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds = 18 × 5s bars
            max_trades_per_day=self.max_trades_per_day,
        )
