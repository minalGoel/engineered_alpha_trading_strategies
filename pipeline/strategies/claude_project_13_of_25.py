"""First Hour Range Breakout v1 — claude_project_13_of_25

Thesis: The first hour (09:15-10:15) establishes a wider, more stable
range than 15-min ORB. Breakout after the first hour with volume
signals a sustained directional move.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "first_hour_range_breakout_v1"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 870     # 14:30
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=1.3, low=1.0, high=2.5),
            TunableParam("target_width_mult", default=1.0, low=0.5, high=2.0),
            TunableParam("breakeven_width_mult", default=0.5, low=0.2, high=1.0),
            TunableParam("vix_min", default=12.0, low=8.0, high=16.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=35.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 1.3)
        target_w = params.get("target_width_mult", 1.0)
        be_w = params.get("breakeven_width_mult", 0.5)
        vix_min = params.get("vix_min", 12.0)
        vix_max = params.get("vix_max", 25.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # ── Volume average ──
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        # ── First hour range per day: bars from 555-614 (09:15-10:14) ──
        fh_high = np.zeros(n, dtype=np.float64)
        fh_low = np.zeros(n, dtype=np.float64)
        fh_mid = np.zeros(n, dtype=np.float64)
        fh_width = np.zeros(n, dtype=np.float64)
        fh_valid = np.zeros(n, dtype=np.bool_)

        cur_day = -1
        day_fh_h = -1e18
        day_fh_l = 1e18
        day_fh_done = False
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                day_fh_h = -1e18
                day_fh_l = 1e18
                day_fh_done = False
            if not day_fh_done and 555 <= time_mins[i] <= 614:
                day_fh_h = max(day_fh_h, high[i])
                day_fh_l = min(day_fh_l, low[i])
            elif time_mins[i] > 614:
                day_fh_done = True
            if day_fh_done and day_fh_h > 0 and day_fh_l < 1e17:
                fh_high[i] = day_fh_h
                fh_low[i] = day_fh_l
                mid = (day_fh_h + day_fh_l) / 2.0
                fh_mid[i] = mid
                if mid > 0:
                    fh_width[i] = (day_fh_h - day_fh_l) / mid
                fh_valid[i] = True

        # ── Filters ──
        vol_ok = volume > vol_mult * avg_vol
        vix_ok = (vix >= vix_min) & (vix <= vix_max)

        # ── Entry ──
        long_entry = fh_valid & (close > fh_high) & vol_ok & vix_ok
        short_entry = fh_valid & (close < fh_low) & vol_ok & vix_ok

        # ── Exit parameters based on first hour width ──
        valid_widths = fh_width[fh_valid]
        median_width = float(np.median(valid_widths)) if len(valid_widths) > 0 else 0.005
        target_pct = target_w * median_width
        # Stop at midpoint ~ 0.5 * width
        stop_pct = 0.5 * median_width
        breakeven_pct = be_w * median_width

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            breakeven_pct=breakeven_pct,
            time_stop_bars=300,
        )
