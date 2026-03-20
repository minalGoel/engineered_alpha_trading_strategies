"""Opening Range Breakout 15-min v1 — claude_project_2_of_25

Thesis: The first 15 minutes establish a range; a breakout with volume
signals directional commitment for the session.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "orb_15min_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("or_width_min", default=0.003, low=0.001, high=0.006),
            TunableParam("or_width_max", default=0.015, low=0.008, high=0.025),
            TunableParam("vol_mult", default=1.5, low=1.0, high=3.0),
            TunableParam("target_mult", default=1.5, low=1.0, high=3.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=35.0),
            TunableParam("breakeven_width_mult", default=1.0, low=0.5, high=2.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        or_width_min = params.get("or_width_min", 0.003)
        or_width_max = params.get("or_width_max", 0.015)
        vol_mult = params.get("vol_mult", 1.5)
        target_mult = params.get("target_mult", 1.5)
        vix_max = params.get("vix_max", 25.0)
        be_width_mult = params.get("breakeven_width_mult", 1.0)

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

        # ── Compute 15-min opening range per day ──
        or_high = np.zeros(n, dtype=np.float64)
        or_low = np.zeros(n, dtype=np.float64)
        or_valid = np.zeros(n, dtype=np.bool_)

        # Rolling volume average
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        # Build OR per day: bars from 570..584 (09:30-09:44)
        cur_day = -1
        day_or_h = 0.0
        day_or_l = 1e18
        day_or_done = False
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                day_or_h = -1e18
                day_or_l = 1e18
                day_or_done = False
            if not day_or_done and 570 <= time_mins[i] <= 584:
                day_or_h = max(day_or_h, high[i])
                day_or_l = min(day_or_l, low[i])
            elif time_mins[i] > 584:
                day_or_done = True
            if day_or_done and day_or_h > 0 and day_or_l < 1e17:
                or_high[i] = day_or_h
                or_low[i] = day_or_l
                or_valid[i] = True

        or_mid = (or_high + or_low) / 2.0
        or_mid = np.where(or_mid > 0, or_mid, 1.0)
        or_width = (or_high - or_low) / or_mid
        or_width = np.nan_to_num(or_width, nan=0.0)

        # ── Filters ──
        width_ok = (or_width > or_width_min) & (or_width < or_width_max)
        vol_ok = volume > vol_mult * avg_vol
        vix_ok = vix < vix_max
        # Entry window: 09:30-12:00
        time_ok = (time_mins >= 585) & (time_mins <= 720)

        # ── Entry ──
        long_entry = or_valid & (close > or_high) & width_ok & vol_ok & vix_ok & time_ok
        short_entry = or_valid & (close < or_low) & width_ok & vol_ok & vix_ok & time_ok

        # ── Target: 1.5x OR width in pct; stop at opposite OR boundary ──
        # Use median or_width as representative target_pct
        valid_widths = or_width[or_valid]
        median_width = float(np.median(valid_widths)) if len(valid_widths) > 0 else 0.005
        target_pct = target_mult * median_width
        stop_pct = median_width  # approximately opposite OR side
        breakeven_pct = be_width_mult * median_width

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
            time_stop_bars=240,
        )
