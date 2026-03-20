"""Opening Range Breakout — GPT_3_of_10

Thesis: The first 15 minutes of trading sets directional bias via overnight news
and institutional orders. Breaking the opening range high/low signals continuation.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "opening_range_breakout_v1"
    is_long_only = False
    session_start = 555   # 09:15 — need to capture opening range
    session_end = 915     # 15:15
    max_trades_per_day = 1

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", default=25.0, low=15.0, high=35.0),
            TunableParam("target_pct", default=0.01, low=0.005, high=0.02),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.015),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_max = params.get("vix_max", 25.0)
        target_pct = params.get("target_pct", 0.01)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Compute opening range (first 15 min: 09:15-09:29 = 555-569) ──
        or_high = np.full(n, np.nan, dtype=np.float64)
        or_low = np.full(n, np.nan, dtype=np.float64)

        unique_days = np.unique(day_ids)
        for d in unique_days:
            day_mask = day_ids == d
            day_indices = np.where(day_mask)[0]
            # Opening range bars: time_minutes 555 to 569 (first 15 bars)
            or_mask = (time_mins[day_indices] >= 555) & (time_mins[day_indices] <= 569)
            or_indices = day_indices[or_mask]
            if len(or_indices) == 0:
                continue
            or_h = np.max(high[or_indices])
            or_l = np.min(low[or_indices])
            # Apply OR levels to all bars of this day AFTER the opening range
            or_high[day_indices] = or_h
            or_low[day_indices] = or_l

        or_high = np.nan_to_num(or_high, nan=1e10)
        or_low = np.nan_to_num(or_low, nan=-1e10)

        # AUDIT FIX: Added upper bound time_mins <= session_end to prevent entries after 15:15
        # ── Entry: only after opening range is complete (from 09:30 = 570) ──
        after_or = (time_mins >= 570) & (time_mins <= self.session_end)
        vix_ok = vix < vix_max

        long_entry = (close > or_high) & after_or & vix_ok
        short_entry = (close < or_low) & after_or & vix_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=20,
        )
