"""Narrow Range Breakout v1 — cursor_opus46max_038

Thesis: NR4/NR7 days (narrowest range in 4/7 days) signal volatility
compression preceding range expansion. First-hour range breakout on NR
days has higher follow-through. Target is 1.5x ATR(14d).
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "cursor_opus46max_038"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 870     # 14:30
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("target_atr_mult", default=1.5, low=1.0, high=2.5),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 1.5)
        vix_max = params.get("vix_max", 22.0)
        tgt_atr_mult = params.get("target_atr_mult", 1.5)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Compute daily ranges and NR flag ──
        # Track per-day high/low
        day_ranges = {}  # day_id -> range_pct
        day_high_map = {}
        day_low_map = {}
        prev_day = -1
        day_start = 0

        for i in range(n):
            d = day_id[i]
            if d not in day_high_map:
                day_high_map[d] = high[i]
                day_low_map[d] = low[i]
            else:
                day_high_map[d] = max(day_high_map[d], high[i])
                day_low_map[d] = min(day_low_map[d], low[i])

        for d in day_high_map:
            if day_low_map[d] > 0:
                day_ranges[d] = (day_high_map[d] - day_low_map[d]) / day_low_map[d] * 100.0
            else:
                day_ranges[d] = 0.0

        # Get sorted unique days
        unique_days = sorted(day_high_map.keys())
        day_to_idx = {d: i for i, d in enumerate(unique_days)}

        is_nr4 = np.zeros(n, dtype=np.bool_)
        is_nr7 = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            d = day_id[i]
            didx = day_to_idx.get(d, 0)
            if didx >= 4:
                cur_range = day_ranges.get(d, 0.0)
                prev_ranges = [day_ranges.get(unique_days[didx-j], 1e10) for j in range(1, 4)]
                if cur_range < min(prev_ranges) and cur_range > 0:
                    is_nr4[i] = True
            if didx >= 7:
                cur_range = day_ranges.get(d, 0.0)
                prev_ranges = [day_ranges.get(unique_days[didx-j], 1e10) for j in range(1, 7)]
                if cur_range < min(prev_ranges) and cur_range > 0:
                    is_nr7[i] = True

        is_nr = is_nr4 | is_nr7

        # ── First-hour high/low (60 bars from day start) ──
        fh_high = np.zeros(n, dtype=np.float64)
        fh_low = np.zeros(n, dtype=np.float64)
        fh_computed = np.zeros(n, dtype=np.bool_)

        prev_day = -1
        day_start = 0
        cur_fh_high = 0.0
        cur_fh_low = 1e18

        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                day_start = i
                cur_fh_high = high[i]
                cur_fh_low = low[i]
            else:
                bars = i - day_start
                if bars < 60:
                    cur_fh_high = max(cur_fh_high, high[i])
                    cur_fh_low = min(cur_fh_low, low[i])

            fh_high[i] = cur_fh_high
            fh_low[i] = cur_fh_low
            if i - day_start >= 60:
                fh_computed[i] = True

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Breakout bar quality ──
        bar_range = high - low
        bar_range_safe = np.where(bar_range > 0, bar_range, 1e-10)
        close_pos = (close - low) / bar_range_safe

        # ── 2-bar confirmation ──
        above_fh_count = np.zeros(n, dtype=np.int32)
        below_fh_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if fh_computed[i] and close[i] > fh_high[i]:
                above_fh_count[i] = above_fh_count[i-1] + 1
            else:
                above_fh_count[i] = 0
            if fh_computed[i] and close[i] < fh_low[i]:
                below_fh_count[i] = below_fh_count[i-1] + 1
            else:
                below_fh_count[i] = 0

        # ── Filters ──
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 615) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            is_nr & fh_computed & (above_fh_count == 2) &
            vol_ok & (close > vwap) & (close_pos > 0.70) &
            vix_ok & time_ok
        )
        short_entry = (
            is_nr & fh_computed & (below_fh_count == 2) &
            vol_ok & (close < vwap) & (close_pos < 0.30) &
            vix_ok & time_ok
        )

        # ── Signal exit: 3 bars back inside first-hour range ──
        inside_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if fh_computed[i] and close[i] >= fh_low[i] and close[i] <= fh_high[i]:
                inside_count[i] = inside_count[i-1] + 1
            else:
                inside_count[i] = 0

        signal_exit_long = inside_count >= 3
        signal_exit_short = inside_count >= 3

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_atr_mult=tgt_atr_mult,
            stop_loss_pct=0.005,
            time_stop_bars=90,
        )
