"""Narrow Range 7 Breakout — gemini_14_of_20

Thesis: When today's range is the narrowest of the last 7 days, volatility
compression is extreme and a breakout of the previous day's high/low tends
to produce a sustained move.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "narrow_range_7_breakout"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 660     # 11:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("range_target_mult", default=1.5, low=0.8, high=3.0),
            TunableParam("range_stop_mult", default=0.5, low=0.2, high=1.0),
            TunableParam("be_pct", default=0.5, low=0.2, high=1.0),
            TunableParam("vol_mult", default=1.0, low=0.5, high=2.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        range_target_mult = params.get("range_target_mult", 1.5)
        range_stop_mult = params.get("range_stop_mult", 0.5)
        be_pct = params.get("be_pct", 0.5)
        vol_mult = params.get("vol_mult", 1.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        day_id = df["day_id"].to_numpy()

        atr14 = _compute_atr(high, low, close, 14)

        # Compute daily high, low, range per day
        sorted_days = sorted(set(day_id))
        day_info = {}  # day -> (high, low, range)
        for d in sorted_days:
            mask = day_id == d
            day_info[d] = (np.max(high[mask]), np.min(low[mask]), np.max(high[mask]) - np.min(low[mask]))

        # For each day, check if previous day was NR7
        day_to_idx = {d: i for i, d in enumerate(sorted_days)}
        is_nr7_day = {}
        prev_day_high_map = {}
        prev_day_low_map = {}
        prev_day_range_map = {}

        for idx, d in enumerate(sorted_days):
            if idx < 7:
                is_nr7_day[d] = False
                if idx > 0:
                    prev_d = sorted_days[idx - 1]
                    prev_day_high_map[d] = day_info[prev_d][0]
                    prev_day_low_map[d] = day_info[prev_d][1]
                    prev_day_range_map[d] = day_info[prev_d][2]
                else:
                    prev_day_high_map[d] = high[0]
                    prev_day_low_map[d] = low[0]
                    prev_day_range_map[d] = high[0] - low[0]
                continue

            prev_d = sorted_days[idx - 1]
            prev_day_high_map[d] = day_info[prev_d][0]
            prev_day_low_map[d] = day_info[prev_d][1]
            prev_day_range_map[d] = day_info[prev_d][2]

            # Check if prev day range is narrowest of last 7 days
            prev_range = day_info[prev_d][2]
            is_narrowest = True
            for k in range(2, 8):  # look back 7 days including prev
                if idx - k < 0:
                    break
                comp_d = sorted_days[idx - k]
                if day_info[comp_d][2] <= prev_range:
                    is_narrowest = False
                    break
            is_nr7_day[d] = is_narrowest

        # Per-bar arrays
        is_nr7 = np.zeros(n, dtype=np.bool_)
        prev_day_high = np.full(n, np.inf, dtype=np.float64)
        prev_day_low = np.full(n, -np.inf, dtype=np.float64)
        prev_day_range = np.ones(n, dtype=np.float64)

        for i in range(n):
            d = day_id[i]
            is_nr7[i] = is_nr7_day.get(d, False)
            prev_day_high[i] = prev_day_high_map.get(d, high[i])
            prev_day_low[i] = prev_day_low_map.get(d, low[i])
            prev_day_range[i] = prev_day_range_map.get(d, 1.0)

        # Volume SMA(20)
        vol_sma20 = np.full(n, 1.0, dtype=np.float64)
        for i in range(19, n):
            vol_sma20[i] = np.mean(volume[i - 19:i + 1])
        vol_sma20 = np.where(vol_sma20 > 1e-10, vol_sma20, 1.0)
        rel_vol = volume / vol_sma20

        # Entries
        long_entry = is_nr7 & (close > prev_day_high) & (rel_vol > vol_mult)
        short_entry = is_nr7 & (close < prev_day_low) & (rel_vol > vol_mult)

        # Target/stop as fraction of prev day range, converted to pct of close
        safe_close = np.where(close > 1e-10, close, 1e-10)
        target_pct_arr = range_target_mult * prev_day_range / safe_close
        stop_pct_arr = range_stop_mult * prev_day_range / safe_close

        # Use median for scalar parameters
        target_pct_scalar = float(np.median(target_pct_arr[target_pct_arr > 0])) if np.any(target_pct_arr > 0) else 0.01
        stop_pct_scalar = float(np.median(stop_pct_arr[stop_pct_arr > 0])) if np.any(stop_pct_arr > 0) else 0.005

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct_scalar,
            stop_loss_pct=stop_pct_scalar,
            breakeven_pct=be_pct / 100.0,
            time_stop_bars=300,
        )
