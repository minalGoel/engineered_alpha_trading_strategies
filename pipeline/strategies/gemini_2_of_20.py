# AUDIT FIX: Added session window filter to long_entry and short_entry
"""Opening Range Breakout 15 — gemini_2_of_20

Thesis: The first 15 bars define the opening range. Breakout beyond that range
with volume confirmation triggers a directional move.
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
    name = "opening_range_breakout_15"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 660     # 11:00
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=2.0, low=1.0, high=4.0),
            TunableParam("breakeven_pct", default=0.003, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 2.0)
        be_pct = params.get("breakeven_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        day_id = df["day_id"].to_numpy()

        # Compute 15-bar opening range high/low per day
        or_high = np.zeros(n, dtype=np.float64)
        or_low = np.zeros(n, dtype=np.float64)
        or_ready = np.zeros(n, dtype=np.bool_)

        day_bar_count = np.zeros(n, dtype=np.int32)
        prev_day = -1
        count = 0
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                count = 0
            count += 1
            day_bar_count[i] = count

        # Build OR high/low: rolling max/min of first 15 bars per day
        cur_day = -1
        cur_high = 0.0
        cur_low = 1e18
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                cur_high = high[i]
                cur_low = low[i]
            else:
                if day_bar_count[i] <= 15:
                    cur_high = max(cur_high, high[i])
                    cur_low = min(cur_low, low[i])
            or_high[i] = cur_high
            or_low[i] = cur_low
            or_ready[i] = day_bar_count[i] > 15

        or_range = or_high - or_low
        or_range = np.where(or_range > 1e-10, or_range, 1e-10)

        # Volume SMA(20)
        vol_sma = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - 19)
            vol_sma[i] = np.mean(volume[start:i + 1])
        vol_sma = np.where(vol_sma > 1e-10, vol_sma, 1.0)
        vol_ok = volume > vol_mult * vol_sma

        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Entries
        long_entry = or_ready & (close > or_high) & vol_ok & in_session
        short_entry = or_ready & (close < or_low) & vol_ok & in_session

        # Target: 2x range from breakout level
        # Use target_pct based on 2x range / close approximation
        # Actually use ATR-based: target = 2x OR range expressed via target_indicator
        target_long = or_high + 2.0 * or_range
        target_short = or_low - 2.0 * or_range
        # Use the long target as target_indicator; for shorts the state machine mirrors
        target_indicator = np.where(close >= or_high, target_long, target_short)

        # Stop: opposite side of OR
        # Express as pct: or_range / close
        avg_stop_pct = np.nanmean(or_range / np.where(close > 1e-10, close, 1e-10))
        avg_stop_pct = float(np.nan_to_num(avg_stop_pct, nan=0.005))

        # ATR for atr_arr
        atr20 = _compute_atr(high, low, close, 20)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr20,
            target_indicator=target_indicator,
            use_target_indicator=True,
            stop_loss_pct=avg_stop_pct,
            breakeven_pct=be_pct,
            time_stop_bars=200,
        )
