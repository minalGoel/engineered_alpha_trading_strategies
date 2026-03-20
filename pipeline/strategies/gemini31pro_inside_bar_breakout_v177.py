"""Inside Bar Breakout — cursor_gemini31pro_strategy_177

Thesis: An inside bar represents intraday equilibrium and volatility contraction.
A breakout of the inside bar on increased volume indicates a resolution of this
equilibrium. Previous bar was inside bar, current bar breaks out of the mother
bar's range. Volume filter: rel_volume > 1.2. Target: 14 ATR. Time stop: 15 bars.
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
    name = "gemini31pro_inside_bar_breakout_v177"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rel_vol_thresh", default=1.2, low=0.8, high=2.0),
            TunableParam("stop_atr_mult", default=1.5, low=0.8, high=3.0),
            TunableParam("target_atr_mult", default=14.0, low=5.0, high=25.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rel_vol_t = params.get("rel_vol_thresh", 1.2)
        stop_atr = params.get("stop_atr_mult", 1.5)
        tgt_atr = params.get("target_atr_mult", 14.0)
        be_pct = params.get("breakeven_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high_arr = df["high"].to_numpy().astype(np.float64)
        low_arr = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr14 = _compute_atr(high_arr, low_arr, close, 14)

        # Inside bar detection:
        # Bar i is inside if high[i] < high[i-1] AND low[i] > low[i-1]
        # The "mother bar" is bar i-1 (the bar that contains the inside bar)
        is_inside = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            is_inside[i] = (high_arr[i] < high_arr[i - 1]) and (low_arr[i] > low_arr[i - 1])

        # Entry: previous bar was inside bar, current bar breaks the mother bar
        # Mother bar for inside bar at i is bar i-1, so mother high = high[i-1], mother low = low[i-1]
        # Entry at bar i+1: REF(is_inside, 1) and close breaks REF(mother_high, 1) or REF(mother_low, 1)
        # Actually from JSON: REF(is_inside, 1) AND close > REF(prev_high, 1)
        # prev_high at bar i = high[i-1], so REF(prev_high, 1) at bar i = high[i-2]
        # The mother bar high for inside bar at i-1 is high[i-2]
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            if is_inside[i - 1]:
                # Mother bar is i-2
                mother_high = high_arr[i - 2]
                mother_low = low_arr[i - 2]
                if close[i] > mother_high:
                    long_entry[i] = True
                if close[i] < mother_low:
                    short_entry[i] = True

        # Relative volume filter
        vol_sma20 = np.full(n, 1.0, dtype=np.float64)
        for i in range(19, n):
            vol_sma20[i] = np.mean(volume[i - 19:i + 1])
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        rel_vol = volume / vol_sma20
        vol_ok = rel_vol > rel_vol_t

        time_ok = (time_mins >= 570) & (time_mins <= 870)

        long_entry = long_entry & vol_ok & time_ok
        short_entry = short_entry & vol_ok & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_atr_mult=tgt_atr,
            breakeven_pct=be_pct,
            time_stop_bars=15,
        )
