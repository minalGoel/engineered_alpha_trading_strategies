# AUDIT FIX: rolling_max/min(5) includes current bar, so close > high_5 is always False
#             (since close <= high[i] which is included in rolling_max(5)).
#             Fix: shift the rolling window by 1 to reference *previous* 5 bars only.
# AUDIT FIX: Added session window filter (time_mins <= session_end) to prevent post-session signals.
"""Volume Breakout — GPT_8_of_10

Thesis: High-volume breakouts above the 5-bar high or below the 5-bar low
signal strong directional conviction. Volume confirms institutional participation.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "volume_breakout_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=2.0, low=1.2, high=4.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.01),
            TunableParam("target_pct", default=0.01, low=0.005, high=0.02),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 2.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        target_pct = params.get("target_pct", 0.01)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        # ── 5-bar rolling high / low (shifted by 1 so current bar is excluded) ──
        # rolling_max(5) on "high" includes current bar; close can never exceed
        # current bar's high, so close > rolling_max(5) is always False.
        # Shift by 1 to compare against the *previous* 5 bars' range.
        high_5 = df["high"].rolling_max(5).shift(1).to_numpy().astype(np.float64)
        low_5 = df["low"].rolling_min(5).shift(1).to_numpy().astype(np.float64)
        high_5 = np.nan_to_num(high_5, nan=1e10)
        low_5 = np.nan_to_num(low_5, nan=-1e10)

        # ── Volume filter ──
        avg_vol_10 = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol_10 = np.nan_to_num(avg_vol_10, nan=1.0)
        avg_vol_10 = np.clip(avg_vol_10, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol_10

        # ── Time filter: skip first 10 bars and enforce session end ──
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        time_ok = (time_mins >= 565) & (time_mins <= self.session_end)  # 09:25–15:15 IST

        # ── Entry ──
        long_entry = (close > high_5) & vol_ok & time_ok
        short_entry = (close < low_5) & vol_ok & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=15,
        )
