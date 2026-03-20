# AUDIT FIX: Added session window enforcement — entries were firing outside session_start/session_end
"""Afternoon Profit Fade — gemini_10_of_20

Thesis: Near end of day, extreme intraday moves (> 3% from open) tend to
partially revert as traders take profits. Fade the move targeting VWAP.
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
    name = "afternoon_profit_fade"
    is_long_only = False
    session_start = 885   # 14:45
    session_end = 925     # 15:25
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("day_return_thresh", default=0.03, low=0.015, high=0.05),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ret_thresh = params.get("day_return_thresh", 0.03)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=close[0] if n > 0 else 0.0)

        # Day open price (first open of each day)
        day_open = np.zeros(n, dtype=np.float64)
        prev_day = -1
        cur_open = 0.0
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                cur_open = opn[i]
            day_open[i] = cur_open

        # Day return
        safe_open = np.where(day_open > 1e-10, day_open, 1e-10)
        day_return = (close - day_open) / safe_open

        # Session window filter
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # After 14:45 (885) — also enforce session_end (925)
        after_time = (time_mins >= 885) & in_session

        atr14 = _compute_atr(high, low, close, 14)

        # Long: day down big, fade short side (buy the dip)
        long_entry = after_time & (day_return < -ret_thresh) & (close < vwap)
        # Short: day up big, fade long side (sell the rip)
        short_entry = after_time & (day_return > ret_thresh) & (close > vwap)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=vwap.copy(),
            use_target_indicator=True,
            stop_loss_pct=stop_pct,
            time_stop_bars=40,
        )
