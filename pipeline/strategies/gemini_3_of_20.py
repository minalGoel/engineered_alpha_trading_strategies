# AUDIT FIX: Added session window filter to long_entry and short_entry
"""Sector Relative Strength Momentum — gemini_3_of_20

Thesis: When a stock outperforms its index (relative strength ratio > 1.02 over
a 50-bar SMA baseline), ride that momentum. Confirm with index trend (EMA20)
and VWAP alignment.
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
    name = "sector_relative_strength_momentum"
    is_long_only = False
    session_start = 630   # 10:30
    session_end = 900     # 15:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ratio_long_thresh", default=1.02, low=1.005, high=1.05),
            TunableParam("ratio_short_thresh", default=0.98, low=0.95, high=0.995),
            TunableParam("target_pct", default=0.008, low=0.003, high=0.015),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ratio_long = params.get("ratio_long_thresh", 1.02)
        ratio_short = params.get("ratio_short_thresh", 0.98)
        tgt_pct = params.get("target_pct", 0.008)
        stop_pct = params.get("stop_loss_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=1.0)
        index_close = np.where(index_close > 1e-10, index_close, 1.0)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=close[0] if n > 0 else 0.0)

        # Stock/index ratio
        ratio = close / index_close

        # 50-bar SMA of ratio (baseline)
        ratio_sma50 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - 49)
            ratio_sma50[i] = np.mean(ratio[start:i + 1])
        ratio_sma50 = np.where(ratio_sma50 > 1e-10, ratio_sma50, 1.0)

        # Relative strength: ratio / sma50 baseline
        rel_strength = ratio / ratio_sma50

        # EMA(20) of index
        ema20_idx = np.zeros(n, dtype=np.float64)
        alpha = 2.0 / 21.0
        ema20_idx[0] = index_close[0]
        for i in range(1, n):
            ema20_idx[i] = alpha * index_close[i] + (1 - alpha) * ema20_idx[i - 1]

        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Entries
        long_entry = ((rel_strength > ratio_long) &
                       (index_close > ema20_idx) &
                       (close > vwap) & in_session)
        short_entry = ((rel_strength < ratio_short) &
                        (index_close < ema20_idx) &
                        (close < vwap) & in_session)

        # Signal exit: VWAP cross
        signal_exit_long = close < vwap
        signal_exit_short = close > vwap

        atr14 = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=300,
        )
