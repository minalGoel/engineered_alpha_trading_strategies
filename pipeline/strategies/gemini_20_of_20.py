"""Fibonacci Intraday Retracement — gemini_20_of_20

Thesis: After the morning move establishes a range, the 61.8% Fibonacci
retracement level acts as strong support/resistance. Enter when price
touches the fib level and confirms with VWAP, targeting the morning extreme.
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
    name = "fibonacci_intraday_retracement"
    is_long_only = False
    session_start = 660   # 11:00
    session_end = 900     # 15:00
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("fib_level", default=0.618, low=0.5, high=0.786),
            TunableParam("stop_pct", default=0.4, low=0.2, high=0.8),
            TunableParam("be_pct", default=0.5, low=0.2, high=1.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        fib_level = params.get("fib_level", 0.618)
        stop_pct = params.get("stop_pct", 0.4)
        be_pct = params.get("be_pct", 0.5)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        day_id = df["day_id"].to_numpy()

        atr14 = _compute_atr(high, low, close, 14)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=close[0] if n > 0 else 0.0)

        # Compute morning high/low (first 60 bars of each day)
        # and fibonacci retracement levels
        morning_high = np.full(n, np.nan, dtype=np.float64)
        morning_low = np.full(n, np.nan, dtype=np.float64)
        fib_long_level = np.full(n, np.nan, dtype=np.float64)   # support for long
        fib_short_level = np.full(n, np.nan, dtype=np.float64)  # resistance for short
        target_level = np.full(n, np.nan, dtype=np.float64)

        sorted_days = sorted(set(day_id))
        for d in sorted_days:
            day_mask = np.where(day_id == d)[0]
            if len(day_mask) == 0:
                continue

            # First 60 bars of the day (morning session)
            morning_bars = day_mask[:60]
            m_high = np.max(high[morning_bars])
            m_low = np.min(low[morning_bars])
            m_range = m_high - m_low

            if m_range < 1e-10:
                # No range — skip
                for idx in day_mask:
                    morning_high[idx] = m_high
                    morning_low[idx] = m_low
                    fib_long_level[idx] = m_low
                    fib_short_level[idx] = m_high
                    target_level[idx] = close[idx]
                continue

            # For uptrend retracement (long): fib support = high - fib * range
            fib_support = m_high - fib_level * m_range
            # For downtrend retracement (short): fib resistance = low + fib * range
            fib_resistance = m_low + fib_level * m_range

            for idx in day_mask:
                morning_high[idx] = m_high
                morning_low[idx] = m_low
                fib_long_level[idx] = fib_support
                fib_short_level[idx] = fib_resistance
                # Target: morning high for longs, morning low for shorts
                # We'll use a blended target indicator
                target_level[idx] = m_high  # default to long target

        morning_high = np.nan_to_num(morning_high, nan=high[0] if n > 0 else 0.0)
        morning_low = np.nan_to_num(morning_low, nan=low[0] if n > 0 else 0.0)
        fib_long_level = np.nan_to_num(fib_long_level, nan=low[0] if n > 0 else 0.0)
        fib_short_level = np.nan_to_num(fib_short_level, nan=high[0] if n > 0 else 0.0)
        target_level = np.nan_to_num(target_level, nan=close[0] if n > 0 else 0.0)

        # Long: low touches fib support level + close above it + close > VWAP
        long_entry = (
            (low <= fib_long_level)
            & (close > fib_long_level)
            & (close > vwap)
        )

        # Short: high touches fib resistance level + close below it + close < VWAP
        short_entry = (
            (high >= fib_short_level)
            & (close < fib_short_level)
            & (close < vwap)
        )

        # Target indicator: morning high for longs, morning low for shorts
        # Use morning_high as target (works for long side; state machine handles short)
        # For a unified approach, use the morning extreme on the entry side
        target_ind = np.where(long_entry, morning_high, morning_low)
        # For bars with no entry, use morning_high as default
        target_ind = np.where(long_entry | short_entry, target_ind, morning_high)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=target_ind,
            use_target_indicator=True,
            stop_loss_pct=stop_pct / 100.0,
            breakeven_pct=be_pct / 100.0,
            time_stop_bars=180,
        )
