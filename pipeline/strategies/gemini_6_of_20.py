"""European Open Momentum — gemini_6_of_20

Thesis: European market close at ~13:30 local time can inject fresh momentum
into the local market. Trade aligned with VWAP and index direction.
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
    name = "european_open_momentum"
    is_long_only = False
    session_start = 810   # 13:30
    session_end = 915     # 15:15
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("mom_long_thresh", default=1.005, low=1.001, high=1.015),
            TunableParam("mom_short_thresh", default=0.995, low=0.985, high=0.999),
            TunableParam("target_pct", default=0.007, low=0.003, high=0.012),
            TunableParam("stop_loss_pct", default=0.0035, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        mom_long = params.get("mom_long_thresh", 1.005)
        mom_short = params.get("mom_short_thresh", 0.995)
        tgt_pct = params.get("target_pct", 0.007)
        stop_pct = params.get("stop_loss_pct", 0.0035)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=1.0)
        index_close = np.where(index_close > 1e-10, index_close, 1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=close[0] if n > 0 else 0.0)

        # 30-bar momentum: close / close[i-30]
        momentum = np.ones(n, dtype=np.float64)
        for i in range(30, n):
            prev = close[i - 30]
            if prev > 1e-10:
                momentum[i] = close[i] / prev

        # Index direction: current > 10-bar ago
        idx_bullish = np.zeros(n, dtype=np.bool_)
        idx_bearish = np.zeros(n, dtype=np.bool_)
        for i in range(10, n):
            idx_bullish[i] = index_close[i] > index_close[i - 10]
            idx_bearish[i] = index_close[i] < index_close[i - 10]

        # After 13:31 = 811
        after_start = time_mins >= 811

        atr14 = _compute_atr(high, low, close, 14)

        # Entries
        long_entry = (after_start & (close > vwap) &
                       (momentum > mom_long) & idx_bullish)
        short_entry = (after_start & (close < vwap) &
                        (momentum < mom_short) & idx_bearish)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=120,
        )
