# AUDIT FIX: Added session window filter to long_entry and short_entry
"""VWAP Pinch Breakout — gemini_5_of_20

Thesis: When EMA(20) and VWAP converge tightly (pinch), a breakout with volume
confirms directional momentum.
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
    name = "vwap_pinch_breakout"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 12

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("pinch_thresh", default=0.001, low=0.0003, high=0.003),
            TunableParam("target_atr_mult", default=3.0, low=1.5, high=5.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.006),
            TunableParam("breakeven_pct", default=0.003, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        pinch_thresh = params.get("pinch_thresh", 0.001)
        tgt_atr = params.get("target_atr_mult", 3.0)
        stop_pct = params.get("stop_loss_pct", 0.003)
        be_pct = params.get("breakeven_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=close[0] if n > 0 else 0.0)

        # EMA(20)
        ema20 = np.zeros(n, dtype=np.float64)
        alpha = 2.0 / 21.0
        ema20[0] = close[0]
        for i in range(1, n):
            ema20[i] = alpha * close[i] + (1 - alpha) * ema20[i - 1]

        # Pinch: abs(EMA20 - VWAP) / close < threshold
        safe_close = np.where(close > 1e-10, close, 1e-10)
        pinch = np.abs(ema20 - vwap) / safe_close < pinch_thresh

        # Volume rising: current > 5-bar ago
        vol_rising = np.zeros(n, dtype=np.bool_)
        for i in range(5, n):
            vol_rising[i] = volume[i] > volume[i - 5]

        # ATR(20)
        atr20 = _compute_atr(high, low, close, 20)

        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Entries
        long_entry = pinch & (close > ema20) & (ema20 > vwap) & vol_rising & in_session
        short_entry = pinch & (close < ema20) & (ema20 < vwap) & vol_rising & in_session

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr20,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_atr_mult=tgt_atr,
            stop_loss_pct=stop_pct,
            breakeven_pct=be_pct,
            time_stop_bars=120,
        )
