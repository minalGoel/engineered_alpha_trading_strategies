# AUDIT FIX: Added session window filter to long_entry and short_entry
"""Previous Day High/Low Reversal — gemini_18_of_20

Thesis: When price pierces the previous day's high or low but fails to hold,
it signals a false breakout/breakdown. Combined with RSI extremes, fade the
move expecting a reversal back toward VWAP.
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


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "prev_day_hl_reversal"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 920     # 15:20
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_long_thresh", default=30.0, low=15.0, high=40.0),
            TunableParam("rsi_short_thresh", default=70.0, low=60.0, high=85.0),
            TunableParam("stop_pct", default=0.4, low=0.2, high=0.8),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rsi_long = params.get("rsi_long_thresh", 30.0)
        rsi_short = params.get("rsi_short_thresh", 70.0)
        stop_pct = params.get("stop_pct", 0.4)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        day_id = df["day_id"].to_numpy()

        atr14 = _compute_atr(high, low, close, 14)
        rsi14 = _compute_rsi(close, 14)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=close[0] if n > 0 else 0.0)

        # Compute previous day high/low
        sorted_days = sorted(set(day_id))
        day_hl = {}
        for d in sorted_days:
            mask = day_id == d
            day_hl[d] = (np.max(high[mask]), np.min(low[mask]))

        prev_day_high = np.full(n, np.inf, dtype=np.float64)
        prev_day_low = np.full(n, -np.inf, dtype=np.float64)
        for idx, d in enumerate(sorted_days):
            if idx > 0:
                prev_d = sorted_days[idx - 1]
                prev_h, prev_l = day_hl[prev_d]
            else:
                prev_h, prev_l = high[0], low[0]
            mask = day_id == d
            prev_day_high[mask] = prev_h
            prev_day_low[mask] = prev_l

        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Long: low pierces prev day low but close recovers above it + RSI < thresh
        long_entry = (low < prev_day_low) & (close > prev_day_low) & (rsi14 < rsi_long) & in_session

        # Short: high pierces prev day high but close falls back below + RSI > thresh
        short_entry = (high > prev_day_high) & (close < prev_day_high) & (rsi14 > rsi_short) & in_session

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=vwap.copy(),
            use_target_indicator=True,
            stop_loss_pct=stop_pct / 100.0,
            time_stop_bars=40,
        )
