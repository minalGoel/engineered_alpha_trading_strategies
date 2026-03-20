"""VIX Filtered Index Trend — cursor_gemini31pro_strategy_124

Thesis: When VIX is high (>21), trend breakouts via EMA(9)/EMA(21) crossover
with VWAP confirmation offer massive reward/risk. 1min timeframe, FNO universe.
No additional VIX filter (entry condition IS the VIX filter). Time stop: 60 bars.
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


def _ema(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


class Strategy(BaseStrategy):
    name = "gemini31pro_vix_filtered_index_trend_v124"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_entry_min", default=21.0, low=12.0, high=30.0),
            TunableParam("stop_atr_mult", default=1.5, low=0.8, high=3.0),
            TunableParam("target_atr_mult", default=20.0, low=8.0, high=30.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_entry = params.get("vix_entry_min", 21.0)
        stop_atr = params.get("stop_atr_mult", 1.5)
        tgt_atr = params.get("target_atr_mult", 20.0)
        be_pct = params.get("breakeven_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr14 = _compute_atr(high, low, close, 14)

        # EMA crossover
        ema9 = _ema(close, 9)
        ema21 = _ema(close, 21)

        prev_ema9 = np.roll(ema9, 1)
        prev_ema9[0] = ema9[0]
        prev_ema21 = np.roll(ema21, 1)
        prev_ema21[0] = ema21[0]

        cross_up = (prev_ema9 <= prev_ema21) & (ema9 > ema21)
        cross_down = (prev_ema9 >= prev_ema21) & (ema9 < ema21)

        # VWAP daily reset
        df2 = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns(
            (pl.col("_ctv") / pl.col("_cv")).alias("_vwap")
        )
        vwap = df2["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=close[0])

        # Volume surge
        vol_sma20 = np.full(n, 1.0, dtype=np.float64)
        for i in range(19, n):
            vol_sma20[i] = np.mean(volume[i - 19:i + 1])
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        vol_ok = volume > vol_sma20

        # Filters
        vix_ok = vix > vix_entry
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        long_entry = vix_ok & cross_up & (close > vwap) & vol_ok & time_ok
        short_entry = vix_ok & cross_down & (close < vwap) & vol_ok & time_ok

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
            time_stop_bars=60,
        )
