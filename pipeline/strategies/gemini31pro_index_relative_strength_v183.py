"""Index Relative Strength — cursor_gemini31pro_strategy_183

Thesis: Stocks showing resilience while index falls explode upwards when the
index stabilizes. 5min timeframe, nifty200. ROC lookback: 20.
Index threshold: -30 bps, stock > 1.6 bps.
VIX > 15. No volume filter. Target: VWAP cross. Time stop: 60 bars.
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


def _roc(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        denom = arr[i - period]
        if abs(denom) > 1e-10:
            out[i] = (arr[i] - denom) / denom * 10000.0
    return out


class Strategy(BaseStrategy):
    name = "gemini31pro_index_relative_strength_v183"
    is_long_only = False
    session_start = 570
    session_end = 915
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("idx_thresh", default=30.0, low=10.0, high=60.0),
            TunableParam("stock_thresh", default=1.6, low=0.5, high=5.0),
            TunableParam("vix_min", default=15.0, low=10.0, high=25.0),
            TunableParam("stop_atr_mult", default=1.5, low=0.8, high=3.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        idx_thresh = params.get("idx_thresh", 30.0)
        stock_thresh = params.get("stock_thresh", 1.6)
        vix_min = params.get("vix_min", 15.0)
        stop_atr = params.get("stop_atr_mult", 1.5)
        be_pct = params.get("breakeven_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=close[0])
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr14 = _compute_atr(high, low, close, 14)

        index_ret = _roc(index_close, 20)
        stock_ret = _roc(close, 20)

        prev_idx_ret = np.roll(index_ret, 1)
        prev_idx_ret[0] = index_ret[0]

        # VWAP
        df2 = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns(
            (pl.col("_ctv") / pl.col("_cv")).alias("_vwap")
        )
        vwap = np.nan_to_num(df2["_vwap"].to_numpy().astype(np.float64), nan=close[0])

        # Volume surge (for confirmation, even though JSON says no volume filter)
        vol_sma20 = np.full(n, 1.0, dtype=np.float64)
        for i in range(19, n):
            vol_sma20[i] = np.mean(volume[i - 19:i + 1])
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        vol_ok = volume > vol_sma20

        vix_ok = vix > vix_min
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        long_entry = (index_ret < -idx_thresh) & (stock_ret > stock_thresh) & (index_ret > prev_idx_ret) & vol_ok & vix_ok & time_ok
        short_entry = (index_ret > idx_thresh) & (stock_ret < -stock_thresh) & (index_ret < prev_idx_ret) & vol_ok & vix_ok & time_ok

        signal_exit_long = close >= vwap
        signal_exit_short = close <= vwap

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=vwap,
            use_target_indicator=True,
            stop_loss_atr_mult=stop_atr,
            breakeven_pct=be_pct,
            time_stop_bars=60,
        )
