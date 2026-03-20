"""Pullback Momentum — Grok_9_of_10

Thesis: Pullbacks to VWAP in strong momentum stocks offer low-risk entries.
Uses VWAP z-score (normalized by ATR) with volume and index confirmation.
Target is VWAP touch (mean reversion).
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "pullback_momentum_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 9

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=0.8, low=0.3, high=2.0),
            TunableParam("vix_max", default=19.0, low=12.0, high=28.0),
            TunableParam("rel_vol_thresh", default=1.5, low=1.0, high=3.0),
            TunableParam("stop_atr_mult", default=1.2, low=0.5, high=2.5),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 0.8)
        vix_max = params.get("vix_max", 19.0)
        rv_thresh = params.get("rel_vol_thresh", 1.5)
        stop_atr = params.get("stop_atr_mult", 1.2)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        idx = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)

        atr14 = _compute_atr(high, low, close, 14)
        atr14_safe = np.where(atr14 > 0, atr14, 1e10)

        # VWAP (daily reset)
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Z-score: (close - vwap) / ATR
        zscore = (close - vwap) / atr14_safe

        # Relative volume
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        rel_vol = volume / avg_vol_20

        # Index 5-bar return
        idx_ret5 = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            if idx[i-5] > 0:
                idx_ret5[i] = (idx[i] - idx[i-5]) / idx[i-5]

        # Filters
        vix_ok = vix < vix_max
        vol_ok = rel_vol > rv_thresh

        # Entry: pullback from VWAP
        long_entry = (zscore < -zs_thresh) & vol_ok & vix_ok & (idx_ret5 > 0.001)
        short_entry = (zscore > zs_thresh) & vol_ok & vix_ok & (idx_ret5 < -0.001)

        # Target: VWAP touch
        target_indicator = vwap.copy()

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=target_indicator,
            stop_loss_atr_mult=stop_atr,
            use_target_indicator=True,
            time_stop_bars=25,
        )
