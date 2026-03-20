"""Bid-Ask Imbalance v1 — cursor_opus46max_101

Thesis: When bid-side depth significantly exceeds ask-side depth (approximated
via volume-weighted price position within the bar), the resulting imbalance
predicts short-term price movement.  Since L2 order book data is unavailable
in OHLCV, we proxy bid-ask imbalance with (close - low) / (high - low) ratio
combined with volume confirmation and VWAP positioning.
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


def _ema(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = arr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_101"
    is_long_only = False
    session_start = 575   # 09:35 (skip first 20 min)
    session_end = 915     # 15:15
    max_trades_per_day = 20
    assumptions = ["L2 order book proxied via bar close position ratio"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bai_zscore_thresh", default=1.5, low=1.0, high=2.5),
            TunableParam("vol_ratio_thresh", default=1.2, low=0.8, high=2.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.001, low=0.0005, high=0.002),
            TunableParam("target_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        bai_zs_thresh = params.get("bai_zscore_thresh", 1.5)
        vol_ratio_thresh = params.get("vol_ratio_thresh", 1.2)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.001)
        target_pct = params.get("target_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Proxy bid-ask imbalance: (close - low) / (high - low) mapped to [-1, 1]
        bar_range = high - low
        bar_range = np.where(bar_range < 1e-10, 1e-10, bar_range)
        bai_raw = 2.0 * (close - low) / bar_range - 1.0  # -1 = close at low, +1 = close at high

        # EMA(BAI, 3)
        bai_ema = _ema(bai_raw, 3)

        # Z-score of BAI_EMA over 20 bars
        lookback = 20
        bai_zscore = np.zeros(n, dtype=np.float64)
        for i in range(lookback - 1, n):
            window = bai_ema[i - lookback + 1: i + 1]
            mu = np.mean(window)
            std = np.std(window)
            if std > 1e-10:
                bai_zscore[i] = (bai_ema[i] - mu) / std

        # Volume ratio
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(lookback - 1, n):
            avg_vol[i] = np.mean(volume[i - lookback + 1: i + 1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # Confirmation: BAI_zscore sustained for 2 consecutive bars
        sustained_long = np.zeros(n, dtype=np.bool_)
        sustained_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            sustained_long[i] = bai_zscore[i] > 1.0 and bai_zscore[i-1] > 1.0
            sustained_short[i] = bai_zscore[i] < -1.0 and bai_zscore[i-1] < -1.0

        vix_ok = vix < vix_max

        long_entry = ((bai_zscore > bai_zs_thresh) & (vol_ratio > vol_ratio_thresh) &
                      (close > vwap) & vix_ok & sustained_long)
        short_entry = ((bai_zscore < -bai_zs_thresh) & (vol_ratio > vol_ratio_thresh) &
                       (close < vwap) & vix_ok & sustained_short)

        # Signal exit: BAI_zscore crosses zero
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if bai_zscore[i] <= 0 and bai_zscore[i-1] > 0:
                sig_exit_long[i] = True
            if bai_zscore[i] >= 0 and bai_zscore[i-1] < 0:
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=0.0005,
            trailing_activate_pct=0.001,
            time_stop_bars=5,
        )
