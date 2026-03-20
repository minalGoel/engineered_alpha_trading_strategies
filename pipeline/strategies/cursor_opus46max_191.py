"""Quantile Regression v1 — cursor_opus46max_191

Thesis: Quantile regression estimates conditional quantiles of forward returns.
When the 10th percentile of predicted forward return is positive, we have a
high-confidence long signal. Simplified: use rolling percentile-based approach
with VWAP distance, volume ratio, and index return as features.
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
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_191"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("lookback", default=120.0, low=60.0, high=200.0),
            TunableParam("fwd_bars", default=30.0, low=15.0, high=45.0),
            TunableParam("p10_thresh_bps", default=0.0, low=-5.0, high=5.0),
            TunableParam("median_thresh_bps", default=10.0, low=5.0, high=20.0),
            TunableParam("confirm_roc_period", default=3.0, low=2.0, high=5.0),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        lb = int(params.get("lookback", 120.0))
        fwd = int(params.get("fwd_bars", 30.0))
        p10_th = params.get("p10_thresh_bps", 0.0)
        med_th = params.get("median_thresh_bps", 10.0)
        confirm_p = int(params.get("confirm_roc_period", 3.0))
        stop_pct = params.get("stop_loss_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Features
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        vwap_dist = (close - vwap) / safe_vwap * 10000  # bps

        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # Forward returns (historical, for building conditional distribution)
        fwd_ret = np.zeros(n, dtype=np.float64)
        for i in range(n - fwd):
            if close[i] > 0:
                fwd_ret[i] = (close[i + fwd] - close[i]) / close[i] * 10000

        # Conditional quantiles: stratified by VWAP distance direction
        qr_p10 = np.zeros(n, dtype=np.float64)
        qr_p90 = np.zeros(n, dtype=np.float64)
        qr_med = np.zeros(n, dtype=np.float64)

        for i in range(lb + fwd, n):
            # Historical forward returns in similar conditions
            hist_vwap = vwap_dist[i-lb-fwd:i-fwd]
            hist_fwd = fwd_ret[i-lb-fwd:i-fwd]

            # Similar VWAP distance: same sign and within 50% magnitude
            current_sign = 1 if vwap_dist[i] > 0 else -1
            similar = (np.sign(hist_vwap) == current_sign) | (np.abs(hist_vwap) < 10.0)

            if np.sum(similar) >= 10:
                subset = hist_fwd[similar]
                qr_p10[i] = np.percentile(subset, 10)
                qr_p90[i] = np.percentile(subset, 90)
                qr_med[i] = np.percentile(subset, 50)

        # ROC confirmation
        roc = np.zeros(n, dtype=np.float64)
        for i in range(confirm_p, n):
            if close[i-confirm_p] > 0:
                roc[i] = (close[i] - close[i-confirm_p]) / close[i-confirm_p] * 10000

        time_ok = (time_mins >= 615) & (time_mins <= 910)

        long_entry = (
            (qr_p10 > p10_th)
            & (qr_med > med_th)
            & ((close > vwap) | (vwap_dist > -10.0))
            & (roc > 0)
            & time_ok
        )
        short_entry = (
            (qr_p90 < -p10_th)
            & (qr_med < -med_th)
            & ((close < vwap) | (vwap_dist < 10.0))
            & (roc < 0)
            & time_ok
        )

        # Signal exit: quantile prediction flips
        sig_exit_long = qr_p10 < 0
        sig_exit_short = qr_p90 > 0

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=0.0035,
            trailing_activate_pct=0.0025,
            trailing_stop_pct=0.0012,
            time_stop_bars=45,
        )
