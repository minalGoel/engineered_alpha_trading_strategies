"""Autocorrelation Regime v1 — cursor_opus46max_171

Thesis: 1-lag autocorrelation of 1-min returns classifies regime.
Autocorr > 0.15: momentum (trend-follow). Autocorr < -0.15: mean-reversion
(fade the move). Between: neutral, no trade.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_ema(arr, period):
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = arr[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_171"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("autocorr_thresh", default=0.15, low=0.08, high=0.25),
            TunableParam("mom_roc3_bps", default=10.0, low=5.0, high=20.0),
            TunableParam("rev_roc3_bps", default=15.0, low=8.0, high=25.0),
            TunableParam("mom_target_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("rev_target_pct", default=0.002, low=0.001, high=0.003),
            TunableParam("mom_stop_pct", default=0.0018, low=0.001, high=0.003),
            TunableParam("rev_stop_pct", default=0.0012, low=0.0008, high=0.002),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ac_thresh = params.get("autocorr_thresh", 0.15)
        mom_roc3 = params.get("mom_roc3_bps", 10.0)
        rev_roc3 = params.get("rev_roc3_bps", 15.0)
        mom_target = params.get("mom_target_pct", 0.003)
        rev_target = params.get("rev_target_pct", 0.002)
        mom_stop = params.get("mom_stop_pct", 0.0018)
        rev_stop = params.get("rev_stop_pct", 0.0012)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        vwap_dist = (close - vwap) / safe_vwap * 10000.0

        # 1-bar returns
        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0:
                returns[i] = close[i] / close[i - 1] - 1.0

        # Rolling autocorrelation (lag=1, window=30)
        autocorr = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            window = returns[i - 29:i + 1]
            x = window[:-1]
            y = window[1:]
            mx = np.mean(x)
            my = np.mean(y)
            num = np.sum((x - mx) * (y - my))
            den = np.sqrt(np.sum((x - mx) ** 2) * np.sum((y - my) ** 2))
            if den > 1e-20:
                autocorr[i] = num / den

        # Regime
        is_momentum = autocorr > ac_thresh
        is_reversion = autocorr < -ac_thresh

        # 3-bar ROC (bps)
        roc3 = np.zeros(n, dtype=np.float64)
        for i in range(3, n):
            if close[i - 3] > 0:
                roc3[i] = (close[i] / close[i - 3] - 1.0) * 10000.0

        # 1-bar ROC for confirmation
        roc1 = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0:
                roc1[i] = (close[i] / close[i - 1] - 1.0) * 10000.0

        ema5 = _compute_ema(close, 5)

        # Volume filter
        avg_vol = df["volume"].rolling_mean(15).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 570) & (time_mins <= 910)

        # Momentum long: positive autocorr + positive ROC3 + above EMA5 + uptick
        mom_long = is_momentum & (roc3 > mom_roc3) & (close > ema5) & (roc1 > 0) & vol_ok & time_ok
        # Reversion long: negative autocorr + negative ROC3 + below VWAP + uptick
        rev_long = is_reversion & (roc3 < -rev_roc3) & (vwap_dist < -25.0) & (roc1 > 0) & vol_ok & time_ok

        # Momentum short
        mom_short = is_momentum & (roc3 < -mom_roc3) & (close < ema5) & (roc1 < 0) & vol_ok & time_ok
        # Reversion short
        rev_short = is_reversion & (roc3 > rev_roc3) & (vwap_dist > 25.0) & (roc1 < 0) & vol_ok & time_ok

        long_entry = mom_long | rev_long
        short_entry = mom_short | rev_short

        # Signal exit: regime transitions to neutral or opposite
        is_neutral = (~is_momentum) & (~is_reversion)
        sig_exit_long = is_neutral
        sig_exit_short = is_neutral

        avg_target = (mom_target + rev_target) / 2.0
        avg_stop = (mom_stop + rev_stop) / 2.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=avg_stop,
            target_pct=avg_target,
            trailing_stop_pct=0.001,
            trailing_activate_pct=0.002,
            time_stop_bars=45,
        )
