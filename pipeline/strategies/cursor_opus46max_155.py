"""Factor Momentum v1 — cursor_opus46max_155

Thesis: Stocks with strong 60-bar momentum that are above VWAP,
combined with a short-term 5-bar ROC confirmation. Cross-sectional
factor approach simplified to single-stock momentum + VWAP filter.
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
    name = "cursor_opus46max_155"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 900     # 15:00
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("mom60_thresh_bps", default=20.0, low=10.0, high=50.0),
            TunableParam("roc5_thresh_bps", default=0.0, low=-10.0, high=15.0),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_stop_pct", default=0.002, low=0.001, high=0.003),
            TunableParam("trailing_activate_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        mom60_thresh = params.get("mom60_thresh_bps", 20.0)
        roc5_thresh = params.get("roc5_thresh_bps", 0.0)
        target_pct = params.get("target_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.003)
        trail_pct = params.get("trailing_stop_pct", 0.002)
        trail_act = params.get("trailing_activate_pct", 0.003)

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

        # 60-bar momentum (ROC in bps)
        mom60 = np.zeros(n, dtype=np.float64)
        for i in range(60, n):
            if close[i - 60] > 0:
                mom60[i] = (close[i] / close[i - 60] - 1.0) * 10000.0

        # 5-bar ROC (bps)
        roc5 = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            if close[i - 5] > 0:
                roc5[i] = (close[i] / close[i - 5] - 1.0) * 10000.0

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 615) & (time_mins <= 900)

        # Long: positive 60-bar momentum + above VWAP + positive 5-bar ROC
        long_entry = (mom60 > mom60_thresh) & (close > vwap) & \
                     (roc5 > roc5_thresh) & vol_ok & time_ok

        # Short: negative 60-bar momentum + below VWAP + negative 5-bar ROC
        short_entry = (mom60 < -mom60_thresh) & (close < vwap) & \
                      (roc5 < -roc5_thresh) & vol_ok & time_ok

        # Signal exit: momentum reversal
        sig_exit_long = mom60 < 0
        sig_exit_short = mom60 > 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=90,
        )
