"""Rolling Sharpe Adaptive v1 — cursor_opus46max_175

Thesis: EMA(9)/EMA(21) + VWAP momentum base signal. Adaptive sizing concept
based on rolling Sharpe of recent returns (simplified to standard entry/exit
since backtester uses fixed sizing). VWAP distance > 5 bps confirmation.
Signal exit on EMA cross against position.
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
    name = "cursor_opus46max_175"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dist_bps", default=5.0, low=3.0, high=10.0),
            TunableParam("sharpe_window", default=30.0, low=20.0, high=50.0),
            TunableParam("sharpe_floor", default=-1.0, low=-2.0, high=0.0),
            TunableParam("target_pct", default=0.0035, low=0.002, high=0.005),
            TunableParam("stop_pct", default=0.002, low=0.0012, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vwap_bps = params.get("vwap_dist_bps", 5.0)
        sharpe_win = int(params.get("sharpe_window", 30.0))
        sharpe_floor = params.get("sharpe_floor", -1.0)
        target = params.get("target_pct", 0.0035)
        stop = params.get("stop_pct", 0.002)

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

        # EMA(9)/EMA(21)
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)

        # 1-bar returns
        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0:
                returns[i] = close[i] / close[i - 1] - 1.0

        # Rolling Sharpe ratio of returns (proxy for recent strategy performance)
        rolling_sharpe = np.zeros(n, dtype=np.float64)
        for i in range(sharpe_win, n):
            window = returns[i - sharpe_win + 1:i + 1]
            m = np.mean(window)
            s = np.std(window, ddof=1)
            if s > 1e-20:
                rolling_sharpe[i] = m / s * np.sqrt(sharpe_win)

        # Sharpe filter: don't enter in deep drawdown
        sharpe_ok = rolling_sharpe > sharpe_floor

        # EMA cross detection for signal exit
        ema_cross_dn = np.zeros(n, dtype=np.bool_)
        ema_cross_up = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if ema9[i - 1] >= ema21[i - 1] and ema9[i] < ema21[i]:
                ema_cross_dn[i] = True
            if ema9[i - 1] <= ema21[i - 1] and ema9[i] > ema21[i]:
                ema_cross_up[i] = True

        # Volume filter
        avg_vol = df["volume"].rolling_mean(15).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 565) & (time_mins <= 915)

        # Long: EMA9 > EMA21, above VWAP by threshold, Sharpe not tanked
        long_entry = (ema9 > ema21) & (vwap_dist > vwap_bps) & \
                     sharpe_ok & vol_ok & time_ok

        # Short: EMA9 < EMA21, below VWAP by threshold, Sharpe not tanked
        short_entry = (ema9 < ema21) & (vwap_dist < -vwap_bps) & \
                      sharpe_ok & vol_ok & time_ok

        # Signal exit: EMA cross against position
        sig_exit_long = ema_cross_dn
        sig_exit_short = ema_cross_up

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop,
            target_pct=target,
            trailing_stop_pct=0.0012,
            trailing_activate_pct=0.002,
            time_stop_bars=60,
        )
