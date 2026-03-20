"""Mean Reversion Speed (OU Process) v1 — cursor_opus46max_179

Thesis: Estimate the Ornstein-Uhlenbeck mean-reversion speed (theta) from
VWAP deviation series. High theta (>0.05) = fast reversion; trade large VWAP
deviations (Z-score > 2) expecting quick profit. Low theta = skip.
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
    name = "cursor_opus46max_179"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 910     # 15:10
    max_trades_per_day = 7

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ou_window", default=60.0, low=40.0, high=90.0),
            TunableParam("theta_min", default=0.05, low=0.02, high=0.10),
            TunableParam("zscore_entry", default=2.0, low=1.5, high=3.0),
            TunableParam("zscore_stop", default=3.0, low=2.5, high=4.0),
            TunableParam("theta_exit_min", default=0.02, low=0.01, high=0.04),
            TunableParam("theta_confirm_bars", default=10.0, low=5.0, high=15.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ou_win = int(params.get("ou_window", 60.0))
        theta_min = params.get("theta_min", 0.05)
        zs_entry = params.get("zscore_entry", 2.0)
        zs_stop = params.get("zscore_stop", 3.0)
        theta_exit = params.get("theta_exit_min", 0.02)
        theta_confirm = int(params.get("theta_confirm_bars", 10.0))
        stop_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # VWAP deviation
        vwap_dev = close - vwap

        # OLS: delta(vwap_dev) = theta * vwap_dev + intercept => theta = -slope
        ou_theta = np.zeros(n, dtype=np.float64)
        ou_sigma = np.zeros(n, dtype=np.float64)
        for i in range(ou_win + 1, n):
            x = vwap_dev[i-ou_win:i]          # vwap_dev[t-1]
            dy = np.diff(vwap_dev[i-ou_win:i+1])  # delta(vwap_dev)
            x_m = x - np.mean(x)
            dy_m = dy - np.mean(dy)
            denom = np.sum(x_m ** 2)
            if denom > 1e-15:
                slope = np.sum(x_m * dy_m) / denom
                ou_theta[i] = -slope  # theta is negative of slope
                # residuals
                intercept = np.mean(dy) - slope * np.mean(x)
                residuals = dy - (slope * x + intercept)
                ou_sigma[i] = np.std(residuals)

        # Z-score using OU parameters
        zscore_ou = np.zeros(n, dtype=np.float64)
        for i in range(ou_win + 1, n):
            if ou_theta[i] > 0.001 and ou_sigma[i] > 1e-10:
                eq_std = ou_sigma[i] / np.sqrt(2.0 * ou_theta[i])
                if eq_std > 1e-10:
                    zscore_ou[i] = vwap_dev[i] / eq_std

        # Theta above minimum for confirm_bars
        theta_ok = np.zeros(n, dtype=np.bool_)
        for i in range(theta_confirm, n):
            theta_ok[i] = all(ou_theta[i-j] > 0.03 for j in range(theta_confirm))

        # Expected half-life
        exp_hl = np.full(n, 999.0, dtype=np.float64)
        for i in range(ou_win + 1, n):
            if ou_theta[i] > 0.001:
                exp_hl[i] = np.log(2.0) / ou_theta[i]

        time_ok = (time_mins >= 615) & (time_mins <= 910)

        long_entry = (
            (ou_theta > theta_min)
            & (zscore_ou < -zs_entry)
            & (exp_hl < 15.0)
            & theta_ok
            & time_ok
        )
        short_entry = (
            (ou_theta > theta_min)
            & (zscore_ou > zs_entry)
            & (exp_hl < 15.0)
            & theta_ok
            & time_ok
        )

        # Signal exit: theta drops below threshold OR z-score crosses zero
        sig_exit_long = (ou_theta < theta_exit) | (zscore_ou >= 0)
        sig_exit_short = (ou_theta < theta_exit) | (zscore_ou <= 0)

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            time_stop_bars=45,
        )
