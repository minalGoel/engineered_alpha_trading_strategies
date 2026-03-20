"""Robust Regression v1 — cursor_opus46max_190

Thesis: Huber robust regression gives more stable trend estimates than OLS by
down-weighting outliers. Fit on 30-bar window. When slope t-score > 2.0 and
price is below (above) the robust trend line, buy (sell) the dip/rip expecting
reversion to the trend. VWAP confirmation required.
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


def _robust_linreg(y, epsilon=1.35, max_iter=50):
    """Simplified Huber-like robust regression via iteratively reweighted least squares."""
    n = len(y)
    x = np.arange(n, dtype=np.float64)
    x_m = x - np.mean(x)
    y_m = y - np.mean(y)

    # Initial OLS
    denom = np.sum(x_m ** 2)
    if denom < 1e-15:
        return 0.0, np.mean(y), np.zeros(n)

    slope = np.sum(x_m * y_m) / denom
    intercept = np.mean(y) - slope * np.mean(x)

    for _ in range(max_iter):
        residuals = y - (slope * x + intercept)
        mad = np.median(np.abs(residuals - np.median(residuals)))
        sigma = max(mad * 1.4826, 1e-10)
        scaled = np.abs(residuals) / sigma

        # Huber weights
        weights = np.where(scaled <= epsilon, 1.0, epsilon / scaled)

        # Weighted least squares
        w_sum = np.sum(weights)
        if w_sum < 1e-10:
            break
        wx_mean = np.sum(weights * x) / w_sum
        wy_mean = np.sum(weights * y) / w_sum
        xd = x - wx_mean
        yd = y - wy_mean
        denom = np.sum(weights * xd ** 2)
        if denom < 1e-15:
            break
        slope = np.sum(weights * xd * yd) / denom
        intercept = wy_mean - slope * wx_mean

    residuals = y - (slope * x + intercept)
    return slope, intercept, residuals


class Strategy(BaseStrategy):
    name = "cursor_opus46max_190"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 910     # 15:10
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("reg_window", default=30.0, low=20.0, high=50.0),
            TunableParam("tscore_thresh", default=2.0, low=1.5, high=3.0),
            TunableParam("residual_max_sigma", default=2.0, low=1.5, high=3.0),
            TunableParam("tscore_exit_thresh", default=1.5, low=1.0, high=2.0),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        reg_win = int(params.get("reg_window", 30.0))
        ts_thresh = params.get("tscore_thresh", 2.0)
        res_max_sigma = params.get("residual_max_sigma", 2.0)
        ts_exit = params.get("tscore_exit_thresh", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.0025)

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

        # Rolling robust regression
        huber_slope = np.zeros(n, dtype=np.float64)
        huber_line = np.zeros(n, dtype=np.float64)
        residual = np.zeros(n, dtype=np.float64)
        residual_std = np.zeros(n, dtype=np.float64)
        slope_tscore = np.zeros(n, dtype=np.float64)

        for i in range(reg_win - 1, n):
            seg = close[i-reg_win+1:i+1]
            slope, intercept, resids = _robust_linreg(seg)
            huber_slope[i] = slope
            huber_line[i] = intercept + slope * (reg_win - 1)  # value at current bar
            residual[i] = close[i] - huber_line[i]

            # MAD-based robust std
            mad = np.median(np.abs(resids - np.median(resids)))
            residual_std[i] = max(mad * 1.4826, 1e-10)

            # Approximate t-score for slope
            x = np.arange(reg_win, dtype=np.float64)
            sum_x_sq = np.sum((x - np.mean(x)) ** 2)
            if sum_x_sq > 0:
                se = residual_std[i] / np.sqrt(sum_x_sq)
                if se > 1e-10:
                    slope_tscore[i] = slope / se

        # Confirmation: close reverting
        confirm_up = np.zeros(n, dtype=np.bool_)
        confirm_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            confirm_up[i] = close[i] > close[i-1]
            confirm_down[i] = close[i] < close[i-1]

        time_ok = (time_mins >= 585) & (time_mins <= 910)

        # Normalized residual
        norm_res = np.zeros(n, dtype=np.float64)
        for i in range(reg_win - 1, n):
            if residual_std[i] > 1e-10:
                norm_res[i] = residual[i] / residual_std[i]

        long_entry = (
            (slope_tscore > ts_thresh)
            & (residual < 0)
            & (norm_res > -res_max_sigma)
            & (close > vwap)
            & confirm_up
            & time_ok
        )
        short_entry = (
            (slope_tscore < -ts_thresh)
            & (residual > 0)
            & (norm_res < res_max_sigma)
            & (close < vwap)
            & confirm_down
            & time_ok
        )

        # Signal exit: trend weakens or residual crosses zero
        sig_exit_long = (np.abs(slope_tscore) < ts_exit) | (residual >= 0)
        sig_exit_short = (np.abs(slope_tscore) < ts_exit) | (residual <= 0)

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=60,
        )
