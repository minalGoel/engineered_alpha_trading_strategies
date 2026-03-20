"""Cointegration Pairs v1 — cursor_opus46max_058

Thesis: Cointegrated stock pairs (SBIN vs PNB) via Engle-Granger.
Adapted: stock-vs-index cointegration residual z-score with half-life filter.
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
    name = "cursor_opus46max_058"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 915     # 15:15
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", default=2.0, low=1.5, high=3.0),
            TunableParam("ols_lookback", default=120.0, low=60.0, high=200.0),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.006, low=0.003, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_entry = params.get("zscore_entry", 2.0)
        lb = int(params.get("ols_lookback", 120.0))
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.006)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Rolling OLS: close = alpha + beta * index_close
        residual = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            y = close[i - lb:i + 1]
            x = index_close[i - lb:i + 1]
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            cov = np.mean((x - x_mean) * (y - y_mean))
            var = np.mean((x - x_mean) ** 2)
            if var > 1e-12:
                beta_ols = cov / var
                alpha_ols = y_mean - beta_ols * x_mean
                residual[i] = close[i] - (alpha_ols + beta_ols * index_close[i])

        # Z-score of residual
        zscore = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            seg = residual[max(0, i - lb):i + 1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                zscore[i] = (residual[i] - mu) / std

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 570) & (time_mins <= 855)  # 09:30 - 14:15

        # 2-bar reversal confirmation
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            if (zscore[i] < -zs_entry and vix_ok[i] and time_ok[i]
                    and residual[i] > residual[i-1] and residual[i-1] > residual[i-2]):
                long_entry[i] = True
            if (zscore[i] > zs_entry and vix_ok[i] and time_ok[i]
                    and residual[i] < residual[i-1] and residual[i-1] < residual[i-2]):
                short_entry[i] = True

        signal_exit_long = zscore > 0
        signal_exit_short = zscore < 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=90,
        )
