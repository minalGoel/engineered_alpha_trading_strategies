"""Beta Neutral Stat Arb v1 — cursor_opus46max_057

Thesis: High-beta vs low-beta stock alpha spread mean-reverts intraday.
Adapted: stock alpha residual (vs index) z-score entry.
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
    name = "cursor_opus46max_057"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("alpha_z_entry", default=1.8, low=1.2, high=2.8),
            TunableParam("alpha_z_confirm", default=1.5, low=1.0, high=2.2),
            TunableParam("lookback", default=90.0, low=45.0, high=150.0),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        az_entry = params.get("alpha_z_entry", 1.8)
        az_confirm = params.get("alpha_z_confirm", 1.5)
        lb = int(params.get("lookback", 90.0))
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Returns
        stock_ret = np.zeros(n, dtype=np.float64)
        idx_ret = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                stock_ret[i] = (close[i] - close[i-1]) / close[i-1]
            if index_close[i-1] > 0:
                idx_ret[i] = (index_close[i] - index_close[i-1]) / index_close[i-1]

        # Rolling beta (120 bars)
        beta_lb = 120
        beta = np.ones(n, dtype=np.float64)
        for i in range(beta_lb, n):
            sr = stock_ret[i - beta_lb:i + 1]
            ir = idx_ret[i - beta_lb:i + 1]
            cov = np.mean(sr * ir) - np.mean(sr) * np.mean(ir)
            var = np.mean(ir * ir) - np.mean(ir) ** 2
            if var > 1e-12:
                beta[i] = cov / var

        # Alpha residual cumulative
        residual = stock_ret - beta * idx_ret
        alpha_spread = np.cumsum(residual)

        # Z-score
        alpha_z = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            seg = alpha_spread[i - lb:i + 1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                alpha_z[i] = (alpha_spread[i] - mu) / std

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 565) & (time_mins <= 840)

        # Index not trending > 1% in 30 bars
        idx_calm = np.ones(n, dtype=np.bool_)
        for i in range(30, n):
            if index_close[i-30] > 0:
                if abs((index_close[i] - index_close[i-30]) / index_close[i-30]) > 0.01:
                    idx_calm[i] = False

        # Entry with crossing confirmation
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (alpha_z[i-1] < -az_entry and alpha_z[i] > -az_confirm
                    and vix_ok[i] and time_ok[i] and idx_calm[i]):
                long_entry[i] = True
            if (alpha_z[i-1] > az_entry and alpha_z[i] < az_confirm
                    and vix_ok[i] and time_ok[i] and idx_calm[i]):
                short_entry[i] = True

        signal_exit_long = alpha_z > 0
        signal_exit_short = alpha_z < 0

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
