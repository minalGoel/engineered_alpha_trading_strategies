"""Cointegration Residual v1 — cursor_opus46max_185

Thesis: Cointegrated stock pairs share long-run equilibrium. When spread
deviates > 2 sigma, trade the mean-reversion. Adapted for single-instrument
backtesting: stock vs index_close spread z-score.
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
    name = "cursor_opus46max_185"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("spread_window", default=60.0, low=40.0, high=120.0),
            TunableParam("zscore_entry", default=2.0, low=1.5, high=3.0),
            TunableParam("zscore_exit", default=0.3, low=0.1, high=0.8),
            TunableParam("zscore_stop", default=3.5, low=2.5, high=4.5),
            TunableParam("not_expanding_bars", default=10.0, low=5.0, high=15.0),
            TunableParam("confirm_lookback", default=2.0, low=1.0, high=4.0),
            TunableParam("stop_loss_pct", default=0.0035, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        sp_win = int(params.get("spread_window", 60.0))
        zs_entry = params.get("zscore_entry", 2.0)
        zs_exit = params.get("zscore_exit", 0.3)
        zs_stop = params.get("zscore_stop", 3.5)
        not_exp_bars = int(params.get("not_expanding_bars", 10.0))
        confirm_lb = int(params.get("confirm_lookback", 2.0))
        stop_pct = params.get("stop_loss_pct", 0.0035)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Log spread
        log_close = np.log(np.clip(close, 1e-8, None))
        log_index = np.log(np.clip(index_close, 1e-8, None))
        spread = log_close - log_index

        # Z-score of spread
        zscore = np.zeros(n, dtype=np.float64)
        for i in range(sp_win, n):
            seg = spread[i-sp_win:i+1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                zscore[i] = (spread[i] - mu) / std

        # Not expanding: not at min/max over last not_exp_bars
        not_exp_long = np.ones(n, dtype=np.bool_)
        not_exp_short = np.ones(n, dtype=np.bool_)
        for i in range(not_exp_bars, n):
            if zscore[i] < np.min(zscore[i-not_exp_bars:i]):
                not_exp_long[i] = False
            if zscore[i] > np.max(zscore[i-not_exp_bars:i]):
                not_exp_short[i] = False

        # Confirmation: z-score turning
        confirm_long = np.zeros(n, dtype=np.bool_)
        confirm_short = np.zeros(n, dtype=np.bool_)
        for i in range(confirm_lb, n):
            confirm_long[i] = zscore[i] > zscore[i-confirm_lb]
            confirm_short[i] = zscore[i] < zscore[i-confirm_lb]

        time_ok = (time_mins >= 570) & (time_mins <= 910)

        long_entry = (
            (zscore < -zs_entry)
            & not_exp_long
            & confirm_long
            & time_ok
        )
        short_entry = (
            (zscore > zs_entry)
            & not_exp_short
            & confirm_short
            & time_ok
        )

        # Signal exit: z-score reverts near zero or worsens beyond stop
        sig_exit_long = (np.abs(zscore) < zs_exit) | (zscore < -zs_stop)
        sig_exit_short = (np.abs(zscore) < zs_exit) | (zscore > zs_stop)

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=90,
        )
