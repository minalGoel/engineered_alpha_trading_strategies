"""Index Arb NIFTY Components v1 — cursor_opus46max_054

Thesis: Synthetic index from top components diverges from futures-implied level.
Adapted: stock-vs-index basis z-score mean-reversion with uptick/downtick confirm.
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
    name = "cursor_opus46max_054"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 920     # 15:20
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", default=2.0, low=1.5, high=3.0),
            TunableParam("basis_lookback", default=60.0, low=30.0, high=120.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_entry = params.get("zscore_entry", 2.0)
        lb = int(params.get("basis_lookback", 60.0))
        stop_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Basis: stock price relative to index
        basis = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if index_close[i] > 1e-8:
                basis[i] = close[i] / index_close[i]

        # Z-score of basis
        zscore = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            seg = basis[i - lb:i + 1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                zscore[i] = (basis[i] - mu) / std

        time_ok = (time_mins >= 565) & (time_mins <= 900)

        # Uptick / downtick confirmation
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if zscore[i-1] < -zs_entry and zscore[i] > zscore[i-1] and time_ok[i]:
                long_entry[i] = True
            if zscore[i-1] > zs_entry and zscore[i] < zscore[i-1] and time_ok[i]:
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
            time_stop_bars=30,
        )
