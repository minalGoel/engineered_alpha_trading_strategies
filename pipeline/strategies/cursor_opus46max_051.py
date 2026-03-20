"""Pairs Trading Bank v1 — cursor_opus46max_051

Thesis: HDFCBANK and ICICIBANK are structurally linked large-cap private banks.
Adapted for single-instrument backtesting: stock vs index_close spread z-score
for mean-reversion entry/exit.
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
    name = "cursor_opus46max_051"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", default=2.0, low=1.2, high=3.0),
            TunableParam("zscore_confirm", default=1.8, low=1.0, high=2.5),
            TunableParam("spread_lookback", default=120.0, low=60.0, high=200.0),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_entry = params.get("zscore_entry", 2.0)
        zs_confirm = params.get("zscore_confirm", 1.8)
        lb = int(params.get("spread_lookback", 120.0))
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Spread: log(stock) - log(index) ratio
        log_close = np.log(np.clip(close, 1e-8, None))
        log_index = np.log(np.clip(index_close, 1e-8, None))
        spread = log_close - log_index

        # Z-score of spread
        zscore = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            seg = spread[i - lb:i + 1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                zscore[i] = (spread[i] - mu) / std

        # z_score not expanding for last 3 bars
        not_expanding = np.ones(n, dtype=np.bool_)
        for i in range(3, n):
            if abs(zscore[i]) > abs(zscore[i-1]) and abs(zscore[i-1]) > abs(zscore[i-2]):
                not_expanding[i] = False

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 560) & (time_mins <= 870)  # 09:20 - 14:30

        # Long: z < -entry, then crosses above -confirm (momentum turning)
        was_below_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if zscore[i-1] < -zs_entry:
                was_below_entry[i] = True
            elif zscore[i-1] >= -zs_confirm:
                was_below_entry[i] = False
            else:
                was_below_entry[i] = was_below_entry[i-1]

        long_entry = was_below_entry & (zscore > -zs_confirm) & not_expanding & vix_ok & time_ok

        # Short: z > entry, then crosses below confirm
        was_above_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if zscore[i-1] > zs_entry:
                was_above_entry[i] = True
            elif zscore[i-1] <= zs_confirm:
                was_above_entry[i] = False
            else:
                was_above_entry[i] = was_above_entry[i-1]

        short_entry = was_above_entry & (zscore < zs_confirm) & not_expanding & vix_ok & time_ok

        # Signal exit: z crosses zero
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
