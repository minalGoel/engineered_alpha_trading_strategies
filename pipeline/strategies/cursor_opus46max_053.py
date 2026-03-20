"""Pairs Trading Auto v1 — cursor_opus46max_053

Thesis: Auto sector pair spread (MARUTI vs M&M) mean-reverts intraday.
Adapted: stock vs index log-spread z-score with reversal candle confirmation.
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
    name = "cursor_opus46max_053"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", default=2.0, low=1.5, high=3.0),
            TunableParam("spread_lookback", default=100.0, low=50.0, high=180.0),
            TunableParam("vix_max", default=20.0, low=14.0, high=26.0),
            TunableParam("stop_loss_pct", default=0.006, low=0.003, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_entry = params.get("zscore_entry", 2.0)
        lb = int(params.get("spread_lookback", 100.0))
        vix_max = params.get("vix_max", 20.0)
        stop_pct = params.get("stop_loss_pct", 0.006)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Spread
        log_close = np.log(np.clip(close, 1e-8, None))
        log_index = np.log(np.clip(index_close, 1e-8, None))
        spread = log_close - log_index

        # Z-score
        zscore = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            seg = spread[i - lb:i + 1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                zscore[i] = (spread[i] - mu) / std

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 565) & (time_mins <= 840)

        # Reversal candle: close > prev close (long) or close < prev close (short)
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (zscore[i] < -zs_entry and close[i] > close[i-1]
                    and vix_ok[i] and time_ok[i]):
                long_entry[i] = True
            if (zscore[i] > zs_entry and close[i] < close[i-1]
                    and vix_ok[i] and time_ok[i]):
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
