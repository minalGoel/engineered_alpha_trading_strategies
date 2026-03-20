# AUDIT FIX: Added session window enforcement — entries were firing outside session_start/session_end
"""Nifty-Bank Lag Arbitrage — gemini_13_of_20

Thesis: Bank Nifty components sometimes lag the broader index. When the index
moves sharply but the stock hasn't caught up, enter expecting convergence.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "nifty_bank_lag_arbitrage"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 920     # 15:20
    max_trades_per_day = 30

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("index_ret_thresh", default=0.3, low=0.1, high=0.6),
            TunableParam("stock_ret_max", default=0.1, low=0.0, high=0.2),
            TunableParam("target_pct", default=0.3, low=0.1, high=0.6),
            TunableParam("stop_pct", default=0.15, low=0.05, high=0.3),
            TunableParam("vix_max", default=20.0, low=12.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        index_ret_thresh = params.get("index_ret_thresh", 0.3)
        stock_ret_max = params.get("stock_ret_max", 0.1)
        target_pct = params.get("target_pct", 0.3)
        stop_pct = params.get("stop_pct", 0.15)
        vix_max = params.get("vix_max", 20.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)

        atr14 = _compute_atr(high, low, close, 14)

        # 5-bar returns (percentage)
        lookback = 5
        index_ret5 = np.zeros(n, dtype=np.float64)
        stock_ret5 = np.zeros(n, dtype=np.float64)
        for i in range(lookback, n):
            if abs(index_close[i - lookback]) > 1e-10:
                index_ret5[i] = (index_close[i] - index_close[i - lookback]) / index_close[i - lookback] * 100.0
            if abs(close[i - lookback]) > 1e-10:
                stock_ret5[i] = (close[i] - close[i - lookback]) / close[i - lookback] * 100.0

        # Session window filter
        time_mins_arr = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins_arr >= self.session_start) & (time_mins_arr <= self.session_end)

        vix_ok = vix < vix_max

        # Long: index up > thresh%, stock lagging (< stock_ret_max%)
        long_entry = (
            (index_ret5 > index_ret_thresh)
            & (stock_ret5 < stock_ret_max)
            & vix_ok
            & in_session
        )

        # Short: index down > thresh%, stock lagging (> -stock_ret_max%)
        short_entry = (
            (index_ret5 < -index_ret_thresh)
            & (stock_ret5 > -stock_ret_max)
            & vix_ok
            & in_session
        )

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct / 100.0,
            stop_loss_pct=stop_pct / 100.0,
            time_stop_bars=15,
        )
