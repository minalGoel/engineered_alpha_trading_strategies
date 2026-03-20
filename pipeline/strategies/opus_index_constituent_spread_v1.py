# AUDIT FIX: Added time_ok filter to long_entry and short_entry (session_start=570 to session_end=885)
"""Index Constituent Beta-Adjusted Spread — Opus_18

Thesis: After adjusting for beta, the residual return of a stock vs its
index mean-reverts.  Extreme zscores on the residual signal temporary
dislocations.  Filter for low-VIX and calm index environments.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _rolling_zscore(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling zscore."""
    n = len(arr)
    zs = np.zeros(n, dtype=np.float64)
    for i in range(period - 1, n):
        window = arr[i - period + 1: i + 1]
        mu = np.mean(window)
        sigma = np.std(window)
        if sigma < 1e-12:
            zs[i] = 0.0
        else:
            zs[i] = (arr[i] - mu) / sigma
    return zs


class Strategy(BaseStrategy):
    name = "opus_index_constituent_spread_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", default=2.0, low=1.5, high=3.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("index_ret_max", default=0.01, low=0.005, high=0.02),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_entry = params.get("zscore_entry", 2.0)
        vix_max = params.get("vix_max", 22.0)
        idx_ret_max = params.get("index_ret_max", 0.01)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── 60-bar returns ──
        lookback = 60
        stock_ret = np.zeros(n, dtype=np.float64)
        index_ret = np.zeros(n, dtype=np.float64)
        for i in range(lookback, n):
            if close[i - lookback] > 0:
                stock_ret[i] = (close[i] - close[i - lookback]) / close[i - lookback]
            if index_close[i - lookback] > 0:
                index_ret[i] = (index_close[i] - index_close[i - lookback]) / index_close[i - lookback]

        # ── Rolling beta (60-bar) ──
        beta = np.ones(n, dtype=np.float64)
        for i in range(lookback - 1, n):
            sr = stock_ret[i - lookback + 1: i + 1]
            ir = index_ret[i - lookback + 1: i + 1]
            ir_mean = np.mean(ir)
            sr_mean = np.mean(sr)
            cov = np.mean((ir - ir_mean) * (sr - sr_mean))
            var = np.mean((ir - ir_mean) ** 2)
            if var > 1e-12:
                beta[i] = cov / var
            else:
                beta[i] = 1.0

        # ── Residual return ──
        expected_ret = beta * index_ret
        residual = stock_ret - expected_ret

        # ── Zscore of residual over 120 bars ──
        zscore = _rolling_zscore(residual, 120)

        vix_ok = vix < vix_max
        index_calm = np.abs(index_ret) < idx_ret_max
        atr14 = _compute_atr(high, low, close, 14)
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        long_entry = (zscore < -zs_entry) & vix_ok & index_calm & time_ok
        short_entry = (zscore > zs_entry) & vix_ok & index_calm & time_ok

        # ── Signal exit: zscore crosses zero ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if zscore[i - 1] < 0.0 and zscore[i] >= 0.0:
                sig_exit_long[i] = True
            if zscore[i - 1] > 0.0 and zscore[i] <= 0.0:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=90,
        )
