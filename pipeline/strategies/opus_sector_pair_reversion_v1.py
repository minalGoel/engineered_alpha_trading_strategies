"""Sector Pair Reversion (Stock vs Index) — Opus_17

Thesis: The log-spread between a stock and its index (with a hedge ratio)
mean-reverts.  Enter when the zscore is extreme (±2) and exit when it
crosses zero.  VIX filter avoids regime-change environments.
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
    """Rolling zscore with NaN safety."""
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
    name = "opus_sector_pair_reversion_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", default=2.0, low=1.5, high=3.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_entry = params.get("zscore_entry", 2.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        # Safely compute log prices
        safe_close = np.clip(close, 1e-8, None)
        safe_index = np.clip(index_close, 1e-8, None)
        log_close = np.log(safe_close)
        log_index = np.log(safe_index)

        # ── Rolling hedge ratio (OLS slope over 120 bars) ──
        hedge = np.ones(n, dtype=np.float64)
        for i in range(119, n):
            x = log_index[i - 119: i + 1]
            y = log_close[i - 119: i + 1]
            x_mean = np.mean(x)
            y_mean = np.mean(y)
            cov = np.mean((x - x_mean) * (y - y_mean))
            var = np.mean((x - x_mean) ** 2)
            if var > 1e-12:
                hedge[i] = cov / var
            else:
                hedge[i] = 1.0

        # ── Spread and zscore ──
        spread = log_close - hedge * log_index
        zscore = _rolling_zscore(spread, 120)

        vix_ok = vix < vix_max
        atr14 = _compute_atr(high, low, close, 14)

        long_entry = (zscore < -zs_entry) & vix_ok
        short_entry = (zscore > zs_entry) & vix_ok

        # ── Signal exit: zscore crosses zero ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if zscore[i - 1] < 0.0 and zscore[i] >= 0.0:
                sig_exit_long[i] = True
            if zscore[i - 1] > 0.0 and zscore[i] <= 0.0:
                sig_exit_short[i] = True

        # Breakeven when zscore crosses ±1
        # Approximated via breakeven_pct — signal when zscore is at ±1
        # (handled by state machine breakeven_pct after small profit)
        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=120,
        )
