"""IT Sector Correlation Arbitrage — Opus_22

Thesis: The log price ratio of a stock to its index mean-reverts on
intraday timeframes.  Zscore extremes (±1.5) on this ratio signal
temporary mispricings.  Tighter entry than classic pairs.
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
    name = "opus_it_sector_corr_arb_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", default=1.5, low=1.0, high=2.5),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_entry = params.get("zscore_entry", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)

        # ── Log ratio ──
        safe_close = np.clip(close, 1e-8, None)
        safe_index = np.clip(index_close, 1e-8, None)
        log_ratio = np.log(safe_close) - np.log(safe_index)

        # ── Zscore of log ratio over 120 bars ──
        zscore = _rolling_zscore(log_ratio, 120)

        atr14 = _compute_atr(high, low, close, 14)

        long_entry = zscore < -zs_entry
        short_entry = zscore > zs_entry

        # ── Signal exit: zscore crosses zero ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if zscore[i - 1] < 0.0 and zscore[i] >= 0.0:
                sig_exit_long[i] = True
            if zscore[i - 1] > 0.0 and zscore[i] <= 0.0:
                sig_exit_short[i] = True

        # Breakeven approximation: zscore at ±0.5
        # Using breakeven_pct as proxy
        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            breakeven_pct=0.002,
            time_stop_bars=90,
        )
