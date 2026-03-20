"""Sector Rotation (AM Momentum / PM Mean Reversion) — Opus_19

Thesis: Stocks that outperform their index in the first hour tend to
continue (AM momentum), while afternoon underperformers tend to revert
(PM mean reversion).  Two distinct time windows with different logic.
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


class Strategy(BaseStrategy):
    name = "opus_sector_rotation_v1"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 915     # 15:15
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("outperform_pct", default=0.003, low=0.001, high=0.006),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.007),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        outperf = params.get("outperform_pct", 0.003)
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_pct = params.get("target_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
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

        relative = stock_ret - index_ret

        atr14 = _compute_atr(high, low, close, 14)

        # ── AM Momentum: 10:15–12:00 (615–720) ──
        am_window = (time_mins >= 615) & (time_mins <= 720)
        am_long = am_window & (relative > outperf)
        am_short = am_window & (relative < -outperf)

        # ── PM Mean Reversion: 13:00–15:15 (780–915) ──
        pm_window = (time_mins >= 780) & (time_mins <= 915)
        pm_long = pm_window & (relative < -outperf)   # underperformer reverts up
        pm_short = pm_window & (relative > outperf)    # overperformer reverts down

        long_entry = am_long | pm_long
        short_entry = am_short | pm_short

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            breakeven_pct=0.002,
            time_stop_bars=60,
        )
