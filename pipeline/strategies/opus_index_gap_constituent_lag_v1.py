"""Index Gap Constituent Lag — Opus_32

Thesis: When the index gaps big, lagging constituents catch up. If a stock
under-gaps relative to its beta-adjusted index gap, it tends to rally to
close the shortfall.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "opus_index_gap_constituent_lag_v1"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 600     # 10:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("index_gap_thresh", default=0.005, low=0.003, high=0.01),
            TunableParam("shortfall_thresh", default=0.003, low=0.001, high=0.006),
            TunableParam("target_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        index_gap_thresh = params.get("index_gap_thresh", 0.005)
        shortfall_thresh = params.get("shortfall_thresh", 0.003)
        target_pct = params.get("target_pct", 0.003)
        stop_pct = params.get("stop_loss_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        day_id = df["day_id"].to_numpy().astype(np.int64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Previous day close for stock and index ──
        day_last_close = {}
        day_last_index = {}
        for i in range(n):
            d = day_id[i]
            day_last_close[d] = close[i]
            day_last_index[d] = index_close[i]

        prev_day_close = np.zeros(n, dtype=np.float64)
        prev_day_index = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d = day_id[i]
            if d - 1 in day_last_close:
                prev_day_close[i] = day_last_close[d - 1]
            if d - 1 in day_last_index:
                prev_day_index[i] = day_last_index[d - 1]

        # ── Per-day gaps ──
        day_first_open = {}
        day_first_index = {}
        for i in range(n):
            d = day_id[i]
            if d not in day_first_open:
                day_first_open[d] = open_[i]
                day_first_index[d] = index_close[i]

        stock_gap = np.zeros(n, dtype=np.float64)
        index_gap = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d = day_id[i]
            if prev_day_close[i] > 1e-10:
                stock_gap[i] = (day_first_open[d] - prev_day_close[i]) / prev_day_close[i]
            if prev_day_index[i] > 1e-10:
                index_gap[i] = (day_first_index[d] - prev_day_index[i]) / prev_day_index[i]

        # ── Approximate beta = 1.0 (simplification) ──
        beta = 1.0
        gap_shortfall = stock_gap - beta * index_gap

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Filters ──
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entries ──
        # Long: index gapped up, stock lagged (shortfall negative)
        long_entry = (
            (index_gap > index_gap_thresh)
            & (gap_shortfall < -shortfall_thresh)
            & time_ok
        )
        # Short: index gapped down, stock didn't gap enough (shortfall positive)
        short_entry = (
            (index_gap < -index_gap_thresh)
            & (gap_shortfall > shortfall_thresh)
            & time_ok
        )

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=45,
        )
