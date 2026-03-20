"""Gap Sector Leader v1 — cursor_opus46max_068

Thesis: When sector leader gaps strongly, lagging peers catch up.
Adapted: stock under-gapping vs index (leader) with EMA9 cross confirmation.
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


def _ema(data, period):
    n = len(data)
    out = np.zeros(n, dtype=np.float64)
    out[0] = data[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_068"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 720     # 12:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("leader_gap_min", default=0.008, low=0.005, high=0.015),
            TunableParam("gap_spread_min", default=0.005, low=0.003, high=0.01),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        leader_gap_min = params.get("leader_gap_min", 0.008)
        gap_spread_min = params.get("gap_spread_min", 0.005)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        ema9 = _ema(close, 9)
        atr10 = _compute_atr(df["high"].to_numpy().astype(np.float64),
                              df["low"].to_numpy().astype(np.float64),
                              close, 10)

        # Gap: stock and index
        stock_gap = np.zeros(n, dtype=np.float64)
        index_gap = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        prev_close_s = np.nan
        prev_close_i = np.nan
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) == 0:
                continue
            if not np.isnan(prev_close_s) and prev_close_s > 0:
                stock_gap[idx] = (open_[idx[0]] - prev_close_s) / prev_close_s
            if not np.isnan(prev_close_i) and prev_close_i > 0 and index_close[idx[0]] > 0:
                index_gap[idx] = (index_close[idx[0]] - prev_close_i) / prev_close_i
            prev_close_s = close[idx[-1]]
            prev_close_i = index_close[idx[-1]]

        # Gap spread = index gap (leader) - stock gap (lagger)
        gap_spread = np.abs(index_gap) - np.abs(stock_gap)

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 560) & (time_mins <= 600)

        # EMA9 cross confirmation
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # Index gapped up strongly, stock lagging -> buy stock
            if (index_gap[i] > leader_gap_min and gap_spread[i] > gap_spread_min
                    and close[i] > open_[i] and vix_ok[i] and time_ok[i]
                    and close[i-1] <= ema9[i-1] and close[i] > ema9[i]):
                long_entry[i] = True
            # Index gapped down strongly, stock lagging -> sell stock
            if (index_gap[i] < -leader_gap_min and gap_spread[i] > gap_spread_min
                    and close[i] < open_[i] and vix_ok[i] and time_ok[i]
                    and close[i-1] >= ema9[i-1] and close[i] < ema9[i]):
                short_entry[i] = True

        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr10,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=0.5,
            target_pct=0.005,
            breakeven_pct=0.003,
            time_stop_bars=60,
        )
