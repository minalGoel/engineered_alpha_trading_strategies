"""Relative Reversion v1 — cursor_opus46max_018

Thesis: Within a sector, the intraday return spread between strongest and
weakest stock reverts. We approximate sector median using index_close as
proxy. Trade when relative z-score is extreme and starts narrowing.
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
    name = "cursor_opus46max_018"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rel_zscore_thresh", default=2.0, low=1.5, high=3.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
            TunableParam("trailing_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("trailing_activate", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rz_thresh = params.get("rel_zscore_thresh", 2.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        trail_pct = params.get("trailing_pct", 0.002)
        trail_act = params.get("trailing_activate", 0.003)

        n = len(df)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # ── Intraday returns for stock and index ──
        unique_days = np.unique(day_id)
        stock_ret = np.zeros(n, dtype=np.float64)
        idx_ret = np.zeros(n, dtype=np.float64)
        bar_from_open = np.zeros(n, dtype=np.int32)
        for d in unique_days:
            mask = day_id == d
            idxs = np.where(mask)[0]
            s_open = opn[idxs[0]] if opn[idxs[0]] > 0 else 1.0
            i_open = index_close[idxs[0]] if index_close[idxs[0]] > 0 else 1.0
            for j, gi in enumerate(idxs):
                stock_ret[gi] = (close[gi] - s_open) / s_open * 100.0
                idx_ret[gi] = (index_close[gi] - i_open) / i_open * 100.0
                bar_from_open[gi] = j

        # ── Relative return ──
        rel_ret = stock_ret - idx_ret

        # ── Relative z-score (rolling 60-bar std) ──
        rel_std = np.zeros(n, dtype=np.float64)
        for i in range(59, n):
            rel_std[i] = np.std(rel_ret[i - 59:i + 1])
        rel_std = np.clip(rel_std, 1e-10, None)

        rel_zscore = np.zeros(n, dtype=np.float64)
        for i in range(59, n):
            rel_zscore[i] = rel_ret[i] / rel_std[i]

        # ── Confirmation: z-score narrowing ──
        narrowing = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            narrowing[i] = abs(rel_zscore[i]) < abs(rel_zscore[i - 1])

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.8 * avg_vol

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Filters ──
        time_ok = (time_mins >= 600) & (time_mins <= 870)
        bars_ok = bar_from_open > 45

        # ── Entries ──
        long_entry = (
            (rel_zscore < -rz_thresh)
            & narrowing
            & vol_ok & time_ok & bars_ok
        )
        short_entry = (
            (rel_zscore > rz_thresh)
            & narrowing
            & vol_ok & time_ok & bars_ok
        )

        # ── Signal exit: z-score crosses zero or deepens to 3 ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if rel_zscore[i] >= 0 and rel_zscore[i - 1] < 0:
                sig_exit_long[i] = True
            if rel_zscore[i] < -(rz_thresh + 1.0):
                sig_exit_long[i] = True
            if rel_zscore[i] <= 0 and rel_zscore[i - 1] > 0:
                sig_exit_short[i] = True
            if rel_zscore[i] > (rz_thresh + 1.0):
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=90,
        )
