"""Correlation Breakdown Arb v1 — cursor_opus46max_059

Thesis: Correlated stocks experience intraday correlation breakdowns that snap back.
Adapted: rolling stock-vs-index correlation drop triggers spread reversion entry.
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
    name = "cursor_opus46max_059"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("corr_drop_thresh", default=0.4, low=0.2, high=0.6),
            TunableParam("spread_z_thresh", default=1.5, low=1.0, high=2.5),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        corr_drop_thresh = params.get("corr_drop_thresh", 0.4)
        spread_z_thresh = params.get("spread_z_thresh", 1.5)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Returns
        stock_ret = np.zeros(n, dtype=np.float64)
        idx_ret = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                stock_ret[i] = (close[i] - close[i-1]) / close[i-1]
            if index_close[i-1] > 0:
                idx_ret[i] = (index_close[i] - index_close[i-1]) / index_close[i-1]

        # Rolling correlations
        corr_30 = np.zeros(n, dtype=np.float64)
        corr_120 = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            sr = stock_ret[i-29:i+1]
            ir = idx_ret[i-29:i+1]
            sr_std = np.std(sr)
            ir_std = np.std(ir)
            if sr_std > 1e-10 and ir_std > 1e-10:
                corr_30[i] = np.corrcoef(sr, ir)[0, 1]
        for i in range(120, n):
            sr = stock_ret[i-119:i+1]
            ir = idx_ret[i-119:i+1]
            sr_std = np.std(sr)
            ir_std = np.std(ir)
            if sr_std > 1e-10 and ir_std > 1e-10:
                corr_120[i] = np.corrcoef(sr, ir)[0, 1]

        corr_30 = np.nan_to_num(corr_30, nan=0.0)
        corr_120 = np.nan_to_num(corr_120, nan=0.0)
        corr_drop = corr_120 - corr_30

        # Cumulative relative spread and z-score
        rel_spread = np.cumsum(stock_ret - idx_ret)
        spread_z = np.zeros(n, dtype=np.float64)
        for i in range(60, n):
            seg = rel_spread[i-59:i+1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                spread_z[i] = (rel_spread[i] - mu) / std

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 570) & (time_mins <= 840)

        # 3-bar corr recovery confirmation
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(3, n):
            corr_recovering = (corr_30[i] > corr_30[i-1] and corr_30[i-1] > corr_30[i-2]
                               and corr_30[i-2] > corr_30[i-3])
            if (corr_drop[i] > corr_drop_thresh and spread_z[i] < -spread_z_thresh
                    and corr_recovering and vix_ok[i] and time_ok[i]):
                long_entry[i] = True
            if (corr_drop[i] > corr_drop_thresh and spread_z[i] > spread_z_thresh
                    and corr_recovering and vix_ok[i] and time_ok[i]):
                short_entry[i] = True

        signal_exit_long = spread_z > 0
        signal_exit_short = spread_z < 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=60,
        )
