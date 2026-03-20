"""Cross Listed Arb v1 — cursor_opus46max_060

Thesis: NSE/BSE cross-listed stocks have transient price divergences.
Adapted: stock-vs-index price diff z-score for quick mean-reversion.
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
    name = "cursor_opus46max_060"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 925     # 15:25
    max_trades_per_day = 20

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("diff_thresh_bps", default=5.0, low=3.0, high=12.0),
            TunableParam("diff_z_thresh", default=2.0, low=1.2, high=3.0),
            TunableParam("lookback", default=60.0, low=30.0, high=120.0),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        diff_thresh = params.get("diff_thresh_bps", 5.0)
        diff_z_thresh = params.get("diff_z_thresh", 2.0)
        lb = int(params.get("lookback", 60.0))
        stop_pct = params.get("stop_loss_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)

        # Price diff in bps (stock vs index-normalized price)
        diff_bps = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if index_close[i] > 1e-8:
                diff_bps[i] = (close[i] - index_close[i]) / index_close[i] * 10000.0

        # Z-score
        diff_z = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            seg = diff_bps[i - lb:i + 1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                diff_z[i] = (diff_bps[i] - mu) / std

        # Entry: buy low, sell high on diff
        long_entry = (diff_bps > diff_thresh) & (diff_z > diff_z_thresh)
        short_entry = (diff_bps < -diff_thresh) & (diff_z < -diff_z_thresh)

        signal_exit_long = diff_bps < 0
        signal_exit_short = diff_bps > 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=10,
        )
