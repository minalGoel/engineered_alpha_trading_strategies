# AUDIT FIX: Add session window filtering to prevent entries outside session_start/session_end
"""Market Breadth Signal v1 — cursor_opus46max_120

Thesis: When NIFTY hits new intraday high but breadth (proxied by stock
vs index relationship) deteriorates, the rally is unsustainable and
reverses within 15-30 minutes.  Vice versa for new lows with strong breadth.
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


def _ema(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = arr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_120"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 920     # 15:20
    max_trades_per_day = 6
    assumptions = ["Market breadth proxied via stock-vs-index divergence pattern"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("breadth_ema_long", default=1.2, low=1.0, high=1.5),
            TunableParam("breadth_ema_short", default=0.8, low=0.5, high=1.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.001, low=0.0005, high=0.002),
            TunableParam("target_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        breadth_long = params.get("breadth_ema_long", 1.2)
        breadth_short = params.get("breadth_ema_short", 0.8)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.001)
        target_pct = params.get("target_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Breadth proxy: advance/decline ratio using stock vs index 5-bar returns
        # If stock advancing while index advancing, breadth=1; else breadth check
        stock_ret5 = np.zeros(n, dtype=np.float64)
        index_ret5 = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            if close[i-5] > 1e-10:
                stock_ret5[i] = close[i] - close[i-5]
            if index_close[i-5] > 1e-10:
                index_ret5[i] = index_close[i] - index_close[i-5]

        # Breadth indicator: are stock and index moving together?
        # >1 means stock doing relatively better, <1 means stock lagging
        breadth_raw = np.ones(n, dtype=np.float64)
        for i in range(5, n):
            if index_ret5[i] != 0:
                breadth_raw[i] = 1.0 + (stock_ret5[i] / max(abs(index_ret5[i]), 1e-10) - 1.0) * 0.5
            else:
                breadth_raw[i] = 1.0
        breadth_raw = np.clip(breadth_raw, 0.1, 3.0)
        breadth_ema = _ema(breadth_raw, 10)

        # Index new high/low tracking (cumulative max/min per day)
        idx_max = np.zeros(n, dtype=np.float64)
        idx_min = np.full(n, 1e10, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                idx_max[i] = index_close[i]
                idx_min[i] = index_close[i]
            else:
                idx_max[i] = max(idx_max[i-1], index_close[i])
                idx_min[i] = min(idx_min[i-1], index_close[i])

        is_new_high = (index_close == idx_max) & (index_close > 0)
        is_new_low = (index_close == idx_min) & (index_close > 0)

        vix_ok = vix < vix_max

        # Divergence: index new low but breadth good -> buy
        # Index new high but breadth bad -> sell
        # Confirmation: first tick in direction
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (is_new_low[i] and breadth_ema[i] > breadth_long and
                    close[i] > close[i-1] and vix_ok[i] and in_session[i]):
                long_entry[i] = True
            if (is_new_high[i] and breadth_ema[i] < breadth_short and
                    close[i] < close[i-1] and vix_ok[i] and in_session[i]):
                short_entry[i] = True

        # Signal exit: breadth realigns with index direction
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if breadth_ema[i] < 1.0 and breadth_ema[i-1] >= 1.0:
                sig_exit_long[i] = True
            if breadth_ema[i] > 1.0 and breadth_ema[i-1] <= 1.0:
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=0.0008,
            trailing_activate_pct=0.0012,
            time_stop_bars=30,
        )
