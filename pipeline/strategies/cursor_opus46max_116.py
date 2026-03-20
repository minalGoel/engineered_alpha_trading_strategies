# AUDIT FIX: Add session window filtering to prevent entries outside session_start/session_end
"""Industry Group Reversion v1 — cursor_opus46max_116

Thesis: Within tightly correlated industry sub-groups, individual stock
returns revert to group mean.  Proxy group mean using index, detect
stock deviations, and trade reversion.
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
    name = "cursor_opus46max_116"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 915     # 15:15
    max_trades_per_day = 8
    assumptions = ["Industry group mean proxied using index cumulative return"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("dev_zscore_thresh", default=2.0, low=1.5, high=3.0),
            TunableParam("dev_bps_thresh", default=15.0, low=8.0, high=25.0),
            TunableParam("stop_loss_pct", default=0.0015, low=0.0008, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("dev_zscore_thresh", 2.0)
        bps_thresh = params.get("dev_bps_thresh", 15.0)
        stop_pct = params.get("stop_loss_pct", 0.0015)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Cumulative returns from day start
        stock_ret = np.zeros(n, dtype=np.float64)
        index_ret = np.zeros(n, dtype=np.float64)
        day_open_stock = close[0]
        day_open_index = index_close[0]
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                day_open_stock = close[i]
                day_open_index = index_close[i] if index_close[i] > 1e-10 else 1.0
            if day_open_stock > 1e-10:
                stock_ret[i] = (close[i] - day_open_stock) / day_open_stock * 10000.0
            if day_open_index > 1e-10:
                index_ret[i] = (index_close[i] - day_open_index) / day_open_index * 10000.0

        # Stock deviation from group (index) return
        deviation = stock_ret - index_ret

        # Z-score of deviation over 30 bars
        lookback = 30
        dev_zs = np.zeros(n, dtype=np.float64)
        for i in range(lookback - 1, n):
            window = deviation[i - lookback + 1: i + 1]
            mu = np.mean(window)
            std = np.std(window)
            if std > 1e-10:
                dev_zs[i] = (deviation[i] - mu) / std

        # Volume filter
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.5 * avg_vol

        # Confirmation: z-score starts reverting
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (dev_zs[i-1] <= -zs_thresh and dev_zs[i] > -zs_thresh + 0.3 and
                    deviation[i] < -bps_thresh and vol_ok[i] and in_session[i]):
                long_entry[i] = True
            if (dev_zs[i-1] >= zs_thresh and dev_zs[i] < zs_thresh - 0.3 and
                    deviation[i] > bps_thresh and vol_ok[i] and in_session[i]):
                short_entry[i] = True

        # Signal exit: deviation reverts to 0
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if deviation[i] >= 0 and deviation[i-1] < 0:
                sig_exit_long[i] = True
            if deviation[i] <= 0 and deviation[i-1] > 0:
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
            time_stop_bars=30,
        )
