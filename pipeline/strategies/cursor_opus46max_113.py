"""Large vs Mid Cap Rotation v1 — cursor_opus46max_113

Thesis: Intraday return spread between large-cap and mid-cap reflects
shifting risk appetite.  When mid-caps lag large-caps (risk-off), expect
afternoon catch-up.  Use stock-vs-index relative performance as proxy.
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
    name = "cursor_opus46max_113"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 920     # 15:20
    max_trades_per_day = 2
    assumptions = ["Large/mid-cap spread proxied via stock-vs-index return divergence"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=1.5, low=1.0, high=2.5),
            TunableParam("zscore_confirm", default=1.3, low=0.8, high=2.0),
            TunableParam("vix_max", default=20.0, low=14.0, high=25.0),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 1.5)
        zs_confirm = params.get("zscore_confirm", 1.3)
        vix_max = params.get("vix_max", 20.0)
        stop_pct = params.get("stop_loss_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Stock vs index cumulative return spread
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

        spread = stock_ret - index_ret

        # Z-score over 60 bars
        lookback = 60
        spread_zs = np.zeros(n, dtype=np.float64)
        for i in range(lookback - 1, n):
            window = spread[i - lookback + 1: i + 1]
            mu = np.mean(window)
            std = np.std(window)
            if std > 1e-10:
                spread_zs[i] = (spread[i] - mu) / std

        vix_ok = vix < vix_max
        # Only trade after 11:00 (660 min)
        time_ok = (time_mins >= 660) & (time_mins <= 840)

        # Long: spread z-score > thresh (large-cap outperforming, expect stock catch-up)
        # Short: spread z-score < -thresh
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (spread_zs[i-1] >= zs_thresh and spread_zs[i] < zs_confirm and
                    vix_ok[i] and time_ok[i]):
                long_entry[i] = True
            if (spread_zs[i-1] <= -zs_thresh and spread_zs[i] > -zs_confirm and
                    vix_ok[i] and time_ok[i]):
                short_entry[i] = True

        # Signal exit: spread z-score near zero or VIX spike
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if abs(spread_zs[i]) < 0.3:
                sig_exit_long[i] = True
                sig_exit_short[i] = True
            if vix[i] > 22:
                sig_exit_long[i] = True
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
            time_stop_bars=120,
        )
