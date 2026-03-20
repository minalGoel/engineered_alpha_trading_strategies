"""Banking vs IT Rotation v1 — cursor_opus46max_112

Thesis: BANK NIFTY and NIFTY IT exhibit negative intraday correlation.
When one significantly outperforms the other, mean reversion occurs.
Proxy spread using stock-vs-index relative performance and VIX filter.
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
    name = "cursor_opus46max_112"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 920     # 15:20
    max_trades_per_day = 3
    assumptions = ["Banking-IT spread proxied via stock-vs-index cumulative return z-score"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=2.0, low=1.5, high=3.0),
            TunableParam("zscore_confirm", default=1.8, low=1.2, high=2.5),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 2.0)
        zs_confirm = params.get("zscore_confirm", 1.8)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_id = df["day_id"].to_numpy()

        # Cumulative return spread (stock vs index) as proxy for sector rotation
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

        # Spread: stock - index cumulative return
        spread = stock_ret - index_ret

        # Z-score of spread over 60 bars
        lookback = 60
        spread_zs = np.zeros(n, dtype=np.float64)
        for i in range(lookback - 1, n):
            window = spread[i - lookback + 1: i + 1]
            mu = np.mean(window)
            std = np.std(window)
            if std > 1e-10:
                spread_zs[i] = (spread[i] - mu) / std

        vix_ok = vix < vix_max

        # Long: spread z-score < -thresh (stock underperforming, expect catch-up)
        # Confirmation: z-score starts rising
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (spread_zs[i] > -zs_confirm and spread_zs[i-1] <= -zs_thresh and
                    vix_ok[i]):
                long_entry[i] = True
            if (spread_zs[i] < zs_confirm and spread_zs[i-1] >= zs_thresh and
                    vix_ok[i]):
                short_entry[i] = True

        # Signal exit: spread z-score reverts to 0
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if spread_zs[i] >= 0 and spread_zs[i-1] < 0:
                sig_exit_long[i] = True
            if spread_zs[i] <= 0 and spread_zs[i-1] > 0:
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
            time_stop_bars=90,
        )
