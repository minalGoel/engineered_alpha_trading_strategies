"""Sector Rotation Stat Arb v1 — cursor_opus46max_056

Thesis: Sector return spread dislocations after macro shocks mean-revert.
Adapted: stock-vs-index return spread z-score, macro shock filter.
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
    name = "cursor_opus46max_056"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", default=2.0, low=1.5, high=3.0),
            TunableParam("spread_lookback", default=90.0, low=45.0, high=150.0),
            TunableParam("vix_min", default=12.0, low=8.0, high=16.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_entry = params.get("zscore_entry", 2.0)
        lb = int(params.get("spread_lookback", 90.0))
        vix_min = params.get("vix_min", 12.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_ids = df["day_id"].to_numpy()

        # Return spread cumulative
        stock_ret = np.zeros(n, dtype=np.float64)
        idx_ret = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                stock_ret[i] = (close[i] - close[i-1]) / close[i-1]
            if index_close[i-1] > 0:
                idx_ret[i] = (index_close[i] - index_close[i-1]) / index_close[i-1]

        spread = np.cumsum(stock_ret - idx_ret)

        # Z-score
        zscore = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            seg = spread[i - lb:i + 1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                zscore[i] = (spread[i] - mu) / std

        # Macro shock: abs(index 30-bar return) > 0.5%
        macro_shock = np.zeros(n, dtype=np.bool_)
        for i in range(30, n):
            if index_close[i-30] > 0:
                idx_chg = abs((index_close[i] - index_close[i-30]) / index_close[i-30])
                if idx_chg > 0.005:
                    macro_shock[i] = True

        vix_ok = (vix >= vix_min) & (vix <= vix_max)
        time_ok = (time_mins >= 585) & (time_mins <= 840)  # 09:45 - 14:00

        # 2-bar uptick/downtick confirmation
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            if (zscore[i] < -zs_entry and macro_shock[i] and vix_ok[i] and time_ok[i]
                    and zscore[i] > zscore[i-1] and zscore[i-1] > zscore[i-2]):
                long_entry[i] = True
            if (zscore[i] > zs_entry and macro_shock[i] and vix_ok[i] and time_ok[i]
                    and zscore[i] < zscore[i-1] and zscore[i-1] < zscore[i-2]):
                short_entry[i] = True

        # Exit: z crosses 0.5 (partial reversion)
        signal_exit_long = zscore > 0.5
        signal_exit_short = zscore < -0.5

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=120,
        )
