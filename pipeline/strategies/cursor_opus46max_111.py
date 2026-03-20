"""Sector Momentum Rotation v1 — cursor_opus46max_111

Thesis: Strongest performing sector in first hour continues through midday,
but reverts in last hour.  Use index_close as sector proxy.  Momentum phase
before 13:30, reversion phase after.
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
    name = "cursor_opus46max_111"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 920     # 15:20
    max_trades_per_day = 6
    assumptions = ["Sector rotation proxied via stock-vs-index relative strength"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rel_strength_thresh", default=15.0, low=8.0, high=30.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.0015, low=0.0008, high=0.003),
            TunableParam("target_pct", default=0.003, low=0.0015, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rel_thresh = params.get("rel_strength_thresh", 15.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.0015)
        target_pct = params.get("target_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Cumulative return from session start for stock and index
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

        # Relative strength = stock return - index return (bps)
        rel_strength = stock_ret - index_ret

        # Rotation signal: momentum before 810 (13:30), reversion after
        is_momentum = time_mins < 810
        is_reversion = time_mins >= 810

        vix_ok = vix < vix_max

        # Momentum phase: long strong stocks, short weak ones
        mom_long = is_momentum & (rel_strength > rel_thresh) & (close > vwap) & vix_ok
        mom_short = is_momentum & (rel_strength < -rel_thresh) & (close < vwap) & vix_ok

        # Reversion phase: short previous winners, long previous losers
        rev_long = is_reversion & (rel_strength < -rel_thresh) & vix_ok
        rev_short = is_reversion & (rel_strength > rel_thresh) & vix_ok

        long_entry = mom_long | rev_long
        short_entry = mom_short | rev_short

        # Signal exit: relative strength changes significantly
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # Phase transition triggers exit
            if is_momentum[i-1] and is_reversion[i]:
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
            target_pct=target_pct,
            trailing_stop_pct=0.001,
            trailing_activate_pct=0.002,
            time_stop_bars=120,
        )
