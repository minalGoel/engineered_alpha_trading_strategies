"""Index Momentum Stock Selection — Opus_49

Thesis: When the index has a strong move from open (> 0.5% by bar 45),
stocks aligned with the index direction and above/below VWAP tend to
continue. Enter in same direction; exit when index return crosses zero.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "opus_index_momentum_stock_selection_v1"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("index_ret_thresh", default=0.005, low=0.003, high=0.01),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_stop_pct", default=0.0025, low=0.001, high=0.004),
            TunableParam("trailing_activate_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        idx_thresh = params.get("index_ret_thresh", 0.005)
        vix_max = params.get("vix_max", 22.0)
        tgt_pct = params.get("target_pct", 0.005)
        stp_pct = params.get("stop_loss_pct", 0.003)
        trail_pct = params.get("trailing_stop_pct", 0.0025)
        trail_act = params.get("trailing_activate_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        atr = _compute_atr(high, low, close, 20)

        # ── Index return from open per day ──
        index_ret = np.zeros(n, dtype=np.float64)
        prev_day = -1
        idx_open = 0.0

        for i in range(n):
            if day_id[i] != prev_day:
                idx_open = index_close[i]
                prev_day = day_id[i]
            if idx_open > 0:
                index_ret[i] = (index_close[i] - idx_open) / idx_open
            else:
                index_ret[i] = 0.0

        vix_ok = vix < vix_max
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Long: index_ret > thresh AND close > VWAP AND VIX < max
        long_entry = ((index_ret > idx_thresh) & (close > vwap) & vix_ok & time_ok)

        # Short: index_ret < -thresh AND close < VWAP
        short_entry = ((index_ret < -idx_thresh) & (close < vwap) & vix_ok & time_ok)

        # Signal exit: index_ret crosses 0
        exit_long = np.zeros(n, dtype=np.bool_)
        exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if day_id[i] == day_id[i - 1]:
                if index_ret[i - 1] > 0 and index_ret[i] <= 0:
                    exit_long[i] = True
                if index_ret[i - 1] < 0 and index_ret[i] >= 0:
                    exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=exit_long,
            signal_exit_short=exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stp_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=270,
        )
