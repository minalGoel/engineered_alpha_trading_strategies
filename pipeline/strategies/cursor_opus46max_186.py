"""Granger Causality v1 — cursor_opus46max_186

Thesis: Large-cap liquid stocks lead smaller peers in price discovery. When
the leader moves but the follower hasn't yet responded, trade the follower.
Adapted: use index_close as leader proxy, stock as follower. When index
moves significantly (5-bar ROC > 20 bps) and stock hasn't caught up, trade
the convergence.
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
    name = "cursor_opus46max_186"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("leader_roc_period", default=5.0, low=3.0, high=10.0),
            TunableParam("leader_thresh_bps", default=20.0, low=10.0, high=35.0),
            TunableParam("follower_max_bps", default=5.0, low=2.0, high=10.0),
            TunableParam("lag_gap_min_bps", default=15.0, low=8.0, high=25.0),
            TunableParam("lag_gap_close_bps", default=5.0, low=2.0, high=10.0),
            TunableParam("stop_loss_pct", default=0.0020, low=0.0012, high=0.003),
            TunableParam("trailing_activate_pct", default=0.0015, low=0.001, high=0.0025),
            TunableParam("trailing_stop_pct", default=0.0008, low=0.0005, high=0.0015),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        roc_p = int(params.get("leader_roc_period", 5.0))
        leader_th = params.get("leader_thresh_bps", 20.0)
        follower_max = params.get("follower_max_bps", 5.0)
        gap_min = params.get("lag_gap_min_bps", 15.0)
        gap_close = params.get("lag_gap_close_bps", 5.0)
        stop_pct = params.get("stop_loss_pct", 0.0020)
        trail_act = params.get("trailing_activate_pct", 0.0015)
        trail_pct = params.get("trailing_stop_pct", 0.0008)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Leader (index) and follower (stock) returns
        leader_ret = np.zeros(n, dtype=np.float64)
        follower_ret = np.zeros(n, dtype=np.float64)
        for i in range(roc_p, n):
            if index_close[i-roc_p] > 0:
                leader_ret[i] = (index_close[i] - index_close[i-roc_p]) / index_close[i-roc_p] * 10000
            if close[i-roc_p] > 0:
                follower_ret[i] = (close[i] - close[i-roc_p]) / close[i-roc_p] * 10000

        lag_gap = leader_ret - follower_ret

        # Confirmation: leader move sustained
        leader_sustained_up = np.zeros(n, dtype=np.bool_)
        leader_sustained_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            leader_sustained_up[i] = leader_ret[i] > 15.0
            leader_sustained_down[i] = leader_ret[i] < -15.0

        time_ok = (time_mins >= 570) & (time_mins <= 910)

        long_entry = (
            (leader_ret > leader_th)
            & (follower_ret < follower_max)
            & (lag_gap > gap_min)
            & ((close < vwap) | (np.abs(close - vwap) / np.where(vwap > 0, vwap, 1.0) * 10000 < 5.0))
            & leader_sustained_up
            & time_ok
        )
        short_entry = (
            (leader_ret < -leader_th)
            & (follower_ret > -follower_max)
            & (lag_gap < -gap_min)
            & ((close > vwap) | (np.abs(close - vwap) / np.where(vwap > 0, vwap, 1.0) * 10000 < 5.0))
            & leader_sustained_down
            & time_ok
        )

        # Signal exit: lag gap closes or leader reverses
        sig_exit_long = (lag_gap < gap_close) | (leader_ret < 0)
        sig_exit_short = (lag_gap > -gap_close) | (leader_ret > 0)

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            trailing_activate_pct=trail_act,
            trailing_stop_pct=trail_pct,
            time_stop_bars=30,
        )
