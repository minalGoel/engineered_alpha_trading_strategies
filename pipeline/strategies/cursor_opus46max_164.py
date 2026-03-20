"""Composite Breadth v1 — cursor_opus46max_164

Thesis: Market breadth (advance/decline) divergence from index direction
identifies fragile moves. NIFTY rising with narrow breadth = sell signal;
NIFTY falling with broad breadth = buy signal.
Simplified: use index ROC as proxy, with stock-level VWAP/momentum filters.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_ema(arr, period):
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = arr[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_164"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("roc10_thresh_bps", default=10.0, low=5.0, high=20.0),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("trailing_stop_pct", default=0.0015, low=0.001, high=0.003),
            TunableParam("trailing_activate_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        roc10_thresh = params.get("roc10_thresh_bps", 10.0)
        target_pct = params.get("target_pct", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        trail_pct = params.get("trailing_stop_pct", 0.0015)
        trail_act = params.get("trailing_activate_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        idx_close = df["index_close"].to_numpy().astype(np.float64)
        idx_close = np.nan_to_num(idx_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Index EMAs for trend
        idx_ema10 = _compute_ema(idx_close, 10)
        idx_ema30 = _compute_ema(idx_close, 30)
        nifty_trend = np.sign(idx_ema10 - idx_ema30)

        # Stock 5-bar and 10-bar ROC (bps)
        stock_roc5 = np.zeros(n, dtype=np.float64)
        stock_roc10 = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            if close[i - 5] > 0:
                stock_roc5[i] = (close[i] / close[i - 5] - 1.0) * 10000.0
        for i in range(10, n):
            if close[i - 10] > 0:
                stock_roc10[i] = (close[i] / close[i - 10] - 1.0) * 10000.0

        # Index ROC (proxy for breadth signal - stock diverging from index)
        idx_roc10 = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            if idx_close[i - 10] > 0:
                idx_roc10[i] = (idx_close[i] / idx_close[i - 10] - 1.0) * 10000.0

        # Divergence: stock and index moving in opposite directions
        stock_vs_idx = np.sign(stock_roc10) * np.sign(idx_roc10)

        # Breadth divergence proxy increasing for 3 bars
        breadth_up = np.zeros(n, dtype=np.bool_)
        breadth_dn = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            if stock_roc5[i] > stock_roc5[i - 1] > stock_roc5[i - 2]:
                breadth_up[i] = True
            if stock_roc5[i] < stock_roc5[i - 1] < stock_roc5[i - 2]:
                breadth_dn[i] = True

        time_ok = (time_mins >= 565) & (time_mins <= 910)

        # Long: NIFTY falling but stock breadth improving (stock lagging, to catch up)
        long_entry = (nifty_trend < 0) & (close < vwap) & \
                     (stock_roc10 < -roc10_thresh) & breadth_up & time_ok

        # Short: NIFTY rising but stock overbought (leading narrow rally)
        short_entry = (nifty_trend > 0) & (close > vwap) & \
                      (stock_roc10 > roc10_thresh) & breadth_dn & time_ok

        # Signal exit: divergence resolves
        sig_exit_long = stock_vs_idx > 0
        sig_exit_short = stock_vs_idx > 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=75,
        )
