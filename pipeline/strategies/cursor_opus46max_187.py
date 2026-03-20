"""Information Ratio v1 — cursor_opus46max_187

Thesis: Rolling 60-bar Information Ratio (alpha/tracking_error vs NIFTY 50)
identifies stocks with best risk-adjusted alpha. Enter stocks with high IR
in their current direction. Adapted for single-stock: enter when stock's
IR exceeds threshold, confirming with VWAP.
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
    name = "cursor_opus46max_187"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ir_window", default=60.0, low=40.0, high=90.0),
            TunableParam("ir_thresh", default=1.0, low=0.5, high=2.0),
            TunableParam("ir_persist_bars", default=10.0, low=5.0, high=15.0),
            TunableParam("ir_stability_max", default=2.0, low=1.0, high=3.0),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("target_pct", default=0.0045, low=0.003, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ir_win = int(params.get("ir_window", 60.0))
        ir_thresh = params.get("ir_thresh", 1.0)
        ir_persist = int(params.get("ir_persist_bars", 10.0))
        ir_stab_max = params.get("ir_stability_max", 2.0)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        target_pct = params.get("target_pct", 0.0045)

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

        # Returns
        stock_ret = np.zeros(n, dtype=np.float64)
        index_ret = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                stock_ret[i] = (close[i] - close[i-1]) / close[i-1]
            if index_close[i-1] > 0:
                index_ret[i] = (index_close[i] - index_close[i-1]) / index_close[i-1]

        # Rolling IR
        info_ratio = np.zeros(n, dtype=np.float64)
        alpha_60 = np.zeros(n, dtype=np.float64)
        for i in range(ir_win, n):
            excess = stock_ret[i-ir_win+1:i+1] - index_ret[i-ir_win+1:i+1]
            alpha = np.mean(excess) * ir_win  # scale
            te = np.std(excess) * np.sqrt(ir_win)
            alpha_60[i] = alpha * 10000  # bps
            if te > 1e-10:
                info_ratio[i] = alpha / te

        # IR stability (std of IR over last 20 bars)
        ir_stability = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            ir_stability[i] = np.std(info_ratio[i-19:i+1])

        # IR persistence
        ir_pos_persist = np.zeros(n, dtype=np.bool_)
        ir_neg_persist = np.zeros(n, dtype=np.bool_)
        for i in range(ir_persist, n):
            ir_pos_persist[i] = all(info_ratio[i-j] > 0.5 for j in range(ir_persist))
            ir_neg_persist[i] = all(info_ratio[i-j] < -0.5 for j in range(ir_persist))

        time_ok = (time_mins >= 615) & (time_mins <= 910)

        long_entry = (
            (info_ratio > ir_thresh)
            & (alpha_60 > 0)
            & (close > vwap)
            & ir_pos_persist
            & (ir_stability < ir_stab_max)
            & time_ok
        )
        short_entry = (
            (info_ratio < -ir_thresh)
            & (alpha_60 < 0)
            & (close < vwap)
            & ir_neg_persist
            & (ir_stability < ir_stab_max)
            & time_ok
        )

        # Signal exit: IR drops below threshold
        sig_exit_long = info_ratio < 0.5
        sig_exit_short = info_ratio > -0.5

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
            trailing_activate_pct=0.0025,
            trailing_stop_pct=0.0015,
            time_stop_bars=75,
        )
