"""Relative Strength Momentum — Opus_12

Thesis: Stocks with the strongest first-hour returns that outperform
the index tend to continue outperforming. Compare stock's 60-bar return
against the index return; enter if the stock leads and is above VWAP.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "opus_relative_strength_momentum_v1"
    is_long_only = False
    session_start = 615   # 10:15 (after first hour)
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("min_return_pct", default=0.005, low=0.003, high=0.01),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_stop_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_activate_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        min_ret = params.get("min_return_pct", 0.005)
        target_pct = params.get("target_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.003)
        trailing_pct = params.get("trailing_stop_pct", 0.003)
        trailing_act = params.get("trailing_activate_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── 60-bar returns for stock and index ──
        lookback = 60
        stock_ret_60 = np.zeros(n, dtype=np.float64)
        index_ret_60 = np.zeros(n, dtype=np.float64)

        for i in range(lookback, n):
            prev_close = close[i - lookback]
            if prev_close > 1e-10:
                stock_ret_60[i] = (close[i] - prev_close) / prev_close
            prev_idx = index_close[i - lookback]
            if prev_idx > 1e-10:
                index_ret_60[i] = (index_close[i] - prev_idx) / prev_idx

        # ── Filters ──
        time_ok = (time_mins >= 615) & (time_mins <= 870)

        # ── Long: strong positive return, outperforming index, above VWAP ──
        long_entry = ((stock_ret_60 > min_ret)
                      & (close > vwap)
                      & (stock_ret_60 > index_ret_60)
                      & time_ok)

        # ── Short: strong negative return, underperforming index, below VWAP ──
        short_entry = ((stock_ret_60 < -min_ret)
                       & (close < vwap)
                       & (stock_ret_60 < index_ret_60)
                       & time_ok)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=trailing_pct,
            trailing_activate_pct=trailing_act,
            time_stop_bars=200,
        )
