"""PCA Factor Signal v1 — cursor_opus46max_184

Thesis: PCA on 1-min returns extracts main market/sector factors. Stocks with
extreme residuals (unexplained by top PCs) have idiosyncratic dislocations
that revert. Adapted: use stock vs index_close residual z-score as proxy
for factor-adjusted mispricing.
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
    name = "cursor_opus46max_184"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("residual_window", default=60.0, low=40.0, high=90.0),
            TunableParam("zscore_entry", default=2.0, low=1.5, high=3.0),
            TunableParam("zscore_exit", default=0.5, low=0.2, high=1.0),
            TunableParam("zscore_stop", default=3.5, low=2.5, high=4.5),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        res_win = int(params.get("residual_window", 60.0))
        zs_entry = params.get("zscore_entry", 2.0)
        zs_exit = params.get("zscore_exit", 0.5)
        zs_stop = params.get("zscore_stop", 3.5)
        stop_pct = params.get("stop_loss_pct", 0.003)

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

        # Rolling beta and residual
        residual = np.zeros(n, dtype=np.float64)
        residual_zscore = np.zeros(n, dtype=np.float64)
        for i in range(res_win, n):
            sr = stock_ret[i-res_win+1:i+1]
            ir = index_ret[i-res_win+1:i+1]
            # OLS beta
            ir_m = ir - np.mean(ir)
            sr_m = sr - np.mean(sr)
            denom = np.sum(ir_m ** 2)
            if denom > 1e-15:
                beta = np.sum(ir_m * sr_m) / denom
            else:
                beta = 1.0
            # Current bar residual
            residual[i] = stock_ret[i] - beta * index_ret[i]
            # Rolling residual z-score
            res_hist = sr - beta * ir
            mu = np.mean(res_hist)
            std = np.std(res_hist)
            if std > 1e-10:
                residual_zscore[i] = (residual[i] - mu) / std

        # Confirmation: residual z-score improving
        improving_long = np.zeros(n, dtype=np.bool_)
        improving_short = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            improving_long[i] = residual_zscore[i] > residual_zscore[i-2]
            improving_short[i] = residual_zscore[i] < residual_zscore[i-2]

        time_ok = (time_mins >= 615) & (time_mins <= 910)

        long_entry = (
            (residual_zscore < -zs_entry)
            & (close < vwap)
            & improving_long
            & time_ok
        )
        short_entry = (
            (residual_zscore > zs_entry)
            & (close > vwap)
            & improving_short
            & time_ok
        )

        # Signal exit: residual z-score reverts near zero
        sig_exit_long = (residual_zscore > -zs_exit) | (residual_zscore < -zs_stop)
        sig_exit_short = (residual_zscore < zs_exit) | (residual_zscore > zs_stop)

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=60,
        )
