"""Risk Parity Signal v1 — cursor_opus46max_158

Thesis: Risk parity (inverse-ATR weighting) as a signal generator. When
a stock's 30-bar return diverges from its VWAP positioning and ATR is
relatively low, enter for mean-reversion. Uses ATR-weighted spread concept
simplified to single-stock application.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "cursor_opus46max_158"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("spread_thresh_bps", default=15.0, low=8.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.0035, low=0.002, high=0.005),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        spread_thresh = params.get("spread_thresh_bps", 15.0)
        stop_pct = params.get("stop_loss_pct", 0.0035)
        target_pct = params.get("target_pct", 0.004)
        vix_max = params.get("vix_max", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
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

        atr = _compute_atr(high, low, close, 14)

        # Stock 30-bar return (bps)
        stock_ret30 = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            if close[i - 30] > 0:
                stock_ret30[i] = (close[i] / close[i - 30] - 1.0) * 10000.0

        # Index 30-bar return (bps)
        idx_ret30 = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            if idx_close[i - 30] > 0:
                idx_ret30[i] = (idx_close[i] / idx_close[i - 30] - 1.0) * 10000.0

        # RP spread: stock return vs index return
        rp_spread = stock_ret30 - idx_ret30

        # Spread turning: compare to previous bar
        spread_turning_up = np.zeros(n, dtype=np.bool_)
        spread_turning_dn = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if rp_spread[i] > rp_spread[i - 1]:
                spread_turning_up[i] = True
            if rp_spread[i] < rp_spread[i - 1]:
                spread_turning_dn[i] = True

        # VWAP filters
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        idx_above_vwap = idx_close > 0  # simplified since no index VWAP

        time_ok = (time_mins >= 570) & (time_mins <= 910)
        vix_ok = vix < vix_max

        # Long: stock underperforming index (rp_spread < -thresh), spread turning
        long_entry = (rp_spread < -spread_thresh) & spread_turning_up & \
                     (close < vwap) & vix_ok & time_ok

        # Short: stock outperforming index (rp_spread > thresh), spread turning
        short_entry = (rp_spread > spread_thresh) & spread_turning_dn & \
                      (close > vwap) & vix_ok & time_ok

        # Signal exit: spread reverts to zero
        sig_exit_long = rp_spread >= 0
        sig_exit_short = rp_spread <= 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=90,
        )
