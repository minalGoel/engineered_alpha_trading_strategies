"""Changepoint Detection (CUSUM) v1 — cursor_opus46max_192

Thesis: CUSUM control charts detect structural breaks in the mean of returns.
A positive break signals a new uptrend regime; a negative break signals
a new downtrend. Enter at the beginning of a new regime for timing advantage
over lagging indicators.
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
    name = "cursor_opus46max_192"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("cusum_k_mult", default=0.5, low=0.3, high=1.0),
            TunableParam("cusum_h_mult", default=4.0, low=2.5, high=6.0),
            TunableParam("mean_window", default=30.0, low=15.0, high=50.0),
            TunableParam("fresh_bars", default=3.0, low=1.0, high=5.0),
            TunableParam("below_thresh_bars", default=20.0, low=10.0, high=30.0),
            TunableParam("roc3_thresh_bps", default=5.0, low=2.0, high=10.0),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("target_pct", default=0.0045, low=0.003, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        k_mult = params.get("cusum_k_mult", 0.5)
        h_mult = params.get("cusum_h_mult", 4.0)
        mean_win = int(params.get("mean_window", 30.0))
        fresh = int(params.get("fresh_bars", 3.0))
        below_bars = int(params.get("below_thresh_bars", 20.0))
        roc3_th = params.get("roc3_thresh_bps", 5.0)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        target_pct = params.get("target_pct", 0.0045)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # 1-min returns
        ret = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                ret[i] = (close[i] - close[i-1]) / close[i-1] * 10000  # bps

        # Rolling mean and std
        rolling_mean = np.zeros(n, dtype=np.float64)
        rolling_std = np.zeros(n, dtype=np.float64)
        for i in range(mean_win, n):
            seg = ret[i-mean_win+1:i+1]
            rolling_mean[i] = np.mean(seg)
            rolling_std[i] = max(np.std(seg), 1e-10)

        # CUSUM
        cusum_pos = np.zeros(n, dtype=np.float64)
        cusum_neg = np.zeros(n, dtype=np.float64)
        cusum_thresh = np.zeros(n, dtype=np.float64)
        day_id = df["day_id"].to_numpy()

        for i in range(mean_win, n):
            if i > 0 and day_id[i] != day_id[i-1]:
                cusum_pos[i] = 0.0
                cusum_neg[i] = 0.0
            else:
                k = k_mult * rolling_std[i]
                cusum_pos[i] = max(0, cusum_pos[i-1] + ret[i] - rolling_mean[i] - k)
                cusum_neg[i] = min(0, cusum_neg[i-1] + ret[i] - rolling_mean[i] + k)
            cusum_thresh[i] = h_mult * rolling_std[i]

        # Changepoint signal
        cp_up = cusum_pos > cusum_thresh
        cp_down = np.abs(cusum_neg) > cusum_thresh

        # Fresh break: was below threshold for `below_bars` before current break
        fresh_up = np.zeros(n, dtype=np.bool_)
        fresh_down = np.zeros(n, dtype=np.bool_)
        for i in range(below_bars + fresh, n):
            if cp_up[i]:
                # Check if CUSUM was below threshold for below_bars before
                was_below = True
                for j in range(fresh, below_bars + fresh):
                    if cusum_pos[i-j] > cusum_thresh[i-j]:
                        was_below = False
                        break
                fresh_up[i] = was_below
            if cp_down[i]:
                was_below = True
                for j in range(fresh, below_bars + fresh):
                    if abs(cusum_neg[i-j]) > cusum_thresh[i-j]:
                        was_below = False
                        break
                fresh_down[i] = was_below

        # ROC confirmation
        roc3 = np.zeros(n, dtype=np.float64)
        for i in range(3, n):
            if close[i-3] > 0:
                roc3[i] = (close[i] - close[i-3]) / close[i-3] * 10000

        # Volume filter
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_surge = volume > (1.5 * avg_vol)

        time_ok = (time_mins >= 565) & (time_mins <= 910)

        long_entry = (
            fresh_up & (close > vwap) & (roc3 > roc3_th)
            & vol_surge & time_ok
        )
        short_entry = (
            fresh_down & (close < vwap) & (roc3 < -roc3_th)
            & vol_surge & time_ok
        )

        # Signal exit: opposite changepoint or CUSUM resets
        sig_exit_long = cp_down | (cusum_pos < 0.1)
        sig_exit_short = cp_up | (np.abs(cusum_neg) < 0.1)

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
