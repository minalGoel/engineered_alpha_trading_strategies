"""Hurst Exponent v1 — cursor_opus46max_172

Thesis: Rolling Hurst exponent (R/S method, window=60) classifies regime.
H > 0.55: persistent (trend-follow with EMA 8/20). H < 0.45: anti-persistent
(mean-revert with BB). 0.45-0.55: random walk, no trade.
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


def _hurst_rs(series):
    """Compute Hurst exponent via rescaled range (R/S) method."""
    n = len(series)
    if n < 20:
        return 0.5
    # Use sub-series lengths: powers of 2 up to n//2
    max_k = int(np.floor(np.log2(n)))
    if max_k < 2:
        return 0.5
    ns = []
    rs_vals = []
    for exp in range(2, max_k + 1):
        size = 2 ** exp
        if size > n:
            break
        num_subseries = n // size
        if num_subseries < 1:
            break
        rs_list = []
        for j in range(num_subseries):
            sub = series[j * size:(j + 1) * size]
            m = np.mean(sub)
            y = np.cumsum(sub - m)
            r = np.max(y) - np.min(y)
            s = np.std(sub, ddof=1)
            if s > 1e-20:
                rs_list.append(r / s)
        if len(rs_list) > 0:
            ns.append(size)
            rs_vals.append(np.mean(rs_list))
    if len(ns) < 2:
        return 0.5
    log_n = np.log(np.array(ns, dtype=np.float64))
    log_rs = np.log(np.array(rs_vals, dtype=np.float64))
    # Linear regression: log(R/S) = H * log(n) + c
    mx = np.mean(log_n)
    my = np.mean(log_rs)
    num = np.sum((log_n - mx) * (log_rs - my))
    den = np.sum((log_n - mx) ** 2)
    if den < 1e-20:
        return 0.5
    h = num / den
    return np.clip(h, 0.0, 1.0)


class Strategy(BaseStrategy):
    name = "cursor_opus46max_172"
    is_long_only = False
    session_start = 615   # 10:15
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("hurst_window", default=60.0, low=40.0, high=80.0),
            TunableParam("h_trend_thresh", default=0.55, low=0.52, high=0.60),
            TunableParam("h_revert_thresh", default=0.45, low=0.40, high=0.48),
            TunableParam("persist_target_pct", default=0.0045, low=0.003, high=0.006),
            TunableParam("revert_target_pct", default=0.003, low=0.002, high=0.004),
            TunableParam("persist_stop_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("revert_stop_pct", default=0.002, low=0.0012, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        h_window = int(params.get("hurst_window", 60.0))
        h_trend = params.get("h_trend_thresh", 0.55)
        h_revert = params.get("h_revert_thresh", 0.45)
        p_target = params.get("persist_target_pct", 0.0045)
        r_target = params.get("revert_target_pct", 0.003)
        p_stop = params.get("persist_stop_pct", 0.0025)
        r_stop = params.get("revert_stop_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
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

        # 1-bar returns
        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0:
                returns[i] = close[i] / close[i - 1] - 1.0

        # Rolling Hurst exponent
        hurst = np.full(n, 0.5, dtype=np.float64)
        for i in range(h_window, n):
            window = returns[i - h_window + 1:i + 1]
            hurst[i] = _hurst_rs(window)

        # Regime classification
        is_persistent = hurst > h_trend
        is_anti = hurst < h_revert

        # Regime stability: same regime for 10 bars
        regime_stable = np.zeros(n, dtype=np.bool_)
        for i in range(10, n):
            if is_persistent[i]:
                stable = True
                for j in range(1, 10):
                    if not is_persistent[i - j]:
                        stable = False
                        break
                regime_stable[i] = stable
            elif is_anti[i]:
                stable = True
                for j in range(1, 10):
                    if not is_anti[i - j]:
                        stable = False
                        break
                regime_stable[i] = stable

        # Trend-follow indicators: EMA(8)/EMA(20)
        ema8 = _compute_ema(close, 8)
        ema20 = _compute_ema(close, 20)

        # Mean-revert indicators: BB(20, 2.0)
        sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        sma20 = np.nan_to_num(sma20, nan=0.0)
        std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        std20 = np.nan_to_num(std20, nan=1e10)
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 615) & (time_mins <= 910)

        # Persistent long: EMA crossover trend-follow
        persist_long = is_persistent & (ema8 > ema20) & (close > vwap) & \
                       regime_stable & vol_ok & time_ok
        # Anti-persistent long: BB mean-reversion
        anti_long = is_anti & (close < bb_lower) & (close < vwap) & \
                    regime_stable & vol_ok & time_ok

        # Persistent short
        persist_short = is_persistent & (ema8 < ema20) & (close < vwap) & \
                        regime_stable & vol_ok & time_ok
        # Anti-persistent short
        anti_short = is_anti & (close > bb_upper) & (close > vwap) & \
                     regime_stable & vol_ok & time_ok

        long_entry = persist_long | anti_long
        short_entry = persist_short | anti_short

        # Signal exit: Hurst transitions to random walk zone
        is_random = (~is_persistent) & (~is_anti)
        sig_exit_long = is_random
        sig_exit_short = is_random

        avg_target = (p_target + r_target) / 2.0
        avg_stop = (p_stop + r_stop) / 2.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=sma20.copy(),
            stop_loss_pct=avg_stop,
            target_pct=avg_target,
            trailing_stop_pct=0.0015,
            trailing_activate_pct=0.0025,
            time_stop_bars=60,
        )
