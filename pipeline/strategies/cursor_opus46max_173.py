"""Fractal Dimension v1 — cursor_opus46max_173

Thesis: Higuchi fractal dimension (k_max=8, window=40) measures price path
complexity. FD < 1.35 = trending (smooth path), trade EMA(10)/EMA(25) crossover.
FD >= 1.35 = noisy, stay flat. VWAP slope confirms direction.
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


def _higuchi_fd(series, k_max=8):
    """Compute Higuchi fractal dimension of a 1-D series."""
    n = len(series)
    if n < k_max + 1:
        return 1.5  # default neutral value
    lk = np.zeros(k_max, dtype=np.float64)
    for k in range(1, k_max + 1):
        lm_sum = 0.0
        count = 0
        for m in range(1, k + 1):
            # Number of elements in this sub-series
            num = int(np.floor((n - m) / k))
            if num < 1:
                continue
            length = 0.0
            for i in range(1, num + 1):
                idx_cur = m - 1 + i * k
                idx_prev = m - 1 + (i - 1) * k
                if idx_cur < n and idx_prev < n:
                    length += abs(series[idx_cur] - series[idx_prev])
            # Normalize
            norm = (n - 1) / (num * k * k)
            length *= norm
            lm_sum += length
            count += 1
        if count > 0:
            lk[k - 1] = lm_sum / count
        else:
            lk[k - 1] = 0.0
    # Linear regression of log(Lk) vs log(1/k)
    valid = lk > 0
    if np.sum(valid) < 2:
        return 1.5
    ks = np.arange(1, k_max + 1, dtype=np.float64)
    log_inv_k = np.log(1.0 / ks[valid])
    log_lk = np.log(lk[valid])
    mx = np.mean(log_inv_k)
    my = np.mean(log_lk)
    num = np.sum((log_inv_k - mx) * (log_lk - my))
    den = np.sum((log_inv_k - mx) ** 2)
    if den < 1e-20:
        return 1.5
    fd = num / den
    return np.clip(fd, 1.0, 2.0)


class Strategy(BaseStrategy):
    name = "cursor_opus46max_173"
    is_long_only = False
    session_start = 595   # 09:55
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("fd_window", default=40.0, low=30.0, high=60.0),
            TunableParam("fd_trend_thresh", default=1.35, low=1.25, high=1.45),
            TunableParam("fd_exit_thresh", default=1.45, low=1.38, high=1.55),
            TunableParam("target_pct", default=0.0045, low=0.003, high=0.006),
            TunableParam("stop_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        fd_window = int(params.get("fd_window", 40.0))
        fd_thresh = params.get("fd_trend_thresh", 1.35)
        fd_exit = params.get("fd_exit_thresh", 1.45)
        target = params.get("target_pct", 0.0045)
        stop = params.get("stop_pct", 0.0025)

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

        # Rolling Higuchi fractal dimension
        fd = np.full(n, 1.5, dtype=np.float64)
        for i in range(fd_window, n):
            window = close[i - fd_window + 1:i + 1]
            fd[i] = _higuchi_fd(window, k_max=8)

        is_trending = fd < fd_thresh

        # Regime stability: trending for at least 5 consecutive bars
        regime_stable = np.zeros(n, dtype=np.bool_)
        for i in range(5, n):
            stable = True
            for j in range(5):
                if not is_trending[i - j]:
                    stable = False
                    break
            regime_stable[i] = stable

        # EMA(10)/EMA(25) for trend direction
        ema10 = _compute_ema(close, 10)
        ema25 = _compute_ema(close, 25)

        # VWAP slope (10-bar linear regression slope in bps)
        vwap_slope = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            if vwap[i] > 0:
                w = vwap[i - 9:i + 1]
                x = np.arange(10, dtype=np.float64)
                mx = np.mean(x)
                mw = np.mean(w)
                num = np.sum((x - mx) * (w - mw))
                den = np.sum((x - mx) ** 2)
                if den > 1e-20 and mw > 0:
                    vwap_slope[i] = (num / den) / mw * 10000.0

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 595) & (time_mins <= 910)

        # Long: trending + EMA cross up + above VWAP + VWAP slope up
        long_entry = is_trending & (ema10 > ema25) & (close > vwap) & \
                     (vwap_slope > 0) & regime_stable & vol_ok & time_ok

        # Short: trending + EMA cross down + below VWAP + VWAP slope down
        short_entry = is_trending & (ema10 < ema25) & (close < vwap) & \
                      (vwap_slope < 0) & regime_stable & vol_ok & time_ok

        # Signal exit: FD rises above exit threshold (price becoming noisy)
        sig_exit_long = fd > fd_exit
        sig_exit_short = fd > fd_exit

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop,
            target_pct=target,
            trailing_stop_pct=0.0015,
            trailing_activate_pct=0.0025,
            time_stop_bars=60,
        )
