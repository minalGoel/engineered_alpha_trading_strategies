"""Trade Size Clustering v1 — cursor_opus46max_102

Thesis: Institutional orders split into round-lot sizes create detectable
clusters.  Proxy institutional flow using volume spikes relative to median
and directional bias from close position within bar.
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
    name = "cursor_opus46max_102"
    is_long_only = False
    session_start = 575   # 09:35
    session_end = 920     # 15:20
    max_trades_per_day = 8
    assumptions = ["Tick-level trade data proxied via volume spikes and bar close position"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("large_vol_mult", default=2.0, low=1.5, high=4.0),
            TunableParam("sustained_bars", default=3, low=2, high=5),
            TunableParam("stop_loss_pct", default=0.0015, low=0.0008, high=0.003),
            TunableParam("target_pct", default=0.003, low=0.0015, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        large_vol_mult = params.get("large_vol_mult", 2.0)
        sustained_bars = int(params.get("sustained_bars", 3))
        stop_pct = params.get("stop_loss_pct", 0.0015)
        target_pct = params.get("target_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Volume ratio over 5-bar rolling window (proxy for large trade detection)
        window = 5
        vol_ratio = np.zeros(n, dtype=np.float64)
        median_vol = np.ones(n, dtype=np.float64)
        for i in range(59, n):
            median_vol[i] = np.median(volume[max(0, i-59):i+1])
        median_vol = np.clip(median_vol, 1.0, None)

        # large_trade_pct proxy: fraction of volume in window that exceeds threshold
        large_flag = np.zeros(n, dtype=np.float64)
        for i in range(window - 1, n):
            win_vol = volume[i - window + 1: i + 1]
            med = np.median(volume[max(0, i-59):i+1]) if i >= 59 else np.median(volume[:i+1])
            med = max(med, 1.0)
            large_flag[i] = np.sum(win_vol > large_vol_mult * med) / window

        # Cluster direction: sign of close position within bar, sustained
        bar_range = high - low
        bar_range = np.where(bar_range < 1e-10, 1e-10, bar_range)
        close_pos = 2.0 * (close - low) / bar_range - 1.0  # -1 to +1

        # Rolling sum of close_pos over window
        cluster_dir = np.zeros(n, dtype=np.float64)
        for i in range(window - 1, n):
            cluster_dir[i] = np.sum(close_pos[i - window + 1: i + 1])

        # Sustained volume condition
        sustained_long = np.zeros(n, dtype=np.bool_)
        sustained_short = np.zeros(n, dtype=np.bool_)
        for i in range(sustained_bars - 1, n):
            all_high = True
            for j in range(sustained_bars):
                if large_flag[i - j] < 0.35:
                    all_high = False
                    break
            if all_high and cluster_dir[i] > 0:
                sustained_long[i] = True
            if all_high and cluster_dir[i] < 0:
                sustained_short[i] = True

        long_entry = (large_flag > 0.4) & (cluster_dir > 0) & (close >= vwap) & sustained_long
        short_entry = (large_flag > 0.4) & (cluster_dir < 0) & (close <= vwap) & sustained_short

        # Signal exit: cluster direction reverses sign
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if cluster_dir[i] < 0 and cluster_dir[i-1] >= 0:
                sig_exit_long[i] = True
            if cluster_dir[i] > 0 and cluster_dir[i-1] <= 0:
                sig_exit_short[i] = True

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
            trailing_stop_pct=0.001,
            trailing_activate_pct=0.002,
            time_stop_bars=15,
        )
