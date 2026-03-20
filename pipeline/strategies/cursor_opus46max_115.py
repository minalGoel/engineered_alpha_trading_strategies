# AUDIT FIX: Add session window filtering to prevent entries outside session_start/session_end
"""Sector Leader-Follower v1 — cursor_opus46max_115

Thesis: Within each sector, information diffuses from the largest stock
(leader) to smaller constituents (followers) with a 3-10 min lag.
Proxy leader moves using index, and trade individual stock catch-up.
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
    name = "cursor_opus46max_115"
    is_long_only = False
    session_start = 575   # 09:35
    session_end = 920     # 15:20
    max_trades_per_day = 12
    assumptions = ["Sector leader proxied via index_close; follower is the traded stock"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("leader_zscore_thresh", default=2.0, low=1.5, high=3.0),
            TunableParam("follower_lag_bps", default=5.0, low=3.0, high=10.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.0012, low=0.0006, high=0.002),
            TunableParam("target_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        leader_zs = params.get("leader_zscore_thresh", 2.0)
        lag_bps = params.get("follower_lag_bps", 5.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.0012)
        target_pct = params.get("target_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Leader (index) 3-bar return in bps
        leader_ret = np.zeros(n, dtype=np.float64)
        for i in range(3, n):
            if index_close[i-3] > 1e-10:
                leader_ret[i] = (index_close[i] - index_close[i-3]) / index_close[i-3] * 10000.0

        # Leader return z-score over 60 bars
        leader_zs_arr = np.zeros(n, dtype=np.float64)
        lookback = 60
        for i in range(lookback - 1, n):
            window = leader_ret[i - lookback + 1: i + 1]
            mu = np.mean(window)
            std = np.std(window)
            if std > 1e-10:
                leader_zs_arr[i] = (leader_ret[i] - mu) / std

        # Follower (stock) 3-bar return
        follower_ret = np.zeros(n, dtype=np.float64)
        for i in range(3, n):
            if close[i-3] > 1e-10:
                follower_ret[i] = (close[i] - close[i-3]) / close[i-3] * 10000.0

        # Relative return: follower - leader
        rel_ret = follower_ret - leader_ret

        # Volume filter
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.7 * avg_vol

        vix_ok = vix < vix_max

        # Confirmation: follower starts catching up
        catching_up_long = np.zeros(n, dtype=np.bool_)
        catching_up_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            catching_up_long[i] = close[i] > close[i-1]
            catching_up_short[i] = close[i] < close[i-1]

        long_entry = ((leader_zs_arr > leader_zs) & (rel_ret < -lag_bps) &
                      (close > vwap * 0.998) & catching_up_long & vol_ok & vix_ok & in_session)
        short_entry = ((leader_zs_arr < -leader_zs) & (rel_ret > lag_bps) &
                       (close < vwap * 1.002) & catching_up_short & vol_ok & vix_ok & in_session)

        # Signal exit: relative return crosses zero (caught up) or leader fades
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if rel_ret[i] >= 0 and rel_ret[i-1] < 0:
                sig_exit_long[i] = True
            if rel_ret[i] <= 0 and rel_ret[i-1] > 0:
                sig_exit_short[i] = True
            if abs(leader_zs_arr[i]) < 1.0:
                sig_exit_long[i] = True
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
            trailing_stop_pct=0.0005,
            trailing_activate_pct=0.001,
            time_stop_bars=10,
        )
