# AUDIT FIX: Add session window filtering to prevent entries outside session_start/session_end
"""F&O vs Cash Rotation v1 — cursor_opus46max_114

Thesis: F&O segment stocks attract leveraged speculative flow that tends
to unwind intraday.  When F&O stocks spike on high volume (proxy for OI
buildup), fade the speculative move.  Use volume surge as OI proxy.
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
    name = "cursor_opus46max_114"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 920     # 15:20
    max_trades_per_day = 6
    assumptions = ["OI change proxied via volume surge relative to average"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_surge_thresh", default=1.5, low=1.2, high=3.0),
            TunableParam("return_30_thresh", default=20.0, low=10.0, high=40.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("target_pct", default=0.003, low=0.0015, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_surge = params.get("vol_surge_thresh", 1.5)
        ret_thresh = params.get("return_30_thresh", 20.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.002)
        target_pct = params.get("target_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Volume surge ratio
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # 30-bar return (bps)
        ret_30 = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            if close[i-30] > 1e-10:
                ret_30[i] = (close[i] - close[i-30]) / close[i-30] * 10000.0

        vix_ok = vix < vix_max

        # Fade speculative spikes: short when big up on volume, long when big down
        long_entry = ((vol_ratio > vol_surge) & (ret_30 < -ret_thresh) &
                      (close < vwap) & vix_ok & in_session)
        short_entry = ((vol_ratio > vol_surge) & (ret_30 > ret_thresh) &
                       (close > vwap) & vix_ok & in_session)

        # Signal exit: volume normalizes
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if vol_ratio[i] < 1.0 and vol_ratio[i-1] >= 1.0:
                sig_exit_long[i] = True
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            use_target_indicator=True,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=0.0005,
            trailing_activate_pct=0.001,
            time_stop_bars=60,
        )
