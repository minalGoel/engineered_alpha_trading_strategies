"""VWAP RSI Volume Hybrid v1 — cursor_opus46max_151

Thesis: Intraday price reverts toward VWAP; adding RSI oversold/overbought
filter and abnormal volume confirmation (>1.5x 20-bar avg) improves hit rate.
Triple-confirmation: VWAP distance, RSI extreme, volume spike.
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


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_151"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dist_bps", default=40.0, low=20.0, high=80.0),
            TunableParam("rsi_long_thresh", default=30.0, low=20.0, high=40.0),
            TunableParam("rsi_short_thresh", default=70.0, low=60.0, high=80.0),
            TunableParam("vol_ratio_thresh", default=1.5, low=1.0, high=3.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("trailing_stop_pct", default=0.0015, low=0.001, high=0.003),
            TunableParam("trailing_activate_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vwap_dist_bps = params.get("vwap_dist_bps", 40.0)
        rsi_long = params.get("rsi_long_thresh", 30.0)
        rsi_short = params.get("rsi_short_thresh", 70.0)
        vol_ratio = params.get("vol_ratio_thresh", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.004)
        trail_pct = params.get("trailing_stop_pct", 0.0015)
        trail_act = params.get("trailing_activate_pct", 0.002)

        n = len(df)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP distance in bps
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        vwap_dist = (close - vwap) / safe_vwap * 10000.0

        # RSI
        rsi = _compute_rsi(close, 14)

        # Volume ratio
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume / avg_vol > vol_ratio

        # Uptick / downtick confirmation
        uptick = np.zeros(n, dtype=np.bool_)
        downtick = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            uptick[i] = close[i] > close[i - 1]
            downtick[i] = close[i] < close[i - 1]

        # Time & VIX filters
        time_ok = (time_mins >= 560) & (time_mins <= 900)
        vix_ok = vix < 25.0

        # Entries
        long_cond = (vwap_dist < -vwap_dist_bps) & (rsi < rsi_long) & vol_ok & uptick & time_ok & vix_ok
        short_cond = (vwap_dist > vwap_dist_bps) & (rsi > rsi_short) & vol_ok & downtick & time_ok & vix_ok

        # Signal exits: RSI crosses 50
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if rsi[i - 1] < 50.0 and rsi[i] >= 50.0:
                sig_exit_long[i] = True
            if rsi[i - 1] > 50.0 and rsi[i] <= 50.0:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_cond,
            short_entry=short_cond,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=60,
        )
