"""Volume Clock Signal v1 — cursor_opus46max_104

Thesis: Volume-clock bars normalize for information flow.  RSI and Bollinger
Bands computed on volume-bucketed data produce more stationary signals.
Approximate by weighting standard indicators by relative volume intensity.
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
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_104"
    is_long_only = False
    session_start = 575   # 09:35
    session_end = 920     # 15:20
    max_trades_per_day = 15
    assumptions = ["Volume-clock bars approximated using volume-weighted indicators on time bars"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_long", default=30.0, low=20.0, high=40.0),
            TunableParam("rsi_short", default=70.0, low=60.0, high=80.0),
            TunableParam("bb_std", default=2.0, low=1.5, high=3.0),
            TunableParam("vwap_dev_thresh", default=15.0, low=8.0, high=25.0),
            TunableParam("stop_loss_pct", default=0.0012, low=0.0006, high=0.002),
            TunableParam("target_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rsi_long = params.get("rsi_long", 30.0)
        rsi_short = params.get("rsi_short", 70.0)
        bb_std = params.get("bb_std", 2.0)
        vwap_dev_thresh = params.get("vwap_dev_thresh", 15.0)
        stop_pct = params.get("stop_loss_pct", 0.0012)
        target_pct = params.get("target_pct", 0.0025)

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

        # RSI(14) on close
        rsi = _compute_rsi(close, 14)

        # Bollinger Bands (20, bb_std)
        bb_period = 20
        bb_upper = np.full(n, 1e10, dtype=np.float64)
        bb_lower = np.full(n, -1e10, dtype=np.float64)
        for i in range(bb_period - 1, n):
            window = close[i - bb_period + 1: i + 1]
            mu = np.mean(window)
            std = np.std(window)
            bb_upper[i] = mu + bb_std * std
            bb_lower[i] = mu - bb_std * std

        # VWAP deviation in bps
        safe_vwap = np.where(vwap > 1e-10, vwap, 1e-10)
        vwap_dev = (close - vwap) / safe_vwap * 10000.0

        # RSI turning (confirmation)
        rsi_turning_up = np.zeros(n, dtype=np.bool_)
        rsi_turning_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            rsi_turning_up[i] = rsi[i] > rsi[i-1]
            rsi_turning_down[i] = rsi[i] < rsi[i-1]

        # Volume filter: skip low-volume bars
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.8 * avg_vol

        long_entry = ((rsi < rsi_long) & (close <= bb_lower) &
                      (vwap_dev < -vwap_dev_thresh) & (close > low) &
                      rsi_turning_up & vol_ok)
        short_entry = ((rsi > rsi_short) & (close >= bb_upper) &
                       (vwap_dev > vwap_dev_thresh) & (close < high) &
                       rsi_turning_down & vol_ok)

        # Signal exit: RSI crosses 50
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if rsi[i] >= 50 and rsi[i-1] < 50:
                sig_exit_long[i] = True
            if rsi[i] <= 50 and rsi[i-1] > 50:
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
            trailing_stop_pct=0.0008,
            trailing_activate_pct=0.0015,
            time_stop_bars=10,
        )
