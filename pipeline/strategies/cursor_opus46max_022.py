"""MACD Signal Momentum v1 — cursor_opus46max_022

Thesis: MACD(12,26,9) histogram acceleration (3 consecutive bars of
increasing histogram) signals momentum building. Enter with VWAP
alignment and volume confirmation.
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


def _compute_ema(close, period):
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = close[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_022"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("target_pct", default=0.004, low=0.002, high=0.007),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.004),
            TunableParam("trailing_pct", default=0.0015, low=0.0008, high=0.003),
            TunableParam("trailing_activate", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        tgt_pct = params.get("target_pct", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        trail_pct = params.get("trailing_pct", 0.0015)
        trail_act = params.get("trailing_activate", 0.002)

        n = len(df)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── MACD(12,26,9) ──
        ema12 = _compute_ema(close, 12)
        ema26 = _compute_ema(close, 26)
        macd_line = ema12 - ema26
        macd_signal = _compute_ema(macd_line, 9)
        macd_hist = macd_line - macd_signal

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── Histogram acceleration (3 bars) ──
        hist_accel_long = np.zeros(n, dtype=np.bool_)
        hist_accel_short = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            hist_accel_long[i] = (macd_hist[i] > macd_hist[i - 1] > macd_hist[i - 2])
            hist_accel_short[i] = (macd_hist[i] < macd_hist[i - 1] < macd_hist[i - 2])

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 885)

        # ── Entries ──
        long_entry = (
            (macd_line > macd_signal)
            & (macd_hist > 0)
            & hist_accel_long
            & (close > vwap)
            & vol_ok & time_ok
        )
        short_entry = (
            (macd_line < macd_signal)
            & (macd_hist < 0)
            & hist_accel_short
            & (close < vwap)
            & vol_ok & time_ok
        )

        # ── Signal exit: histogram deceleration or MACD cross ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if macd_hist[i] < macd_hist[i - 1] and macd_hist[i - 1] > 0:
                sig_exit_long[i] = True
            if macd_line[i] < macd_signal[i] and macd_line[i - 1] >= macd_signal[i - 1]:
                sig_exit_long[i] = True
            if macd_hist[i] > macd_hist[i - 1] and macd_hist[i - 1] < 0:
                sig_exit_short[i] = True
            if macd_line[i] > macd_signal[i] and macd_line[i - 1] <= macd_signal[i - 1]:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=40,
        )
