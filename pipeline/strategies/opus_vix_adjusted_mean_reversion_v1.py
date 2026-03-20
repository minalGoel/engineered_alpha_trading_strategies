"""VIX-Adjusted Mean Reversion — Opus_47

Thesis: Adaptive VWAP zscore threshold based on VIX. Higher VIX
means wider thresholds. Enter on extreme deviation from VWAP with
RSI confirmation; exit on zscore crossing zero.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
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
    avg_gain = np.mean(gain[1:period + 1])
    avg_loss = np.mean(loss[1:period + 1])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


class Strategy(BaseStrategy):
    name = "opus_vix_adjusted_mean_reversion_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_long_thresh", default=40.0, low=25.0, high=50.0),
            TunableParam("rsi_short_thresh", default=60.0, low=50.0, high=75.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("breakeven_pct", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rsi_long = params.get("rsi_long_thresh", 40.0)
        rsi_short = params.get("rsi_short_thresh", 60.0)
        vix_max = params.get("vix_max", 25.0)
        be_pct = params.get("breakeven_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Z-score ──
        dev = close - vwap
        std_30 = df["close"].rolling_std(30).to_numpy().astype(np.float64)
        std_30 = np.nan_to_num(std_30, nan=1e10)
        std_30 = np.clip(std_30, 1e-10, None)
        zscore = dev / std_30
        zscore = np.nan_to_num(zscore, nan=0.0)

        rsi = _compute_rsi(close, 14)
        atr = _compute_atr(high, low, close, 20)

        # ── Adaptive threshold: 1.0 + (vix - 12) * 0.1, clamped [1.0, 2.5] ──
        threshold = np.clip(1.0 + (vix - 12.0) * 0.1, 1.0, 2.5)

        vix_ok = vix < vix_max
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Entries
        long_entry = ((zscore < -threshold) & (rsi < rsi_long) & vix_ok & time_ok)
        short_entry = ((zscore > threshold) & (rsi > rsi_short) & vix_ok & time_ok)

        # Signal exit: zscore crosses 0
        exit_long = np.zeros(n, dtype=np.bool_)
        exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if zscore[i - 1] < 0 and zscore[i] >= 0:
                exit_long[i] = True
            if zscore[i - 1] > 0 and zscore[i] <= 0:
                exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=exit_long,
            signal_exit_short=exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=1.5,
            breakeven_pct=be_pct,
            time_stop_bars=90,
        )
