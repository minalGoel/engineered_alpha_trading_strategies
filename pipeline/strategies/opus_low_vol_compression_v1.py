"""Realized Vol Compression Breakout — Opus_26

Thesis: When realized volatility compresses to extreme lows (bottom 10th
percentile over 250 bars), the next directional move tends to be large.
Enter in the direction of VWAP/EMA bias on compression.
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


def _compute_ema(data, period):
    """EMA with Wilder-style warmup."""
    n = len(data)
    ema = np.zeros(n, dtype=np.float64)
    if n < period:
        return ema
    ema[period - 1] = np.mean(data[:period])
    k = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = data[i] * k + ema[i - 1] * (1 - k)
    return ema


class Strategy(BaseStrategy):
    name = "opus_low_vol_compression_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 840     # 14:00
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rvol_pctile_thresh", default=10.0, low=5.0, high=20.0),
            TunableParam("rvol_baseline_mult", default=1.0, low=0.5, high=1.5),
            TunableParam("stop_atr_mult", default=0.8, low=0.5, high=1.5),
            TunableParam("target_atr_mult", default=2.0, low=1.0, high=3.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rvol_pctile_thresh = params.get("rvol_pctile_thresh", 10.0)
        rvol_baseline_mult = params.get("rvol_baseline_mult", 1.0)
        stop_atr = params.get("stop_atr_mult", 0.8)
        target_atr = params.get("target_atr_mult", 2.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── EMA(20) ──
        ema20 = _compute_ema(close, 20)

        # ── Log returns ──
        log_ret = np.zeros(n, dtype=np.float64)
        log_ret[1:] = np.log(np.clip(close[1:] / np.clip(close[:-1], 1e-10, None), 1e-10, None))

        # ── Realized vol (30-bar annualized) ──
        rvol_30 = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            window = log_ret[i - 29:i + 1]
            rvol_30[i] = np.std(window) * np.sqrt(252 * 78)  # annualize: 78 bars/day approx

        # ── Percentile rank of rvol_30 over 250 bars ──
        rvol_pctile = np.full(n, 50.0, dtype=np.float64)
        for i in range(250, n):
            window = rvol_30[i - 249:i + 1]
            valid = window[window > 0]
            if len(valid) > 0:
                rvol_pctile[i] = (np.sum(valid < rvol_30[i]) / len(valid)) * 100.0

        # ── Baseline rvol (median over 250 bars) ──
        rvol_median = np.zeros(n, dtype=np.float64)
        for i in range(250, n):
            window = rvol_30[i - 249:i + 1]
            valid = window[window > 0]
            if len(valid) > 0:
                rvol_median[i] = np.median(valid)

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Filters ──
        compression = rvol_pctile < rvol_pctile_thresh
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entries ──
        long_entry = compression & (close > vwap) & (close > ema20) & time_ok
        short_entry = compression & (close < vwap) & (close < ema20) & time_ok

        # ── Signal exit: rvol expands above baseline ──
        sig_exit = rvol_30 > (rvol_median * rvol_baseline_mult)
        sig_exit[:250] = False

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit,
            signal_exit_short=sig_exit,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_atr_mult=target_atr,
            time_stop_bars=90,
        )
