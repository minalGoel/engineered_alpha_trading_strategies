"""ORB Momentum v22 — Gemini31Pro Strategy 022

Thesis: The opening 15-30 minutes represent overnight information assimilation.
A breakout beyond this range indicates directional institutional order flow.
Uses HIGHEST(high,20)/LOWEST(low,20) for range, VWAP confirmation,
volume > SMA(volume,20). Target 1.9 * ATR(20). Stop 1.5 * ATR(14).
Time stop 30 bars. No VIX filter.
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


def _rolling_max(arr, period):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        out[i] = np.max(arr[i - period + 1:i + 1])
    return out


def _rolling_min(arr, period):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        out[i] = np.min(arr[i - period + 1:i + 1])
    return out


class Strategy(BaseStrategy):
    name = "gemini31pro_orb_momentum_v22"
    is_long_only = False
    session_start = 570
    session_end = 915
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("target_atr_mult", default=1.9, low=1.0, high=3.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        target_mult = params.get("target_atr_mult", 1.9)
        be_pct = params.get("breakeven_pct", 0.005)

        n = len(df)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ORB: HIGHEST(high, 20) and LOWEST(low, 20) ──
        orb_high = _rolling_max(high, 20)
        orb_low = _rolling_min(low, 20)
        orb_high = np.nan_to_num(orb_high, nan=1e10)
        orb_low = np.nan_to_num(orb_low, nan=-1e10)

        # ── Volume filter: volume > SMA(volume, 20) ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── ATR(14) for stops, ATR(20) for target ──
        atr_14 = _compute_atr(high, low, close, 14)
        atr_20 = _compute_atr(high, low, close, 20)

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (close > orb_high) & (close > vwap) & vol_ok & time_ok
        short_entry = (close < orb_low) & (close < vwap) & vol_ok & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr_14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=1.5,
            target_atr_mult=target_mult,
            breakeven_pct=be_pct,
            time_stop_bars=30,
        )
