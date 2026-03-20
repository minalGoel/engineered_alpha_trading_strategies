"""Donchian Channel Breakout — claude_project_23_of_25

Thesis: A breakout above the 30-bar Donchian high with volume and VWAP
confirmation signals the start of a trending move. Use ATR-based targets
and Donchian mid as a trailing signal exit.
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


class Strategy(BaseStrategy):
    name = "donchian_channel_breakout_v1"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=1.3, low=1.0, high=2.5),
            TunableParam("target_atr_mult", default=2.0, low=1.0, high=4.0),
            TunableParam("vix_low", default=12.0, low=8.0, high=16.0),
            TunableParam("vix_high", default=24.0, low=20.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 1.3)
        target_atr_mult = params.get("target_atr_mult", 2.0)
        vix_low = params.get("vix_low", 12.0)
        vix_high = params.get("vix_high", 24.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)

        volume = np.nan_to_num(volume, nan=1.0)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Donchian Channel (30 bars) ──
        dc_period = 30
        donchian_high = np.zeros(n, dtype=np.float64)
        donchian_low = np.zeros(n, dtype=np.float64)
        donchian_mid = np.zeros(n, dtype=np.float64)

        for i in range(dc_period, n):
            donchian_high[i] = np.max(high[i - dc_period:i])
            donchian_low[i] = np.min(low[i - dc_period:i])
            donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0

        # Fill early bars
        for i in range(min(dc_period, n)):
            if i > 0:
                donchian_high[i] = np.max(high[:i + 1])
                donchian_low[i] = np.min(low[:i + 1])
                donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
            else:
                donchian_high[i] = high[i]
                donchian_low[i] = low[i]
                donchian_mid[i] = (high[i] + low[i]) / 2.0

        # ── Volume filter ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol_20

        # ── VIX filter ──
        vix_ok = (vix >= vix_low) & (vix <= vix_high)

        # ── Entry ──
        long_entry = (close > donchian_high) & vol_ok & (close > vwap) & vix_ok
        short_entry = (close < donchian_low) & vix_ok

        # ── Signal exit: close crosses Donchian mid ──
        signal_exit_long = close < donchian_mid
        signal_exit_short = close > donchian_mid

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=donchian_mid.copy(),
            target_atr_mult=target_atr_mult,
            time_stop_bars=240,
        )
