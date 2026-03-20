"""Microstructure Volume Spike — Opus_8

Thesis: A massive volume spike (>5x average) with minimal price movement
(<0.15% range) indicates large institutional accumulation/distribution.
The next move tends to continue in the bar's direction.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "opus_microstructure_volume_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_ratio_thresh", default=5.0, low=3.0, high=8.0),
            TunableParam("max_bar_range_pct", default=0.0015, low=0.0008, high=0.0025),
            TunableParam("target_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_ratio_thresh = params.get("vol_ratio_thresh", 5.0)
        max_range_pct = params.get("max_bar_range_pct", 0.0015)
        target_pct = params.get("target_pct", 0.003)
        stop_pct = params.get("stop_loss_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Volume ratio vs 20-bar avg ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        vol_ratio = volume / avg_vol_20

        # ── Bar range as pct of close ──
        safe_close = np.clip(np.abs(close), 1e-10, None)
        bar_range_pct = (high - low) / safe_close

        # ── Bar direction ──
        bar_bullish = close > open_
        bar_bearish = close < open_

        # ── Conditions ──
        vol_spike = vol_ratio > vol_ratio_thresh
        tiny_range = bar_range_pct < max_range_pct
        time_ok = (time_mins >= 570) & (time_mins <= 885)

        # ── Entries ──
        long_entry = vol_spike & tiny_range & bar_bullish & time_ok
        short_entry = vol_spike & tiny_range & bar_bearish & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=30,
        )
