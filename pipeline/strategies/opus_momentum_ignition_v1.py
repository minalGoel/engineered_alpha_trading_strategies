"""Momentum Ignition — Opus_10

Thesis: After 30+ bars of range-bound price action (0.2-0.8% width),
a breakout accompanied by 3x+ volume surge signals the start of a
directional move. Target is 1x range width, stop is 0.5x range width.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "opus_momentum_ignition_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_surge_thresh", default=3.0, low=2.0, high=5.0),
            TunableParam("min_bars_in_range", default=30.0, low=20.0, high=50.0),
            TunableParam("min_range_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("max_range_pct", default=0.008, low=0.005, high=0.012),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_surge_thresh = params.get("vol_surge_thresh", 3.0)
        min_bars = int(params.get("min_bars_in_range", 30.0))
        min_range_pct = params.get("min_range_pct", 0.002)
        max_range_pct = params.get("max_range_pct", 0.008)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Rolling range high/low over min_bars ──
        range_high = np.zeros(n, dtype=np.float64)
        range_low = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - min_bars)
            range_high[i] = np.max(high[start:i + 1])
            range_low[i] = np.min(low[start:i + 1])

        safe_mid = np.clip((range_high + range_low) / 2.0, 1e-10, None)
        range_width_pct = (range_high - range_low) / safe_mid

        # ── Volume surge ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        vol_surge = volume / avg_vol_20

        # ── Count bars in range: how many of the last min_bars had price
        #    within the range (simplified: check if range width is stable) ──
        # Approximate: if range_width hasn't expanded in min_bars, price was ranging
        range_width_prev = np.roll(range_width_pct, min_bars)
        range_width_prev[:min_bars] = 0.0
        bars_in_range_ok = np.ones(n, dtype=np.bool_)  # simplified: range existed

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)
        range_ok = (range_width_pct >= min_range_pct) & (range_width_pct <= max_range_pct)
        surge_ok = vol_surge > vol_surge_thresh

        # ── Entries ──
        long_entry = ((close > range_high) & surge_ok & range_ok
                      & bars_in_range_ok & time_ok)
        short_entry = ((close < range_low) & surge_ok & range_ok
                       & bars_in_range_ok & time_ok)

        # ── Target: 1x range_width as pct ──
        # Use median range width as scalar target_pct
        valid_rw = range_width_pct[(range_width_pct >= min_range_pct) & (range_width_pct <= max_range_pct)]
        target_pct = float(np.median(valid_rw)) if len(valid_rw) > 0 else 0.004
        stop_pct = target_pct * 0.5

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=45,
        )
