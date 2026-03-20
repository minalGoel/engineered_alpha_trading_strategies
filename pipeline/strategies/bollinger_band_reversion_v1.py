"""Bollinger Band Mean Reversion — GPT_5_of_10

Thesis: Price touching/breaching Bollinger Bands signals temporary overshoot.
Enter on band touch, target the middle band (SMA20).
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "bollinger_band_reversion_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=1.5, low=1.0, high=3.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        # ── Bollinger Bands (20, 2) ──
        sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        sma20 = np.nan_to_num(sma20, nan=0.0)
        std20 = np.nan_to_num(std20, nan=1e10)  # huge std → no signal during warmup

        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20

        # ── Volume filter ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol_20

        # ── Entry ──
        long_entry = (close < bb_lower) & vol_ok
        short_entry = (close > bb_upper) & vol_ok

        # ── Target: middle band (SMA20) ──
        target_indicator = sma20.copy()

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=target_indicator,
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            time_stop_bars=10,
        )
