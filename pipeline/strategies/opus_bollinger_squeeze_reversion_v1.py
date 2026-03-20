"""Bollinger Band Squeeze Reversion — Opus_2

Thesis: When BB bandwidth is at a low percentile (squeeze), price touching
the outer band tends to revert to the midline. Enter on band touch in
low-volatility regimes.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "opus_bollinger_squeeze_reversion_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bw_pctile_thresh", default=20.0, low=10.0, high=35.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.008),
            TunableParam("breakeven_pct", default=0.0025, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        bw_pctile_thresh = params.get("bw_pctile_thresh", 20.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.004)
        be_pct = params.get("breakeven_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Bollinger Bands (20, 2) ──
        sma_20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        std_20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        sma_20 = np.nan_to_num(sma_20, nan=0.0)
        std_20 = np.nan_to_num(std_20, nan=1e10)
        std_20 = np.clip(std_20, 1e-10, None)

        bb_upper = sma_20 + 2.0 * std_20
        bb_lower = sma_20 - 2.0 * std_20
        bb_mid = sma_20.copy()

        # ── Bandwidth ──
        safe_mid = np.clip(np.abs(sma_20), 1e-10, None)
        bandwidth = (bb_upper - bb_lower) / safe_mid

        # ── Bandwidth percentile (120-bar rolling) ──
        bw_pctile = np.full(n, 50.0, dtype=np.float64)
        for i in range(120, n):
            window = bandwidth[i - 120:i]
            valid = window[~np.isnan(window)]
            if len(valid) > 0:
                bw_pctile[i] = (np.sum(valid < bandwidth[i]) / len(valid)) * 100.0

        # ── Filters ──
        squeeze = bw_pctile < bw_pctile_thresh
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (close < bb_lower) & squeeze & vix_ok & time_ok
        short_entry = (close > bb_upper) & squeeze & vix_ok & time_ok

        # ── Target: BB midline (use_target_indicator) ──
        target_indicator = bb_mid.copy()

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=target_indicator,
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=60,
        )
