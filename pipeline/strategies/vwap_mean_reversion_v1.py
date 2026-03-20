# AUDIT FIX: Added session window filter to prevent signals outside session_start/session_end.
"""VWAP Mean Reversion — GPT_2_of_10

Thesis: Price often reverts to VWAP (intraday fair value) after overshooting.
Enter on extreme zscore deviations from VWAP with volume confirmation.
Target is VWAP touch.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=1.5, low=1.0, high=3.0),
            TunableParam("vol_mult", default=2.0, low=1.0, high=4.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 1.5)
        vol_mult = params.get("vol_mult", 2.0)
        stop_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)

        # ── VWAP (daily reset using .over() in expression context) ──
        df_vwap = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_cum_tv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cum_v"),
        ]).with_columns(
            (pl.col("_cum_tv") / pl.col("_cum_v")).alias("_vwap")
        )
        vwap = df_vwap["_vwap"].to_numpy().astype(np.float64)

        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        # ── Z-score of (close - vwap) ──
        dev = close - vwap
        std_30 = df["close"].rolling_std(30).to_numpy().astype(np.float64)
        std_30 = np.nan_to_num(std_30, nan=1e10)
        std_30 = np.clip(std_30, 1e-10, None)
        zscore = dev / std_30

        # ── Volume filter ──
        avg_vol_30 = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol_30 = np.nan_to_num(avg_vol_30, nan=1.0)
        avg_vol_30 = np.clip(avg_vol_30, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol_30

        # ── Time filter: skip first 10 bars of day and enforce session end ──
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        # First 10 bars = 09:15 to 09:24 = minutes 555 to 564
        time_ok = (time_mins >= 565) & (time_mins <= self.session_end)

        # NaN safety
        zscore = np.nan_to_num(zscore, nan=0.0)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Entry conditions ──
        long_entry = (close < vwap) & (zscore < -zs_thresh) & vol_ok & time_ok
        short_entry = (close > vwap) & (zscore > zs_thresh) & vol_ok & time_ok

        # ── Target: VWAP touch ──
        target_indicator = vwap.copy()

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
