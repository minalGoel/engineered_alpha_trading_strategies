"""VWAP +/-2SD Band Reversion — Opus_4

Thesis: Price touching VWAP +/-2 standard-deviation bands with volume
confirmation tends to revert to VWAP +/-1SD. Uses rolling std of
(close - vwap) to compute bands.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "opus_vwap_band_reversion_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=2.0, low=1.5, high=4.0),
            TunableParam("stop_loss_pct", default=0.0035, low=0.002, high=0.006),
            TunableParam("breakeven_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 2.0)
        stop_pct = params.get("stop_loss_pct", 0.0035)
        be_pct = params.get("breakeven_pct", 0.002)

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

        # ── Rolling std of (close - vwap) over 30 bars ──
        dev = close - vwap
        dev_series = pl.Series("dev", dev)
        vwap_std = dev_series.rolling_std(30).to_numpy().astype(np.float64)
        vwap_std = np.nan_to_num(vwap_std, nan=1e10)
        vwap_std = np.clip(vwap_std, 1e-10, None)

        # ── VWAP bands ──
        vwap_upper_2sd = vwap + 2.0 * vwap_std
        vwap_lower_2sd = vwap - 2.0 * vwap_std
        vwap_upper_1sd = vwap + 1.0 * vwap_std
        vwap_lower_1sd = vwap - 1.0 * vwap_std

        # ── Volume filter: volume > 2x 10-bar avg ──
        avg_vol_10 = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol_10 = np.nan_to_num(avg_vol_10, nan=1.0)
        avg_vol_10 = np.clip(avg_vol_10, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol_10

        # ── Time filter ──
        time_ok = (time_mins >= 570) & (time_mins <= 885)

        # ── Entries ──
        # Long: low touched lower 2sd but close recovered above it, with volume
        long_entry = ((low <= vwap_lower_2sd) & (close > vwap_lower_2sd)
                      & vol_ok & time_ok)
        # Short: high touched upper 2sd but close fell below it
        short_entry = ((high >= vwap_upper_2sd) & (close < vwap_upper_2sd)
                       & time_ok)

        # ── Target: vwap +/- 1SD ──
        # For longs target is vwap_lower_1sd (closer to vwap), for shorts vwap_upper_1sd
        # Use a blended target: for long entries use lower_1sd, for shorts use upper_1sd
        # Since target_indicator is a single array, use vwap as the target
        # and the state machine will handle the direction
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
            breakeven_pct=be_pct,
            time_stop_bars=60,
        )
