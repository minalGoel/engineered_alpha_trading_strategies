"""VWAP Mean Reversion v13 — Gemini31Pro Strategy 013

Thesis: Retail panic sells push price far below VWAP; institutions fade.
Uses stddev(20) for z-score, SMA(volume,10) for rel vol.
Z-score thresh -2.4/+2.4, rel vol > 1.3. No VIX filter.
Time stop 120 bars.
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
    name = "gemini31pro_vwap_mean_reversion_v13"
    is_long_only = False
    session_start = 570
    session_end = 915
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=2.4, low=1.5, high=3.5),
            TunableParam("rel_vol_thresh", default=1.3, low=0.8, high=2.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 2.4)
        rel_vol_thresh = params.get("rel_vol_thresh", 1.3)
        be_pct = params.get("breakeven_pct", 0.005)

        n = len(df)

        # ── VWAP (daily reset) ──
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

        # ── Z-score: (close - vwap) / stddev(close, 20) ──
        dev = close - vwap
        std_20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        std_20 = np.nan_to_num(std_20, nan=1e10)
        std_20 = np.clip(std_20, 1e-10, None)
        zscore = dev / std_20
        zscore = np.nan_to_num(zscore, nan=0.0)

        # ── Relative volume: volume / SMA(volume, 10) ──
        avg_vol = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        rel_vol = volume / avg_vol

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            (close < vwap) & (zscore < -zs_thresh) & (rel_vol > rel_vol_thresh) & time_ok
        )
        short_entry = (
            (close > vwap) & (zscore > zs_thresh) & (rel_vol > rel_vol_thresh) & time_ok
        )

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_atr_mult=1.5,
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=120,
        )
