"""VWAP Slope Momentum — claude_project_20_of_25

Thesis: The slope of VWAP captures institutional order flow direction.
A z-scored VWAP slope exceeding thresholds identifies strong momentum
regimes where price-above-VWAP alignment confirms trend participation.
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
    name = "vwap_slope_momentum_v1"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("slope_zscore_thresh", default=1.5, low=0.8, high=2.5),
            TunableParam("target_atr_mult", default=1.5, low=0.8, high=3.0),
            TunableParam("trailing_atr_mult", default=1.0, low=0.5, high=2.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        slope_zscore_thresh = params.get("slope_zscore_thresh", 1.5)
        target_atr_mult = params.get("target_atr_mult", 1.5)
        trailing_atr_mult = params.get("trailing_atr_mult", 1.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

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

        # ── VWAP slope: diff over 10 bars ──
        slope_period = 10
        vwap_slope = np.zeros(n, dtype=np.float64)
        for i in range(slope_period, n):
            vwap_slope[i] = vwap[i] - vwap[i - slope_period]

        # ── Z-score of slope over 50 bars ──
        zscore_window = 50
        slope_zscore = np.zeros(n, dtype=np.float64)
        for i in range(zscore_window, n):
            window = vwap_slope[i - zscore_window:i + 1]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma > 1e-10:
                slope_zscore[i] = (vwap_slope[i] - mu) / sigma

        # ── Entry ──
        long_entry = (slope_zscore > slope_zscore_thresh) & (close > vwap)
        short_entry = (slope_zscore < -slope_zscore_thresh) & (close < vwap)

        # ── Signal exit: close crosses VWAP ──
        signal_exit_long = close < vwap
        signal_exit_short = close > vwap

        # ── Trailing stop from ATR ──
        median_atr = np.nanmedian(atr[atr > 0]) if np.any(atr > 0) else 0.0
        median_close = np.nanmedian(close[close > 0]) if np.any(close > 0) else 1.0
        trailing_pct = (trailing_atr_mult * median_atr / median_close) if median_close > 0 else 0.005

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            target_atr_mult=target_atr_mult,
            trailing_stop_pct=trailing_pct,
            trailing_activate_pct=0.0,
            time_stop_bars=120,
        )
