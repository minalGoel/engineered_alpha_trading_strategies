"""Multi-Factor Composite — claude_project_25_of_25

Thesis: Combining multiple uncorrelated factors into a single composite
score provides a more robust signal than any individual indicator.
Extreme composite scores identify high-conviction mean-reversion setups.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "multi_factor_composite_v1"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("composite_thresh", default=1.5, low=0.8, high=2.5),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.01),
            TunableParam("breakeven_pct", default=0.003, low=0.001, high=0.006),
            TunableParam("w_vwap", default=0.30, low=0.15, high=0.45),
            TunableParam("w_rsi", default=0.20, low=0.10, high=0.30),
            TunableParam("w_vol", default=0.15, low=0.05, high=0.25),
            TunableParam("w_index", default=0.20, low=0.10, high=0.30),
            TunableParam("w_vix", default=0.15, low=0.05, high=0.25),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        composite_thresh = params.get("composite_thresh", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.005)
        breakeven_pct = params.get("breakeven_pct", 0.003)
        w_vwap = params.get("w_vwap", 0.30)
        w_rsi = params.get("w_rsi", 0.20)
        w_vol = params.get("w_vol", 0.15)
        w_index = params.get("w_index", 0.20)
        w_vix = params.get("w_vix", 0.15)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)

        volume = np.nan_to_num(volume, nan=1.0)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = np.nan_to_num(vix, nan=20.0)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Factor 1: VWAP z-score (30-bar) ──
        vwap_diff = close - vwap
        vwap_zscore = np.zeros(n, dtype=np.float64)
        zscore_window = 30
        for i in range(zscore_window, n):
            window = vwap_diff[i - zscore_window:i + 1]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma > 1e-10:
                vwap_zscore[i] = (vwap_diff[i] - mu) / sigma

        # ── Factor 2: RSI deviation from 50 (normalized to z-score-like) ──
        rsi = _compute_rsi(close, 14)
        rsi_dev = (rsi - 50.0) / 25.0  # scale: +/-2 range roughly

        # ── Factor 3: Relative volume z-score ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        vol_ratio = volume / avg_vol_20
        vol_zscore = np.zeros(n, dtype=np.float64)
        for i in range(zscore_window, n):
            window = vol_ratio[i - zscore_window:i + 1]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma > 1e-10:
                vol_zscore[i] = (vol_ratio[i] - mu) / sigma

        # ── Factor 4: Index alignment (stock return - index return, 10-bar) ──
        align_period = 10
        index_align = np.zeros(n, dtype=np.float64)
        for i in range(align_period, n):
            if close[i - align_period] > 0 and index_close[i - align_period] > 0:
                stock_ret = (close[i] - close[i - align_period]) / close[i - align_period]
                idx_ret = (index_close[i] - index_close[i - align_period]) / index_close[i - align_period]
                index_align[i] = stock_ret - idx_ret
        # Normalize to z-score
        index_align_z = np.zeros(n, dtype=np.float64)
        for i in range(zscore_window, n):
            window = index_align[i - zscore_window:i + 1]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma > 1e-10:
                index_align_z[i] = (index_align[i] - mu) / sigma

        # ── Factor 5: VIX regime z-score ──
        vix_zscore = np.zeros(n, dtype=np.float64)
        for i in range(zscore_window, n):
            window = vix[i - zscore_window:i + 1]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma > 1e-10:
                vix_zscore[i] = (vix[i] - mu) / sigma

        # ── Composite score ──
        # Normalize weights
        total_w = w_vwap + w_rsi + w_vol + w_index + w_vix
        if total_w < 1e-10:
            total_w = 1.0
        composite = (
            (w_vwap / total_w) * vwap_zscore
            + (w_rsi / total_w) * rsi_dev
            + (w_vol / total_w) * vol_zscore
            + (w_index / total_w) * index_align_z
            + (w_vix / total_w) * vix_zscore
        )

        # ── Session time filter ──
        # AUDIT FIX: missing session time filter caused entries outside session window
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entry ──
        long_entry = (composite < -composite_thresh) & time_ok
        short_entry = (composite > composite_thresh) & time_ok

        # ── Signal exit: composite crosses 0 ──
        signal_exit_long = composite > 0.0
        signal_exit_short = composite < 0.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            breakeven_pct=breakeven_pct,
            time_stop_bars=120,
        )
