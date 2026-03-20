"""Expiry Day Premium Decay (VWAP Z-Score Reversion) — Opus_37

Thesis: On high-gamma days (weekly expiry), VWAP acts as a strong
attractor. Extreme z-score deviations with declining VIX revert to VWAP.
Implemented as generic VWAP zscore reversion with tighter thresholds.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "opus_expiry_day_premium_decay_v1"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 840     # 14:00
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=1.5, low=1.0, high=2.5),
            TunableParam("stop_atr_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("breakeven_pct", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 1.5)
        stop_atr = params.get("stop_atr_mult", 1.5)
        be_pct = params.get("breakeven_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Z-score of (close - vwap) ──
        dev = close - vwap
        std_30 = df["close"].rolling_std(30).to_numpy().astype(np.float64)
        std_30 = np.nan_to_num(std_30, nan=1e10)
        std_30 = np.clip(std_30, 1e-10, None)
        zscore = dev / std_30
        zscore = np.nan_to_num(zscore, nan=0.0)

        # ── VIX declining (bar-over-bar) ──
        vix_change = np.zeros(n, dtype=np.float64)
        vix_change[1:] = vix[1:] - vix[:-1]
        vix_declining = vix_change < 0

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Filters ──
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entries ──
        long_entry = (zscore < -zs_thresh) & vix_declining & time_ok
        short_entry = (zscore > zs_thresh) & vix_declining & time_ok

        # ── Signal exit: zscore crosses 0 ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if zscore[i] >= 0 and zscore[i - 1] < 0:
                sig_exit_long[i] = True
            if zscore[i] <= 0 and zscore[i - 1] > 0:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            use_target_indicator=True,
            stop_loss_atr_mult=stop_atr,
            breakeven_pct=be_pct,
            time_stop_bars=90,
        )
