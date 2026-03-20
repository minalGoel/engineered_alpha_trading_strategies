"""Order Flow Imbalance — Opus_35

Thesis: Volume delta (volume * sign(close - open)) z-score measures
aggressive buying/selling. Extreme z-scores predict short-term continuation.
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
    name = "opus_order_flow_imbalance_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=2.0, low=1.5, high=3.0),
            TunableParam("target_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("stop_loss_pct", default=0.0015, low=0.001, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 2.0)
        target_pct = params.get("target_pct", 0.002)
        stop_pct = params.get("stop_loss_pct", 0.0015)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Volume delta: volume * sign(close - open) ──
        bar_delta = volume * np.sign(close - open_)

        # ── Cumulative delta over 10 bars (rolling sum) ──
        cum_delta_10 = np.zeros(n, dtype=np.float64)
        for i in range(9, n):
            cum_delta_10[i] = np.sum(bar_delta[i - 9:i + 1])

        # ── Z-score of cum_delta_10 over 60 bars ──
        delta_zscore = np.zeros(n, dtype=np.float64)
        for i in range(60, n):
            window = cum_delta_10[i - 59:i + 1]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma > 1e-10:
                delta_zscore[i] = (cum_delta_10[i] - mu) / sigma

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Filters ──
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entries ──
        long_entry = (delta_zscore > zs_thresh) & (close > vwap) & time_ok
        short_entry = (delta_zscore < -zs_thresh) & (close < vwap) & time_ok

        # ── Signal exit: delta_zscore crosses 0 ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if delta_zscore[i] <= 0 and delta_zscore[i - 1] > 0:
                sig_exit_long[i] = True
            if delta_zscore[i] >= 0 and delta_zscore[i - 1] < 0:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=20,
        )
