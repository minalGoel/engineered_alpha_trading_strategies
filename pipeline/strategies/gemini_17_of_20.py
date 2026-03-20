"""VIX Spike Mean Reversion — gemini_17_of_20

Thesis: When VIX spikes sharply (fear), oversold stocks near VWAP support
tend to revert. Combine VIX spike detection with RSI extremes and VWAP
for high-probability mean reversion entries.
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
    name = "vix_spike_mean_reversion"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 920     # 15:20
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_spike_pct", default=5.0, low=2.0, high=10.0),
            TunableParam("rsi_long_thresh", default=30.0, low=15.0, high=40.0),
            TunableParam("rsi_short_thresh", default=70.0, low=60.0, high=85.0),
            TunableParam("target_pct", default=0.6, low=0.3, high=1.5),
            TunableParam("stop_pct", default=0.3, low=0.1, high=0.6),
            TunableParam("vix_min", default=15.0, low=10.0, high=20.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_spike_pct = params.get("vix_spike_pct", 5.0)
        rsi_long = params.get("rsi_long_thresh", 30.0)
        rsi_short = params.get("rsi_short_thresh", 70.0)
        target_pct = params.get("target_pct", 0.6)
        stop_pct = params.get("stop_pct", 0.3)
        vix_min = params.get("vix_min", 15.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)

        atr14 = _compute_atr(high, low, close, 14)
        rsi14 = _compute_rsi(close, 14)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=close[0] if n > 0 else 0.0)

        # VIX spike: VIX up > spike_pct% over 10 bars
        vix_change = np.zeros(n, dtype=np.float64)
        lookback = 10
        for i in range(lookback, n):
            if abs(vix[i - lookback]) > 1e-10:
                vix_change[i] = (vix[i] - vix[i - lookback]) / vix[i - lookback] * 100.0
        vix_spike = vix_change > vix_spike_pct

        vix_ok = vix > vix_min

        # Long: VIX spike + RSI < 30 + close < VWAP
        long_entry = vix_spike & (rsi14 < rsi_long) & (close < vwap) & vix_ok

        # Short: VIX spike + RSI > 70 + close > VWAP
        short_entry = vix_spike & (rsi14 > rsi_short) & (close > vwap) & vix_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct / 100.0,
            stop_loss_pct=stop_pct / 100.0,
            time_stop_bars=30,
        )
