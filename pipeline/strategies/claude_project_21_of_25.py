"""MACD Histogram Divergence — claude_project_21_of_25

Thesis: When price makes a new low but the MACD histogram is higher than
its prior trough (bullish divergence), it signals weakening downward
momentum and a pending reversal. Mirror logic for bearish divergence.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_ema(arr, period):
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    alpha = 2.0 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1.0 - alpha) * ema[i - 1]
    return ema


class Strategy(BaseStrategy):
    name = "macd_histogram_divergence_v1"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("target_pct", default=0.004, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.006),
            TunableParam("breakeven_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        target_pct = params.get("target_pct", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.003)
        breakeven_pct = params.get("breakeven_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── MACD(12, 26, 9) ──
        ema12 = _compute_ema(close, 12)
        ema26 = _compute_ema(close, 26)
        macd_line = ema12 - ema26
        signal_line = _compute_ema(macd_line, 9)
        macd_hist = macd_line - signal_line

        # ── Rolling min/max over 20 bars for price and histogram ──
        lookback = 20
        price_rolling_min = np.full(n, np.inf, dtype=np.float64)
        price_rolling_max = np.full(n, -np.inf, dtype=np.float64)
        hist_rolling_min = np.full(n, np.inf, dtype=np.float64)
        hist_rolling_max = np.full(n, -np.inf, dtype=np.float64)

        for i in range(lookback, n):
            window_start = i - lookback
            price_rolling_min[i] = np.min(close[window_start:i])
            price_rolling_max[i] = np.max(close[window_start:i])
            hist_rolling_min[i] = np.min(macd_hist[window_start:i])
            hist_rolling_max[i] = np.max(macd_hist[window_start:i])

        # ── Bullish divergence: price makes new low but histogram is higher ──
        bullish_div = (close < price_rolling_min) & (macd_hist > hist_rolling_min)
        # ── Bearish divergence: price makes new high but histogram is lower ──
        bearish_div = (close > price_rolling_max) & (macd_hist < hist_rolling_max)

        # ── Entry ──
        long_entry = bullish_div & (close < vwap)
        short_entry = bearish_div & (close > vwap)

        # ── Signal exit: VWAP touch ──
        signal_exit_long = close >= vwap
        signal_exit_short = close <= vwap

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=vwap.copy(),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            breakeven_pct=breakeven_pct,
            time_stop_bars=60,
        )
