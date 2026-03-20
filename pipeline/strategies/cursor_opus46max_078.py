"""Vol-Adjusted Mean Reversion v1 — cursor_opus46max_078

Thesis: Dynamically scale mean-reversion z-score threshold based on
the ratio of 30-min to 120-min realized vol. Low vol ratio => tighter
threshold (1.5 sigma); high vol ratio => wider threshold (2.5+ sigma).
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
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss > 0:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss > 0:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i + 1] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_078"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("base_threshold", default=1.5, low=1.0, high=2.0),
            TunableParam("vol_ratio_coeff", default=1.0, low=0.5, high=2.0),
            TunableParam("rsi_long_thresh", default=30.0, low=20.0, high=40.0),
            TunableParam("rsi_short_thresh", default=70.0, low=60.0, high=80.0),
            TunableParam("vol_ratio_max", default=1.5, low=1.2, high=2.0),
            TunableParam("vix_max", default=22.0, low=18.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        base_thresh = params.get("base_threshold", 1.5)
        vol_coeff = params.get("vol_ratio_coeff", 1.0)
        rsi_long = params.get("rsi_long_thresh", 30.0)
        rsi_short = params.get("rsi_short_thresh", 70.0)
        vol_ratio_max = params.get("vol_ratio_max", 1.5)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Returns ──
        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0:
                returns[i] = (close[i] - close[i - 1]) / close[i - 1]

        # ── Vol ratio: rolling_std(returns, 30) / rolling_std(returns, 120) ──
        vol_30 = np.zeros(n, dtype=np.float64)
        vol_120 = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            vol_30[i] = np.std(returns[i - 30:i])
        for i in range(120, n):
            vol_120[i] = np.std(returns[i - 120:i])
        vol_ratio = np.where(vol_120 > 0, vol_30 / vol_120, 1.0)

        # ── Price z-score: (close - SMA60) / rolling_std(close, 60) ──
        sma60 = np.zeros(n, dtype=np.float64)
        std60 = np.zeros(n, dtype=np.float64)
        for i in range(60, n):
            sma60[i] = np.mean(close[i - 60:i])
            std60[i] = np.std(close[i - 60:i])
        price_z = np.where(std60 > 0, (close - sma60) / std60, 0.0)

        # ── Adaptive threshold ──
        adaptive_thresh = base_thresh + vol_coeff * vol_ratio

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        vix_ok = vix < vix_max
        vol_ok = vol_ratio < vol_ratio_max

        # ── Entry: price_z crosses back toward zero after exceeding threshold ──
        long_entry = (
            (price_z < -adaptive_thresh + 0.2) &
            (price_z > -adaptive_thresh - 1.0) &
            (rsi < rsi_long) &
            (close < vwap) &
            vol_ok &
            vix_ok
        )

        short_entry = (
            (price_z > adaptive_thresh - 0.2) &
            (price_z < adaptive_thresh + 1.0) &
            (rsi > rsi_short) &
            (close > vwap) &
            vol_ok &
            vix_ok
        )

        # ── Signal exit: price_z crosses 0 (mean reversion complete) ──
        signal_exit_long = price_z > 0.0
        signal_exit_short = price_z < 0.0

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=sma60.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            time_stop_bars=60,
        )
