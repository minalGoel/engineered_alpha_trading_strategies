"""Connors RSI Extreme v1 — claude_project_12_of_25

Thesis: Extreme short-term RSI(3) readings with streak confirmation
near VWAP signal oversold/overbought conditions ripe for reversal.
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
    name = "connors_rsi_extreme_v1"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi3_long_thresh", default=10.0, low=3.0, high=20.0),
            TunableParam("rsi3_short_thresh", default=90.0, low=80.0, high=97.0),
            TunableParam("vwap_band_pct", default=0.005, low=0.002, high=0.015),
            TunableParam("stop_loss_pct", default=0.0035, low=0.002, high=0.006),
            TunableParam("breakeven_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rsi3_long = params.get("rsi3_long_thresh", 10.0)
        rsi3_short = params.get("rsi3_short_thresh", 90.0)
        vwap_band = params.get("vwap_band_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.0035)
        be_pct = params.get("breakeven_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── RSI(3) ──
        rsi3 = _compute_rsi(close, 3)

        # ── Streak count: consecutive up/down bars ──
        streak = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                streak[i] = max(streak[i - 1], 0) + 1
            elif close[i] < close[i - 1]:
                streak[i] = min(streak[i - 1], 0) - 1
            else:
                streak[i] = 0

        # ── Near VWAP ──
        near_vwap_below = (close >= vwap * (1 - vwap_band)) & (close <= vwap * (1 + vwap_band * 0.5))
        near_vwap_above = (close <= vwap * (1 + vwap_band)) & (close >= vwap * (1 - vwap_band * 0.5))

        # ── RSI crosses 50 for signal exit ──
        rsi14 = _compute_rsi(close, 14)
        rsi_cross_50_up = rsi14 > 50
        rsi_cross_50_down = rsi14 < 50

        # ── Entry ──
        long_entry = (rsi3 < rsi3_long) & near_vwap_below & (streak < -1)
        short_entry = (rsi3 > rsi3_short) & near_vwap_above & (streak > 1)

        # ── Signal exit: RSI crosses 50 or VWAP touch (use target indicator) ──
        signal_exit_long = rsi_cross_50_up
        signal_exit_short = rsi_cross_50_down

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=45,
        )
