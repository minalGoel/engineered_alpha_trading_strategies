"""Round Number Reversal — claude_project_17_of_25

Thesis: Round numbers (multiples of 50 or 100) act as psychological
support/resistance. When price is near a round number with exhausted
momentum and extreme RSI, expect a reversal.
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
    name = "round_number_reversal_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("round_proximity_pct", default=0.0015, low=0.0005, high=0.003),
            TunableParam("rsi_long_thresh", default=35.0, low=20.0, high=45.0),
            TunableParam("rsi_short_thresh", default=65.0, low=55.0, high=80.0),
            TunableParam("target_pct", default=0.003, low=0.001, high=0.006),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("breakeven_pct", default=0.0015, low=0.0005, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        round_prox = params.get("round_proximity_pct", 0.0015)
        rsi_long = params.get("rsi_long_thresh", 35.0)
        rsi_short = params.get("rsi_short_thresh", 65.0)
        target_pct = params.get("target_pct", 0.003)
        stop_pct = params.get("stop_loss_pct", 0.002)
        breakeven_pct = params.get("breakeven_pct", 0.0015)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)

        # ── RSI(7) ──
        rsi = _compute_rsi(close, 7)

        # ── 5-bar momentum ──
        momentum = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            momentum[i] = close[i] - close[i - 5]

        # ── Round number proximity ──
        # Multiples of 50 for prices < 500, multiples of 100 for >= 500
        near_round = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            p = close[i]
            if p <= 0:
                continue
            step = 50.0 if p < 500.0 else 100.0
            nearest_round = round(p / step) * step
            dist_pct = abs(p - nearest_round) / p
            near_round[i] = dist_pct < round_prox

        # ── Entry ──
        long_entry = near_round & (momentum < 0) & (rsi < rsi_long)
        short_entry = near_round & (momentum > 0) & (rsi > rsi_short)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            breakeven_pct=breakeven_pct,
            time_stop_bars=45,
        )
