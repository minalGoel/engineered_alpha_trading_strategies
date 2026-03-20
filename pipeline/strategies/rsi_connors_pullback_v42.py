"""rsi_connors_pullback_v42 — Connors RSI-2 Pullback adapted for NIFTY 5-second options.

Mechanism: When NIFTY is above its 17-minute SMA (institutional flow is net bullish),
a 30-second sharp selloff driving RSI(6) below 15 reflects stop-loss hunting or retail
panic. VWAP-benchmarked algos and index arbitrageurs inject buy flow within 15-45s,
producing a swift mean-reversion bounce. Mirror logic applies on the short side.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average using cumsum. Returns NaN for bars < period."""
    n = len(close)
    sma = np.full(n, np.nan)
    if n < period:
        return sma
    cumsum = np.cumsum(close)
    sma[period - 1:] = (
        cumsum[period - 1:] - np.concatenate([[0.0], cumsum[:n - period]])
    ) / period
    return sma


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI. Returns 50.0 (neutral) for warm-up bars."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    # Seed with simple mean of first period
    avg_gain[period] = np.mean(gains[1: period + 1])
    avg_loss[period] = np.mean(losses[1: period + 1])

    # Wilder's exponential smoothing
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    for i in range(period, n):
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


class Strategy(BaseStrategy):
    """Connors RSI-2 Pullback for NIFTY 5-second index options.

    Entry: RSI(6) extreme (< 15 oversold / > 85 overbought) with SMA(200) trend gate.
    Hold:  15-90 seconds (time_stop_bars=18).
    """

    name = "rsi_connors_pullback_v42"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 240            # 200-bar SMA warmup + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_low",   15.0,  5.0, 25.0),   # oversold threshold
            TunableParam("rsi_high",  85.0, 75.0, 95.0),   # overbought threshold
            TunableParam("stop_pts",   3.0,  2.0,  6.0),   # option premium stop
            TunableParam("target_pts", 5.0,  3.0, 10.0),   # option premium target
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        rsi_low   = float(params.get("rsi_low",   15.0))
        rsi_high  = float(params.get("rsi_high",  85.0))
        stop_pts  = float(params.get("stop_pts",   3.0))
        target_pts = float(params.get("target_pts", 5.0))

        # Indicators on spot data
        sma_200 = _compute_sma(close, 200)
        rsi_6   = _compute_rsi(close, 6)

        # SMA validity mask — no signal until SMA is warmed up
        sma_valid = ~np.isnan(sma_200)
        # Use close as fallback in comparisons to avoid nan propagation
        sma_filled = np.where(sma_valid, sma_200, close)

        # Session filter
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # buy_ce: NIFTY above 17-min SMA (uptrend) + RSI(6) extreme oversold (30s panic)
        buy_ce = in_session & sma_valid & (close > sma_filled) & (rsi_6 < rsi_low)

        # buy_pe: NIFTY below 17-min SMA (downtrend) + RSI(6) extreme overbought (30s exhaustion)
        buy_pe = in_session & sma_valid & (close < sma_filled) & (rsi_6 > rsi_high)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
