"""RSI Connors Pullback v46 — 5-second NIFTY index options strategy.

Converts the Connors RSI(2) pullback strategy from NIFTY50 stocks (15-min bars)
to NIFTY index options at 5-second resolution.

Mechanism: On NIFTY, RSI(2) < 20 while close > SMA(120) signals a micro-correction
exhaustion within an intraday uptrend — market makers and VWAP algos step in and
snap price back within 30-60 seconds. Mirror logic for downtrend (RSI(2) > 80).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed RSI. Returns array of same length; first `period` bars are NaN."""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]

    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # Seed with simple mean of first `period` changes
    avg_gain = np.mean(gain[1 : period + 1])
    avg_loss = np.mean(loss[1 : period + 1])

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _compute_sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average. First `period-1` bars are filled with close[i] (neutral)."""
    n = len(close)
    sma = np.empty(n)
    # Warmup: fill with current close so close > sma is never spuriously True
    for i in range(min(period, n)):
        sma[i] = close[i]
    for i in range(period, n):
        sma[i] = np.mean(close[i - period : i])
    return sma


class Strategy(BaseStrategy):
    name = "rsi_connors_pullback_v46"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup for SMA(120)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_oversold", 20.0, 10.0, 30.0),
            TunableParam("rsi_overbought", 80.0, 70.0, 90.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot close (forward-fill NaN in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──
        rsi_oversold = float(params.get("rsi_oversold", 20.0))
        rsi_overbought = float(params.get("rsi_overbought", 80.0))
        stop_pts = float(params.get("stop_pts", 4.0))
        target_pts = float(params.get("target_pts", 6.0))

        # ── SMA(120) — 10-minute intraday trend filter ──
        # Warmup bars (i < 120) are filled with close[i] so no spurious signal fires.
        sma_120 = _compute_sma(close, 120)

        # ── RSI(2) — ultra-short exhaustion detector ──
        # NaN first 2 bars → fill with neutral 50 to suppress signals during warmup.
        rsi_2 = _compute_rsi(close, 2)
        rsi_2 = np.where(np.isnan(rsi_2), 50.0, rsi_2)

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # ── Entry signals ──
        # Buy CE: intraday uptrend + micro-correction exhausted → snap-back long
        buy_ce = in_session & (close > sma_120) & (rsi_2 < rsi_oversold)

        # Buy PE: intraday downtrend + micro-rally exhausted → snap-back short
        buy_pe = in_session & (close < sma_120) & (rsi_2 > rsi_overbought)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
