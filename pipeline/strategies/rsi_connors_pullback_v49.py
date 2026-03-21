"""rsi_connors_pullback_v49 — Connors RSI-2 Pullback on NIFTY (5-second bars).

Mechanism: On NIFTY, institutional TWAP/VWAP algorithms executing directional mandates
over 15-30 minute windows create persistent micro-trends visible as price holding above
the 15-minute SMA. Within these flows, every 60-120 seconds there is a brief momentum
pause as one algo batch completes — visible as RSI(12) dipping below 45. Delta-hedging
market makers provide structural bids at these exhaustion points, triggering resumption
of the institutional flow. Entering ATM calls at RSI(12) < 45 while close > SMA(180)
exploits this predictable 30-90 second re-acceleration.

Original: Connors RSI-2 strategy on top-50 FNO stocks, 1-min bars, 30-60 min hold.
Adaptation: Index NIFTY, 5s bars, 30-120s hold. SMA(200,1-min)→SMA(180,5s)=15min.
RSI(2,1-min)→RSI(12,5s)=1min (minimum viable RSI at 5s frequency).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_sma(close: np.ndarray, period: int) -> np.ndarray:
    """Rolling simple moving average; NaN for first (period-1) bars."""
    n = len(close)
    sma = np.full(n, np.nan)
    if n < period:
        return sma
    cumsum = np.cumsum(close)
    sma[period - 1] = cumsum[period - 1] / period
    sma[period:] = (cumsum[period:] - cumsum[:-period]) / period
    return sma


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed RSI; neutral 50.0 before warmup completes."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]

    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)

    # Seed with simple average over first period
    avg_gain = np.mean(gain[1 : period + 1])
    avg_loss = np.mean(loss[1 : period + 1])

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    # Wilder smoothing for subsequent bars
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return rsi


class Strategy(BaseStrategy):
    """Connors RSI-2 micro-pullback strategy adapted for NIFTY 5-second bars.

    Buy CE when NIFTY is above its 15-min SMA and RSI(12) dips below 45 (brief
    exhaustion within uptrend). Buy PE when NIFTY is below 15-min SMA and RSI(12)
    spikes above 55 (brief bounce in downtrend).
    """

    name = "rsi_connors_pullback_v49"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip pre-open noise
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 240            # 20 min warmup (covers SMA(180) + RSI(12))

    def tunable_params(self) -> list[TunableParam]:
        return [
            # RSI thresholds only — not indicator periods
            TunableParam("rsi_buy_threshold", 45.0, 35.0, 55.0),
            TunableParam("rsi_sell_threshold", 55.0, 45.0, 65.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before converting) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Indicators on spot data ──
        sma_180 = _compute_sma(close, 180)   # 15-min SMA trend filter
        rsi_12 = _compute_rsi(close, 12)     # 1-min RSI exhaustion signal

        # Replace NaN sma with neutral (use close itself — no trend signal)
        sma_180_safe = np.where(np.isnan(sma_180), close, sma_180)

        # ── Parameters ──
        rsi_buy_thr = float(params.get("rsi_buy_threshold", 45.0))
        rsi_sell_thr = float(params.get("rsi_sell_threshold", 55.0))

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # ── Signals ──
        # buy_ce: uptrend (close > 15-min SMA) + brief RSI exhaustion (RSI dip below threshold)
        buy_ce = in_session & (close > sma_180_safe) & (rsi_12 < rsi_buy_thr)

        # buy_pe: downtrend (close < 15-min SMA) + brief RSI bounce (RSI spike above threshold)
        buy_pe = in_session & (close < sma_180_safe) & (rsi_12 > rsi_sell_thr)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 4 pts = ~8 NIFTY spot pts at delta 0.5.
            # If pullback extends 8 spot pts while above SMA, institutional flow has paused
            # (not pulsed) — exit to avoid riding a deeper correction.
            stop_points=np.full(n, 3),
            # Target: 7 pts = ~14 NIFTY spot pts at delta 0.5.
            # Micro-trend resumptions after RSI(12) exhaustion cover 12-20 spot pts in 60-120s;
            # 14 spot pts captures ~60% of the expected move at 1:1.75 risk-reward.
            target_points=np.full(n, 6),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
