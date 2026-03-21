"""
intraday_rsi_divergence_v1 — RSI Divergence on NIFTY 5-second bars.

Mechanism:
  On NIFTY, when institutional algo selling (or buying) drives the index to a new
  1-minute price extreme, but RSI(18) at that extreme is higher (for a lower low) or
  lower (for a higher high) than it was at the prior 1-minute extreme, the directional
  order flow is decelerating. At 5-second resolution this deceleration is visible before
  the full reversal, giving a 30-90 second lead. Market-maker delta-hedging amplifies
  the initial snap-back as they unwind hedges placed during the exhausted directional leg.

Converted from: trading_strategies/unique_strategies_all/Strategy_26.json
Original: RSI(9) divergence on 1-min FnO stocks, hold 5-30 min.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI. Returns array same length as close, NaN for first `period` bars."""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Initial average (simple mean over first period)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum with NaN for first (window-1) positions."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.min(arr[i - window + 1 : i + 1])
    return out


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling maximum with NaN for first (window-1) positions."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        out[i] = np.max(arr[i - window + 1 : i + 1])
    return out


class Strategy(BaseStrategy):
    name = "intraday_rsi_divergence_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    # RSI(18) + 24-bar swing windows + 12-bar lag = 54 bars; use 72 for safety
    max_lookback = 72

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum RSI divergence gap (RSI units) to qualify as a real divergence
            TunableParam("rsi_div_threshold", 3.0, 1.0, 8.0),
            # RSI level below which bullish divergence is in oversold territory
            TunableParam("rsi_oversold", 45.0, 35.0, 52.0),
            # RSI level above which bearish divergence is in overbought territory
            TunableParam("rsi_overbought", 55.0, 48.0, 65.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Parameters ──────────────────────────────────────────────────────────
        rsi_div_threshold = params.get("rsi_div_threshold", 3.0)
        rsi_oversold      = params.get("rsi_oversold",      45.0)
        rsi_overbought    = params.get("rsi_overbought",    55.0)

        # ── Extract spot arrays (forward-fill NaN in Polars first) ────────────
        close = (
            spot_df.select(pl.col("close").forward_fill())
            ["close"].to_numpy().astype(np.float64)
        )
        time_min = spot_df["time_minutes"].to_numpy()

        # ── RSI(18) — 90-second momentum indicator ───────────────────────────
        rsi18 = _compute_rsi(close, 18)

        # ── Swing windows: 12 bars = 1 minute each ───────────────────────────
        # Recent window  [bar-11 .. bar]        → current 1-minute extreme
        # Prior window   [bar-23 .. bar-12]     → prior 1-minute extreme
        low_recent  = _rolling_min(close, 12)          # min of last 12 bars
        high_recent = _rolling_max(close, 12)

        # Prior window: shift low_recent/high_recent by 12 bars
        low_prior  = np.full(n, np.nan)
        high_prior = np.full(n, np.nan)
        low_prior[12:]  = low_recent[:-12]
        high_prior[12:] = high_recent[:-12]

        # RSI 12 bars ago — compare momentum at recent vs prior swing
        rsi_lagged = np.full(n, np.nan)
        rsi_lagged[12:] = rsi18[:-12]

        # ── Replace NaN with neutral values ──────────────────────────────────
        rsi18      = np.where(np.isnan(rsi18),      50.0, rsi18)
        rsi_lagged = np.where(np.isnan(rsi_lagged), 50.0, rsi_lagged)
        # NaN in swing windows → no signal (use inf/-inf so conditions fail)
        low_recent  = np.where(np.isnan(low_recent),  np.inf,  low_recent)
        high_recent = np.where(np.isnan(high_recent), -np.inf, high_recent)
        low_prior   = np.where(np.isnan(low_prior),   -np.inf, low_prior)
        high_prior  = np.where(np.isnan(high_prior),  np.inf,  high_prior)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Bullish divergence → buy CE ───────────────────────────────────────
        # Price lower low in last 1 min vs prior 1 min; RSI higher at that low
        bullish_div = (
            in_session
            & (low_recent  < low_prior)                         # price: lower low
            & (rsi18 > rsi_lagged + rsi_div_threshold)          # RSI: higher low
            & (rsi18 < rsi_oversold)                            # oversold context
            & (close > low_recent)                              # price bouncing
        )

        # ── Bearish divergence → buy PE ───────────────────────────────────────
        # Price higher high in last 1 min vs prior 1 min; RSI lower at that high
        bearish_div = (
            in_session
            & (high_recent > high_prior)                        # price: higher high
            & (rsi18 < rsi_lagged - rsi_div_threshold)          # RSI: lower high
            & (rsi18 > rsi_overbought)                          # overbought context
            & (close < high_recent)                             # price retreating
        )

        # ── Build output arrays ───────────────────────────────────────────────
        buy_ce = bullish_div.astype(bool)
        buy_pe = bearish_div.astype(bool)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # stop: 3 pts = ~6 NIFTY spot pts; if violated in 30s, thesis wrong
            stop_points=np.full(n, 3.0),
            # target: 5 pts = ~10 NIFTY spot pts; lower half of typical snap-back
            target_points=np.full(n, 5.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
