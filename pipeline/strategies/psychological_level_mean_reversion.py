"""Psychological Level Mean Reversion — 5-second NIFTY index options.

Thesis: NIFTY participants cluster stop-loss and limit orders at round 100-multiple
levels (24000, 24100, etc.). When price briefly pierces below (or above) such a level,
stop-hunt selling (or buying) is quickly absorbed by resting limit orders, causing a
30-90 second snapback. Entry when price is within pierce_threshold_pct of the level AND
3-minute RSI confirms short-term exhaustion.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothed RSI. Returns 50.0 for warmup bars."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.mean(gains[1:period + 1])
    avg_loss = np.mean(losses[1:period + 1])

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


class Strategy(BaseStrategy):
    name = "psychological_level_mean_reversion"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip opening auction noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 72             # 6 min warmup: enough for RSI(36) to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Percentage distance from nearest 100-multiple that defines "near the level"
            # Default 0.0006 ≈ 14-15 NIFTY pts at 24000 — tight enough to filter noise
            TunableParam("pierce_threshold_pct", 0.0006, 0.0003, 0.0015),
            # RSI thresholds for exhaustion detection
            TunableParam("rsi_oversold", 35.0, 25.0, 45.0),
            TunableParam("rsi_overbought", 65.0, 55.0, 75.0),
            # Stop/target in option premium points
            TunableParam("stop_pts", 4.0, 3.0, 7.0),
            TunableParam("target_pts", 7.0, 5.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before numpy conversion to handle any mid-session NaN gaps
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        pierce_pct = params.get("pierce_threshold_pct", 0.0006)
        rsi_oversold = params.get("rsi_oversold", 35.0)
        rsi_overbought = params.get("rsi_overbought", 65.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # Nearest 100-multiple psychological level (e.g. 24000, 24100, 23900)
        safe_close = np.where(close > 0, close, 1.0)
        round_level = np.round(close / 100.0) * 100.0
        dist_signed = close - round_level       # + = above level, - = below level
        dist_pct = np.abs(dist_signed) / safe_close

        # Within pierce_threshold_pct of the round level
        near_level = dist_pct < pierce_pct

        # RSI(36) = 3-minute exhaustion signal on spot index
        # 36 bars chosen as TRIGGER window to detect near-term exhaustion;
        # NOT mechanically scaled from original RSI(14) on 1-min bars.
        rsi = _compute_rsi(close, 36)

        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # Buy CE: price just below 100-multiple + RSI oversold → snapback up
        buy_ce = in_session & near_level & (dist_signed < 0.0) & (rsi < rsi_oversold)

        # Buy PE: price just above 100-multiple + RSI overbought → snapback down
        buy_pe = in_session & near_level & (dist_signed > 0.0) & (rsi > rsi_overbought)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
