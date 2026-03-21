"""demark_sequential_v1 — TD Sequential exhaustion counter-trend on NIFTY 5-second bars.

Mechanism: A TD Sequential buy setup (9 consecutive 5s bars each closing below their
4-bar-ago close) captures a 45-second micro-selldown on NIFTY. When the 9-count completes
with DeMark perfection (bar 8 or 9 registers a lower low than bar 6), marginal sellers
are absorbed by resting buy orders — the index snaps back as delta-hedging market makers
and VWAP-benchmarked algos resume upward reference price tracking. Sell setup is symmetric.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed RSI."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gains[1:period + 1])
    avg_loss[period] = np.mean(losses[1:period + 1])

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _compute_td_counts(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute TD Sequential buy and sell counts.

    Buy count: consecutive bars where close < close[4], reset on close >= close[4].
    Sell count: consecutive bars where close > close[4], reset on close <= close[4].
    """
    n = len(close)
    buy_count = np.zeros(n, dtype=np.int32)
    sell_count = np.zeros(n, dtype=np.int32)

    for i in range(4, n):
        if close[i] < close[i - 4]:
            buy_count[i] = buy_count[i - 1] + 1
        # else stays 0

        if close[i] > close[i - 4]:
            sell_count[i] = sell_count[i - 1] + 1
        # else stays 0

    return buy_count, sell_count


def _compute_perfection(
    buy_count: np.ndarray,
    sell_count: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """DeMark perfection rules at count == 9.

    Buy perfection: at bar 9 (index i), low[bar8] or low[bar9] <= low[bar6].
    Within the count of 9: bar9=i, bar8=i-1, bar6=i-3.

    Sell perfection: high[bar8] or high[bar9] >= high[bar6].
    """
    n = len(buy_count)
    buy_perfected = np.zeros(n, dtype=bool)
    sell_perfected = np.zeros(n, dtype=bool)

    for i in range(8, n):
        if buy_count[i] == 9:
            # bar 9=i, bar 8=i-1, bar 6=i-3
            buy_perfected[i] = (low[i] <= low[i - 3]) or (low[i - 1] <= low[i - 3])
        if sell_count[i] == 9:
            sell_perfected[i] = (high[i] >= high[i - 3]) or (high[i - 1] >= high[i - 3])

    return buy_perfected, sell_perfected


class Strategy(BaseStrategy):
    """TD Sequential exhaustion counter-trend strategy on NIFTY 5-second bars.

    Signals on a perfected 9-count TD Sequential setup with RSI confirmation.
    Buy CE on buy setup (9 consecutive closes below close[4] + perfection + RSI oversold).
    Buy PE on sell setup (9 consecutive closes above close[4] + perfection + RSI overbought).
    """

    name = "demark_sequential_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 60             # 5-min warmup (60 × 5s) for RSI stability

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_oversold", 35.0, 25.0, 45.0),
            TunableParam("rsi_overbought", 65.0, 55.0, 75.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        rsi_oversold = params.get("rsi_oversold", 35.0)
        rsi_overbought = params.get("rsi_overbought", 65.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # TD Sequential counts
        buy_count, sell_count = _compute_td_counts(close)

        # DeMark perfection at count == 9
        buy_perfected, sell_perfected = _compute_perfection(buy_count, sell_count, high, low)

        # RSI(12) — 60-second window, matches setup duration + 3-bar buffer
        rsi = _compute_rsi(close, 12)

        # VIX filter: skip high-volatility trending regimes where count extends
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < 22.0

        # Buy CE: buy setup exhaustion (downtrend exhausted, expect bounce)
        buy_ce = (
            in_session
            & vix_ok
            & (buy_count == 9)
            & buy_perfected
            & (rsi < rsi_oversold)
        )

        # Buy PE: sell setup exhaustion (uptrend exhausted, expect pullback)
        buy_pe = (
            in_session
            & vix_ok
            & (sell_count == 9)
            & sell_perfected
            & (rsi > rsi_overbought)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
