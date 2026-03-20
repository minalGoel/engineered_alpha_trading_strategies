"""Stochastic Reversion v1 — NIFTY ATM options, 5-second bars.

Original: Stochastic(14,3,3) crossover reversion on NIFTY200 stocks, 1-min bars,
5-25 min hold.

Adaptation: Compressed to 70-second stochastic (14 × 5s bars) to detect fast
intraday range extremes on NIFTY index. Enter ATM call on bullish cross from
oversold; ATM put on bearish cross from overbought. VWAP proximity + extended-extreme
filter to avoid trending conditions.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    k_period: int = 14,
    smooth: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute smoothed Stochastic %K and %D.

    %K = SMA(raw_k, smooth) where raw_k = (close - lowest_low) / (highest_high - lowest_low) * 100
    %D = SMA(%K, smooth)
    """
    n = len(close)
    raw_k = np.full(n, 50.0)

    for i in range(k_period - 1, n):
        h = np.max(high[i - k_period + 1 : i + 1])
        lo = np.min(low[i - k_period + 1 : i + 1])
        denom = h - lo
        raw_k[i] = (close[i] - lo) / denom * 100.0 if denom > 0.0 else 50.0

    # Smooth %K with SMA(smooth)
    stoch_k = np.full(n, 50.0)
    for i in range(smooth - 1, n):
        stoch_k[i] = np.mean(raw_k[i - smooth + 1 : i + 1])

    # %D = SMA of smoothed %K
    stoch_d = np.full(n, 50.0)
    for i in range(smooth - 1, n):
        stoch_d[i] = np.mean(stoch_k[i - smooth + 1 : i + 1])

    return stoch_k, stoch_d


def _compute_session_vwap(
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Compute per-session VWAP (cumulative from session open each day)."""
    n = len(close)
    vwap = close.copy()
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -1

    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        v = float(volume[i])
        cum_pv += close[i] * v
        cum_v += v
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]

    return vwap


def _count_consecutive_extreme(condition_arr: np.ndarray) -> np.ndarray:
    """For each bar i, count how many consecutive bars ending at i-1 had condition True."""
    n = len(condition_arr)
    counts = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        counts[i] = (counts[i - 1] + 1) if condition_arr[i - 1] else 0
    return counts


class Strategy(BaseStrategy):
    name = "stochastic_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip opening noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 60             # 14 bars stochastic + 6 bars smoothing + buffer = ~5 min

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("oversold_threshold", 20.0, 10.0, 30.0),
            TunableParam("overbought_threshold", 80.0, 70.0, 90.0),
            TunableParam("vwap_proximity", 0.005, 0.002, 0.015),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Raw price arrays (forward-fill NaN in Polars first) ──────────────
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ───────────────────────────────────────────────────────
        oversold = params.get("oversold_threshold", 20.0)
        overbought = params.get("overbought_threshold", 80.0)
        vwap_prox = params.get("vwap_proximity", 0.005)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Stochastic %K and %D (14-bar, 3-bar smoothed) ───────────────────
        # 14 × 5s = 70 seconds — detects exhaustion at the fast intraday extreme
        stoch_k, stoch_d = _compute_stochastic(high, low, close, k_period=14, smooth=3)

        # ── Session VWAP ─────────────────────────────────────────────────────
        vwap = _compute_session_vwap(close, volume, day_id)

        # ── VIX filter ───────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Cross detection ───────────────────────────────────────────────────
        prev_k = np.empty(n)
        prev_d = np.empty(n)
        prev_k[0] = 50.0
        prev_d[0] = 50.0
        prev_k[1:] = stoch_k[:-1]
        prev_d[1:] = stoch_d[:-1]

        bullish_cross = (stoch_k > stoch_d) & (prev_k <= prev_d)
        bearish_cross = (stoch_k < stoch_d) & (prev_k >= prev_d)

        # ── Extended extreme filter ──────────────────────────────────────────
        # Skip if stochastic has been in extreme zone for >= 30 bars (2.5 min)
        # — signal indicates trending, not exhaustion
        was_oversold = stoch_k < oversold
        was_overbought = stoch_k > overbought
        bars_in_oversold = _count_consecutive_extreme(was_oversold)
        bars_in_overbought = _count_consecutive_extreme(was_overbought)

        # ── Composite filters ─────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        low_vix = vix_close < 25.0

        # ── Buy CE: bullish cross from oversold ──────────────────────────────
        in_oversold = (stoch_k < oversold) & (stoch_d < oversold)
        near_vwap_down = close > vwap * (1.0 - vwap_prox)   # not too far below VWAP
        bullish_bar = close > open_
        not_trending_down = bars_in_oversold < 30

        buy_ce = (
            in_session
            & low_vix
            & bullish_cross
            & in_oversold
            & near_vwap_down
            & bullish_bar
            & not_trending_down
        )

        # ── Buy PE: bearish cross from overbought ────────────────────────────
        in_overbought = (stoch_k > overbought) & (stoch_d > overbought)
        near_vwap_up = close < vwap * (1.0 + vwap_prox)     # not too far above VWAP
        bearish_bar = close < open_
        not_trending_up = bars_in_overbought < 30

        buy_pe = (
            in_session
            & low_vix
            & bearish_cross
            & in_overbought
            & near_vwap_up
            & bearish_bar
            & not_trending_up
        )

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
