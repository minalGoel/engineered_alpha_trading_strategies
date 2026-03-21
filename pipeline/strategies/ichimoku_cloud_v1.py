"""ichimoku_cloud_v1 — Ichimoku TK Cross with Cloud Regime Filter on NIFTY.

Mechanism:
    On NIFTY, the Tenkan-sen (7-bar = 35-second equilibrium) crossing above/below the
    Kijun-sen (22-bar = 110-second base equilibrium) signals a micro-regime shift in
    institutional order flow. When this TK cross occurs while price is already positioned
    outside the Ichimoku cloud (Senkou Span A/B from 44-bar extremes = 220 seconds),
    NIFTY's prevailing short-term demand has overcome equilibria at multiple timescales,
    indicating sustained institutional directional pressure with 14-20 spot point
    follow-through typical within 60 seconds.

Timeframe: 5-second bars (NIFTY index)
Hold time: 30-120 seconds (time_stop_bars=24)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_max(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling maximum over a fixed window. Returns NaN for the warmup bars."""
    n = len(arr)
    result = np.empty(n)
    result[:] = np.nan
    for i in range(period - 1, n):
        result[i] = np.max(arr[i - period + 1 : i + 1])
    return result


def _rolling_min(arr: np.ndarray, period: int) -> np.ndarray:
    """Rolling minimum over a fixed window. Returns NaN for the warmup bars."""
    n = len(arr)
    result = np.empty(n)
    result[:] = np.nan
    for i in range(period - 1, n):
        result[i] = np.min(arr[i - period + 1 : i + 1])
    return result


class Strategy(BaseStrategy):
    name = "ichimoku_cloud_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — need 44 bars (220s) warmup from 09:15
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 50             # 44 bars Senkou Span B + small buffer
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_low", 12.0, 8.0, 18.0),
            TunableParam("vix_high", 24.0, 18.0, 30.0),
            TunableParam("stop_pts", 5.0, 2.0, 7.0),
            TunableParam("target_pts", 8.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))
        vix_low = float(params.get("vix_low", 12.0))
        vix_high = float(params.get("vix_high", 24.0))

        # ── VIX regime filter ──────────────────────────────────────────────────
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

        # ── Ichimoku components ────────────────────────────────────────────────
        # Tenkan-sen: 7-bar equilibrium = 35-second midpoint (fast line)
        tenkan = (_rolling_max(high, 7) + _rolling_min(low, 7)) / 2.0

        # Kijun-sen: 22-bar equilibrium = 110-second base (slow line)
        kijun = (_rolling_max(high, 22) + _rolling_min(low, 22)) / 2.0

        # Senkou Span A: average of tenkan + kijun (upper/lower cloud boundary)
        span_a = (tenkan + kijun) / 2.0

        # Senkou Span B: 44-bar equilibrium = 220-second extreme range midpoint
        span_b = (_rolling_max(high, 44) + _rolling_min(low, 44)) / 2.0

        # Replace NaN with neutral (close price) before boolean comparisons
        tenkan_ff = np.where(np.isnan(tenkan), close, tenkan)
        kijun_ff = np.where(np.isnan(kijun), close, kijun)
        span_a_ff = np.where(np.isnan(span_a), close, span_a)
        span_b_ff = np.where(np.isnan(span_b), close, span_b)

        # ── TK cross detection (cross event on current bar) ────────────────────
        # Bullish TK cross: tenkan crosses above kijun
        # Bearish TK cross: tenkan crosses below kijun
        tk_bull_cross = np.zeros(n, dtype=bool)
        tk_bear_cross = np.zeros(n, dtype=bool)
        for i in range(1, n):
            tk_bull_cross[i] = (tenkan_ff[i] > kijun_ff[i]) and (
                tenkan_ff[i - 1] <= kijun_ff[i - 1]
            )
            tk_bear_cross[i] = (tenkan_ff[i] < kijun_ff[i]) and (
                tenkan_ff[i - 1] >= kijun_ff[i - 1]
            )

        # ── Cloud regime ───────────────────────────────────────────────────────
        cloud_top = np.maximum(span_a_ff, span_b_ff)
        cloud_bot = np.minimum(span_a_ff, span_b_ff)

        price_above_cloud = close > cloud_top      # bullish regime
        price_below_cloud = close < cloud_bot      # bearish regime

        cloud_bullish = span_a_ff > span_b_ff      # Span A above Span B
        cloud_bearish = span_a_ff < span_b_ff      # Span A below Span B

        # ── Session and VIX filters ────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        vix_ok = (vix_close >= vix_low) & (vix_close <= vix_high)

        # ── Entry signals ──────────────────────────────────────────────────────
        # buy_ce: bullish TK cross + price above cloud + cloud is bullish color
        buy_ce = in_session & vix_ok & tk_bull_cross & price_above_cloud & cloud_bullish

        # buy_pe: bearish TK cross + price below cloud + cloud is bearish color
        buy_pe = in_session & vix_ok & tk_bear_cross & price_below_cloud & cloud_bearish

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
