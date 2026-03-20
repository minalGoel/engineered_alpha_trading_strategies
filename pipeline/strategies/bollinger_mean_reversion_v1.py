"""Bollinger Mean Reversion v1 — 5-second NIFTY band-touch reversion.

Mechanism: On NIFTY, every intraday selloff that pushes the index beyond its
5-minute Bollinger lower band (2.5σ) represents statistical exhaustion of the
selling program — stop-loss cascades and TWAP sell algos running out of residual
order. At this exact bar, VWAP-benchmarked institutional buyers begin absorbing
with limit bids, and the 5-second candle closes back above the lower band with a
bullish body (close > open). At 5-second resolution we enter immediately on this
reversal candle rather than waiting for a full 1-minute bar, capturing the first
10-20 spot point bounce (4-6 option points) before mean-reversion algos drive
price back toward the 5-minute SMA. Symmetric logic applies for upper band
rejections.

Original: bollinger_mean_reversion_v1 — NIFTY 200 stocks, 1-min bars,
BB(20, 2.5σ). Hold 5-20 min.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_bb(close: np.ndarray, window: int, sigma: float):
    """Compute Bollinger Band mid, upper, lower with a rolling window.

    Returns arrays of length n; first (window-1) elements are NaN.
    """
    n = len(close)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = close[i - window + 1 : i + 1]
        m = np.mean(w)
        s = np.std(w, ddof=0)
        mid[i] = m
        upper[i] = m + sigma * s
        lower[i] = m - sigma * s
    return mid, upper, lower


class Strategy(BaseStrategy):
    name = "bollinger_mean_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min of opening noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup (2× the 60-bar lookback)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Band width multiplier — 2.5σ selects extreme touches
            TunableParam("sigma_mult", 2.5, 1.8, 3.2),
            # Min band width (% of mid) to exclude squeeze regimes
            TunableParam("min_bb_width_pct", 0.15, 0.05, 0.40),
            # Max pct_b distance from band edge at entry
            TunableParam("pct_b_entry_max", 0.08, 0.02, 0.15),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before numpy conversion
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high_arr = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low_arr = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        open_arr = spot_df["open"].fill_null(strategy="forward").to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()

        sigma_mult = params.get("sigma_mult", 2.5)
        min_bb_width_pct = params.get("min_bb_width_pct", 0.15)
        pct_b_entry_max = params.get("pct_b_entry_max", 0.08)

        # 5-minute Bollinger Bands (60 bars × 5s)
        # Compressed from original 20-min to 5-min: we trade the FAST reversion
        # to the near-term mean, not the slow drift to the 20-min average.
        lookback = 60
        bb_mid, bb_upper, bb_lower = _rolling_bb(close, lookback, sigma_mult)

        # Band width as % of mid — filter out squeeze regimes
        bb_width_pct = np.where(
            bb_mid > 0,
            (bb_upper - bb_lower) / bb_mid * 100.0,
            0.0,
        )

        # %B: position within band [0=lower band, 1=upper band]
        band_range = bb_upper - bb_lower
        safe_range = np.where(band_range > 0, band_range, 1.0)
        pct_b = (close - bb_lower) / safe_range

        # VIX: above 20 bands become unreliable for mean-reversion
        # (institutional flows sustain trends rather than reverting)
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort(
                    "datetime"
                ),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Filters ──────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        has_bb = ~np.isnan(bb_mid)
        band_wide = bb_width_pct > min_bb_width_pct  # exclude squeezes
        vix_ok = vix_close < 20.0                    # exclude high-vol trending regime

        base_filter = in_session & has_bb & band_wide & vix_ok

        # ── Long entry (buy CE) ───────────────────────────────────────────────
        # Low touched or pierced lower band, candle closed back above it (bullish reversal),
        # and close is still near the lower band edge (not already half-way to mean).
        touched_lower = (low_arr <= bb_lower) & (close > bb_lower)
        bullish_bar = close > open_arr
        near_lower = pct_b < pct_b_entry_max

        buy_ce = base_filter & touched_lower & bullish_bar & near_lower

        # ── Short entry (buy PE) ──────────────────────────────────────────────
        # High touched or pierced upper band, candle closed back below it (bearish rejection),
        # and close is still near the upper band edge.
        touched_upper = (high_arr >= bb_upper) & (close < bb_upper)
        bearish_bar = close < open_arr
        near_upper = pct_b > (1.0 - pct_b_entry_max)

        buy_pe = base_filter & touched_upper & bearish_bar & near_upper

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 4 pts — if NIFTY extends 8 more spot pts beyond band, band-walk confirmed
            # Target: 6 pts — 50% reversion of typical 5-min band width = ~12 spot pts = 6 opt pts
            stop_points=np.full(n, 4.0),
            target_points=np.full(n, 6.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,         # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
