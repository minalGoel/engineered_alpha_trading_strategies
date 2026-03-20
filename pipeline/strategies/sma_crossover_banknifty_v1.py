"""
sma_crossover_banknifty_v1 — Triple SMA Pullback on BANKNIFTY

Mechanism:
    On BANKNIFTY, institutional TWAP/VWAP algorithms working directional orders create
    persistent micro-trends visible as SMA alignment across 3-min, 15-min, and 60-min
    windows. When all three align (SMA36 > SMA180 > SMA720 for bullish, reverse for
    bearish) AND price retouches the fast 3-min SMA after a brief pullback, continuous
    demand/supply absorption by institutional desks is confirmed. We enter on that
    pullback-to-SMA and ride the next 30-50 spot point continuation leg.

Original: Triple SMA(10/29/100) on BANKNIFTY 1-min bars, hold 5-20 min.
Converted to: SMA(36/180/720) on 5s bars with pullback-to-SMA36 entry filter, hold 30-120s.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average; returns NaN for first (period-1) bars."""
    out = np.full(len(arr), np.nan)
    cumsum = np.cumsum(arr)
    out[period - 1] = cumsum[period - 1] / period
    for i in range(period, len(arr)):
        out[i] = (cumsum[i] - cumsum[i - period]) / period
    return out


class Strategy(BaseStrategy):
    name = "sma_crossover_banknifty_v1"
    underlying = "BANKNIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 5
    max_lookback = 720            # 60-min warmup — longest SMA needs 720 bars

    def tunable_params(self) -> list[TunableParam]:
        return [
            # How close to SMA36 price must be for a pullback entry
            # 0.0005 = 0.05% ~ 25 BANKNIFTY points at 50000 level
            TunableParam("pullback_threshold", 0.0005, 0.0002, 0.0015),
            # Minimum SMA separation: fast - mid must exceed this fraction
            # Prevents trading nearly-flat SMA stacks
            TunableParam("sma_separation", 0.0002, 0.0001, 0.0010),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        pullback_threshold = params.get("pullback_threshold", 0.0005)
        sma_separation = params.get("sma_separation", 0.0002)

        # --- Extract spot close and time ---
        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy(allow_copy=True).astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy(allow_copy=True)

        # --- Compute SMAs ---
        sma36 = _sma(close, 36)    # 3-min fast — trade trigger
        sma180 = _sma(close, 180)  # 15-min mid  — trend context
        sma720 = _sma(close, 720)  # 60-min slow — regime filter

        # Replace leading NaN with 0 for safe boolean comparisons
        sma36_safe = np.where(np.isnan(sma36), 0.0, sma36)
        sma180_safe = np.where(np.isnan(sma180), 0.0, sma180)
        sma720_safe = np.where(np.isnan(sma720), 0.0, sma720)

        # --- Session filter ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # --- Warmup mask: require all SMAs to be valid ---
        all_valid = (~np.isnan(sma36)) & (~np.isnan(sma180)) & (~np.isnan(sma720))

        # --- Bullish SMA stack ---
        # sma36 > sma180 > sma720, with minimum separation to avoid flat stacks
        bull_stack = (
            (sma36_safe > sma180_safe + sma180_safe * sma_separation) &
            (sma180_safe > sma720_safe + sma720_safe * sma_separation)
        )

        # --- Bearish SMA stack ---
        bear_stack = (
            (sma36_safe < sma180_safe - sma180_safe * sma_separation) &
            (sma180_safe < sma720_safe - sma720_safe * sma_separation)
        )

        # --- Pullback-to-SMA36 condition ---
        # Price within pullback_threshold fraction of sma36
        near_sma36 = np.abs(close - sma36_safe) / np.where(sma36_safe > 0, sma36_safe, 1.0) < pullback_threshold

        # Price above sma36 (still in uptrend, just touching support)
        price_above_sma36 = close > sma36_safe

        # Price below sma36 (still in downtrend, just touching resistance)
        price_below_sma36 = close < sma36_safe

        # --- Entry signals ---
        buy_ce = in_session & all_valid & bull_stack & near_sma36 & price_above_sma36
        buy_pe = in_session & all_valid & bear_stack & near_sma36 & price_below_sma36

        # No signal-based exits — rely on stop/target/time stop
        sell_ce = np.zeros(n, dtype=bool)
        sell_pe = np.zeros(n, dtype=bool)

        # --- Stop/target in option premium points ---
        # Stop: 4 pts (~8 BANKNIFTY spot pts at delta 0.5) — beyond this, micro-trend is broken
        # Target: 7 pts (~14 BANKNIFTY spot pts) — captures first 30-40% of typical 30-50pt continuation
        stop_pts = np.full(n, 4.0)
        target_pts = np.full(n, 7.0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=sell_ce,
            sell_pe=sell_pe,
            stop_points=stop_pts,
            target_points=target_pts,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
