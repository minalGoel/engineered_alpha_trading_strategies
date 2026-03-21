"""
momentum_ignition_v1 — NIFTY range-breakout stop-cascade strategy.

When NIFTY consolidates for ~20 minutes in a tight range, bracket-order traders
and mean-reversion algos accumulate positions near the boundaries with stops clustered
just outside. A breakout bar with 2.5x+ volume surge signals institutional order flow
overwhelming the range. Sequential stop-loss cascades push NIFTY 20-35 additional
spot points in 30-90 seconds. We enter on the ignition bar at 5-second resolution,
before the full cascade is priced in.

Converted from: trading_strategies/unique_strategies_all/Strategy_29.json
Original: equity FnO range-breakout, 30-bar 1-min range, 10-30 min hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam

RANGE_BARS = 240   # 20-minute range lookback (20 min × 12 bars/min = 240 bars)
VOL_BARS = 60      # 5-minute volume baseline (5 min × 12 bars/min = 60 bars)


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling maximum over window bars (result[i] = max of arr[i-window : i])."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = np.max(arr[i - window:i])
    return out


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum over window bars (result[i] = min of arr[i-window : i])."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = np.min(arr[i - window:i])
    return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean over window bars (result[i] = mean of arr[i-window : i])."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window, n):
        out[i] = np.mean(arr[i - window:i])
    return out


class Strategy(BaseStrategy):
    name = "momentum_ignition_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min of open noise
    session_end_minutes = 915     # 15:15 IST — no new breakout entries near close
    max_trades_per_day = 6
    max_lookback = 300            # 25-min warmup (240 range + 60 vol avg)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_surge_threshold", 2.5, 1.5, 4.0),
            TunableParam("range_width_min_pct", 0.10, 0.05, 0.20),
            TunableParam("range_width_max_pct", 0.40, 0.25, 0.60),
            TunableParam("strong_close_threshold", 0.70, 0.60, 0.85),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract and forward-fill spot arrays ──────────────────────────────
        close = (
            spot_df["close"]
            .fill_nan(None)
            .fill_null(strategy="forward")
            .to_numpy()
        )
        high = (
            spot_df["high"]
            .fill_nan(None)
            .fill_null(strategy="forward")
            .to_numpy()
        )
        low = (
            spot_df["low"]
            .fill_nan(None)
            .fill_null(strategy="forward")
            .to_numpy()
        )
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        vol_surge_threshold = params.get("vol_surge_threshold", 2.5)
        range_width_min = params.get("range_width_min_pct", 0.10)
        range_width_max = params.get("range_width_max_pct", 0.40)
        strong_close_thr = params.get("strong_close_threshold", 0.70)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── 20-minute rolling range (prior bars only — no lookahead) ──────────
        # range_high[i] = max of high over bars [i-240, i)  (bar i excluded)
        range_high = _rolling_max(high, RANGE_BARS)
        range_low = _rolling_min(low, RANGE_BARS)

        # Range width as % of current close
        with np.errstate(invalid="ignore", divide="ignore"):
            range_width_pct = np.where(
                np.isnan(range_high) | (close == 0),
                np.nan,
                (range_high - range_low) / close * 100.0,
            )

        # Meaningful consolidation band for NIFTY
        consolidated = (
            ~np.isnan(range_width_pct)
            & (range_width_pct >= range_width_min)
            & (range_width_pct <= range_width_max)
        )

        # ── 5-minute volume baseline ───────────────────────────────────────────
        vol_avg = _rolling_mean(volume, VOL_BARS)
        vol_surge = np.where(
            np.isnan(vol_avg) | (vol_avg == 0),
            0.0,
            volume / vol_avg,
        )

        # ── Breakout detection (current close vs prior bar's range boundary) ──
        # Shift range_high / range_low forward by 1 to use previous bar's range
        prev_range_high = np.empty(n)
        prev_range_high[0] = np.nan
        prev_range_high[1:] = range_high[:-1]

        prev_range_low = np.empty(n)
        prev_range_low[0] = np.nan
        prev_range_low[1:] = range_low[:-1]

        breaks_high = ~np.isnan(prev_range_high) & (close > prev_range_high)
        breaks_low = ~np.isnan(prev_range_low) & (close < prev_range_low)

        # ── Strong-close filter ────────────────────────────────────────────────
        bar_range = high - low
        bar_range_safe = np.where(bar_range > 0, bar_range, 1.0)
        close_position = (close - low) / bar_range_safe  # 0=at low, 1=at high

        strong_bull_close = close_position >= strong_close_thr
        strong_bear_close = close_position <= (1.0 - strong_close_thr)

        # ── VIX regime filter ─────────────────────────────────────────────────
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

        vix_ok = (vix_close >= 12.0) & (vix_close <= 22.0)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        base_filter = (
            in_session
            & consolidated
            & (vol_surge >= vol_surge_threshold)
            & vix_ok
        )

        buy_ce = base_filter & breaks_high & strong_bull_close
        buy_pe = base_filter & breaks_low & strong_bear_close

        # Mutual exclusion — both cannot fire on same bar (edge case)
        both = buy_ce & buy_pe
        buy_ce = buy_ce & ~both
        buy_pe = buy_pe & ~both

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
