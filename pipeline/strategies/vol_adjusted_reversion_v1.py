"""vol_adjusted_reversion_v1 — Adaptive Vol-Scaled Mean Reversion on NIFTY.

Mechanism:
    When NIFTY's 5-min realized vol (last 60 bars) is compressing relative to
    its 30-min baseline (360 bars), VWAP-benchmarked institutional TWAP flow
    dominates and any 10-min z-score deviation rapidly reverts. The adaptive
    threshold (tighter in calm vol regimes, wider in turbulent ones) gates entry
    to only high-quality mean-reversion setups, confirmed by a 3-min RSI extreme.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Simple RSI using Wilder smoothing."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # First average
    avg_gain = np.mean(gain[1:period + 1])
    avg_loss = np.mean(loss[1:period + 1])
    for i in range(period, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation using Welford-style accumulation."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window, n):
        result[i] = np.std(arr[i - window:i], ddof=0)
    return result


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window, n):
        result[i] = np.mean(arr[i - window:i])
    return result


class Strategy(BaseStrategy):
    name = "vol_adjusted_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min; slow vol needs 360 bars (30 min)
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 360            # 30 min warmup for slow vol window

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Base z-score threshold (calm vol regime)
            TunableParam("thresh_base", 1.5, 0.8, 2.2),
            # Additional threshold per unit of vol_ratio
            TunableParam("thresh_scale", 0.8, 0.3, 1.5),
            # Max vol_ratio allowed (rejects trending/expanding vol environments)
            TunableParam("vol_ratio_limit", 1.5, 0.8, 2.5),
            # RSI level for oversold (buy_ce) / overbought (buy_pe = 100 - this)
            TunableParam("rsi_extreme", 35.0, 25.0, 45.0),
            # VIX level above which we skip (trending panic)
            TunableParam("vix_limit", 22.0, 16.0, 30.0),
            # Stop and target in option premium points
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

        # ── Parameters ────────────────────────────────────────────────────────
        thresh_base = params.get("thresh_base", 1.5)
        thresh_scale = params.get("thresh_scale", 0.8)
        vol_ratio_limit = params.get("vol_ratio_limit", 1.5)
        rsi_extreme = params.get("rsi_extreme", 35.0)
        vix_limit = params.get("vix_limit", 22.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Spot data ──────────────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── VIX close (aligned to spot bars) ──────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Log returns for realized vol ───────────────────────────────────────
        log_ret = np.zeros(n)
        log_ret[1:] = np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], 1.0))

        # ── Price z-score: 10-min rolling mean/std (120 bars) ─────────────────
        ZSCORE_WINDOW = 120
        roll_mean = _rolling_mean(close, ZSCORE_WINDOW)
        roll_std = _rolling_std(close, ZSCORE_WINDOW)
        roll_std_safe = np.where(roll_std > 0.01, roll_std, 0.01)
        price_z = (close - roll_mean) / roll_std_safe
        # Mask warmup bars
        price_z[:ZSCORE_WINDOW] = 0.0

        # ── Vol ratio: fast(5-min=60 bars) / slow(30-min=360 bars) ───────────
        VOL_FAST = 60
        VOL_SLOW = 360
        vol_fast = _rolling_std(log_ret, VOL_FAST)
        vol_slow = _rolling_std(log_ret, VOL_SLOW)
        vol_slow_safe = np.where(vol_slow > 1e-8, vol_slow, 1e-8)
        vol_ratio = vol_fast / vol_slow_safe
        # Mask warmup
        vol_ratio[:VOL_SLOW] = 1.0

        # ── Adaptive threshold ─────────────────────────────────────────────────
        adaptive_thresh = thresh_base + thresh_scale * vol_ratio

        # ── RSI(36) — 3-min RSI for direction confirmation ────────────────────
        RSI_PERIOD = 36
        rsi = _rsi(close, RSI_PERIOD)

        # ── Session filter ─────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # Require 30-min warmup (360 bars) before trading
        warmup_done = np.zeros(n, dtype=bool)
        warmup_done[VOL_SLOW:] = True

        # ── VIX regime filter ──────────────────────────────────────────────────
        low_vix = vix_close < vix_limit

        # ── Vol regime filter (not wildly expanding) ───────────────────────────
        calm_vol = vol_ratio < vol_ratio_limit

        # ── Entry signals ──────────────────────────────────────────────────────
        # buy_ce: price stretched below adaptive threshold, RSI oversold → bullish reversion
        buy_ce = (
            in_session
            & warmup_done
            & low_vix
            & calm_vol
            & (price_z < -adaptive_thresh)
            & (rsi < rsi_extreme)
        )

        # buy_pe: price stretched above adaptive threshold, RSI overbought → bearish reversion
        buy_pe = (
            in_session
            & warmup_done
            & low_vix
            & calm_vol
            & (price_z > adaptive_thresh)
            & (rsi > (100.0 - rsi_extreme))
        )

        # Mutual exclusion (edge case: can't be both)
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
            time_stop_bars=18,       # 90 seconds = 18 bars × 5s
            max_trades_per_day=self.max_trades_per_day,
        )
