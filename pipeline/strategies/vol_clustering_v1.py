"""vol_clustering_v1 — GARCH Vol Clustering Breakout on NIFTY 5-second bars.

Thesis: On NIFTY, GARCH(1,1) vol clustering at 5-second resolution reflects
institutional order bursts. When conditional sigma rank exceeds the 75th
percentile for 3+ consecutive bars AND price makes a directional breakout
(new 100-second high/low) with EMA alignment, institutional flow is likely
still executing — trade the continuation for 30-90 seconds.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average; NaN for the first (period-1) bars."""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    alpha = 2.0 / (period + 1)
    result[period - 1] = np.mean(arr[:period])
    for i in range(period, n):
        result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
    return result


def _garch11_sigma(ret: np.ndarray, omega: float = 1e-6,
                   alpha: float = 0.10, beta: float = 0.85) -> np.ndarray:
    """GARCH(1,1) conditional standard deviation on return series.

    sigma2[t] = omega + alpha * ret[t-1]^2 + beta * sigma2[t-1]
    Initialized at the unconditional variance.
    """
    n = len(ret)
    sigma2 = np.zeros(n)
    # Unconditional variance as initializer; clamp so alpha+beta < 1
    denom = max(1.0 - alpha - beta, 1e-6)
    sigma2[0] = omega / denom
    for i in range(1, n):
        sigma2[i] = omega + alpha * ret[i - 1] ** 2 + beta * sigma2[i - 1]
    return np.sqrt(np.maximum(sigma2, 1e-16))


def _rolling_percentile_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling percentile rank of arr[i] within arr[i-window+1 : i+1].

    Returns values in [0, 100]. Zero for the first (window-1) bars.
    """
    n = len(arr)
    result = np.zeros(n)
    for i in range(window - 1, n):
        w = arr[i - window + 1: i + 1]
        result[i] = float(np.sum(w <= arr[i])) / window * 100.0
    return result


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling maximum over the last `window` bars (inclusive)."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.max(arr[i - window + 1: i + 1])
    return result


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum over the last `window` bars (inclusive)."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        result[i] = np.min(arr[i - window + 1: i + 1])
    return result


class Strategy(BaseStrategy):
    name = "vol_clustering_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup for GARCH + percentile rank

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_pct_threshold", 75.0, 60.0, 90.0),
            TunableParam("cluster_bars_min", 3.0, 2.0, 6.0),
            TunableParam("vix_min", 13.0, 10.0, 18.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
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

        # ── Extract spot arrays (forward-fill NaN before numpy) ──────────────
        close = (
            spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        )
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ───────────────────────────────────────────────────────
        vol_pct_threshold = float(params.get("vol_pct_threshold", 75.0))
        cluster_bars_min = int(params.get("cluster_bars_min", 3.0))
        vix_min = float(params.get("vix_min", 13.0))
        stop_pts = float(params.get("stop_pts", 3.0))
        target_pts = float(params.get("target_pts", 6.0))

        # ── 5-second log returns ──────────────────────────────────────────────
        ret = np.zeros(n)
        ret[1:] = np.log(np.maximum(close[1:], 1e-8) / np.maximum(close[:-1], 1e-8))

        # ── GARCH(1,1) conditional sigma ──────────────────────────────────────
        sigma = _garch11_sigma(ret, omega=1e-6, alpha=0.10, beta=0.85)

        # ── Rolling 60-bar (5-min) percentile rank of sigma ───────────────────
        vol_pct = _rolling_percentile_rank(sigma, window=60)

        # ── High-vol cluster: N+ consecutive bars above threshold ─────────────
        above = vol_pct > vol_pct_threshold
        consec = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            consec[i] = (consec[i - 1] + 1) if above[i] else 0
        high_vol_cluster = consec >= cluster_bars_min

        # ── EMA 8 (40s) and EMA 21 (105s) for directional bias ───────────────
        ema8 = _ema(close, 8)
        ema21 = _ema(close, 21)
        # Forward-fill NaN (replace with close as neutral placeholder)
        ema8 = np.where(np.isfinite(ema8), ema8, close)
        ema21 = np.where(np.isfinite(ema21), ema21, close)

        # ── 20-bar rolling high/low on close (100 seconds) ───────────────────
        roll_close_high = _rolling_max(close, 20)
        roll_close_low = _rolling_min(close, 20)
        # Use previous bar's rolling max/min so we don't use current bar
        prev_roll_high = np.roll(roll_close_high, 1)
        prev_roll_high[0] = roll_close_high[0]
        prev_roll_low = np.roll(roll_close_low, 1)
        prev_roll_low[0] = roll_close_low[0]
        # Replace NaN from first 19 bars with close (no false breakouts)
        prev_roll_high = np.where(np.isfinite(prev_roll_high), prev_roll_high, close)
        prev_roll_low = np.where(np.isfinite(prev_roll_low), prev_roll_low, close)

        new_high = close >= prev_roll_high
        new_low = close <= prev_roll_low

        # ── India VIX filter ─────────────────────────────────────────────────
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

        vix_ok = vix_close >= vix_min

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # ── Directional filters ───────────────────────────────────────────────
        bullish = (close > ema8) & (ema8 > ema21)
        bearish = (close < ema8) & (ema8 < ema21)

        # ── Entry signals ─────────────────────────────────────────────────────
        buy_ce = in_session & vix_ok & high_vol_cluster & bullish & new_high
        buy_pe = in_session & vix_ok & high_vol_cluster & bearish & new_low

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
