"""Bootstrap Confidence Interval Momentum Strategy — NIFTY 5-second options.

Detects statistically significant directional bias in NIFTY 5-second returns using
block bootstrap resampling. Enters ATM options when the 90% bootstrap CI of the mean
of the last 120 5-second log returns (10 min) is entirely above/below zero.

Mechanism: On NIFTY, sustained institutional/FII order flow creates serial autocorrelation
in 5-second returns across a 5-15 minute window. When the block bootstrap CI (block_size=6,
30-second blocks to capture NIFTY micro-trend autocorrelation) is entirely positive, the
directional bias is statistically robust — not a single-bar spike but a sustained regime.
The same institutional order inventory that created the significance tends to persist for
another 30-90 seconds, making continuation probabilistically favourable.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _block_bootstrap_ci(
    returns_window: np.ndarray,
    n_reps: int = 200,
    block_size: int = 6,
) -> tuple[float, float]:
    """Compute 90% CI for mean return using vectorized block bootstrap.

    Args:
        returns_window: 1-D array of log returns (length n).
        n_reps: Number of bootstrap replicates.
        block_size: Block length in bars (6 bars = 30 seconds at 5s frequency).

    Returns:
        (ci_lower, ci_upper) — 5th and 95th percentiles of bootstrap distribution.
        Returns (nan, nan) if window is too short.
    """
    n = len(returns_window)
    n_blocks = n // block_size
    if n_blocks < 2:
        return np.nan, np.nan

    max_start = n - block_size  # inclusive upper bound for block start
    if max_start < 0:
        return np.nan, np.nan

    # Vectorized block bootstrap: sample n_blocks random starting positions per rep
    # starts shape: (n_reps, n_blocks)
    starts = np.random.randint(0, max_start + 1, size=(n_reps, n_blocks))

    # Build index matrix: (n_reps, n_blocks * block_size)
    offsets = np.arange(block_size)  # shape (block_size,)
    # idx shape: (n_reps, n_blocks, block_size) → (n_reps, n_blocks * block_size)
    idx = (starts[:, :, np.newaxis] + offsets[np.newaxis, np.newaxis, :]).reshape(
        n_reps, n_blocks * block_size
    )
    # Trim to exact window length (n_blocks * block_size may exceed n slightly)
    idx = idx[:, :n]

    # Gather bootstrap samples and compute mean per replicate
    boot_means = returns_window[idx].mean(axis=1)  # shape (n_reps,)

    return float(np.percentile(boot_means, 5)), float(np.percentile(boot_means, 95))


class Strategy(BaseStrategy):
    name = "bootstrap_confidence_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — after 10-min bootstrap warmup
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 120            # 120 × 5s = 10 min bootstrap window

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum CI lower/upper bound away from zero (in log-return units)
            # 0.00008 ≈ 0.8 bps per 5-second bar ≈ 0.096% per 120-bar window
            TunableParam("ci_min_signal", 0.00008, 0.00003, 0.00020),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
            TunableParam("vix_cap", 22.0, 16.0, 28.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot data ──────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        ci_min_signal = params.get("ci_min_signal", 0.00008)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)
        vix_cap = params.get("vix_cap", 22.0)

        # ── VIX filter ─────────────────────────────────────────────────────────
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

        # ── 5-second log returns ───────────────────────────────────────────────
        log_ret = np.zeros(n)
        valid = close > 0
        log_ret[1:] = np.where(
            valid[1:] & valid[:-1],
            np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], 1.0)),
            0.0,
        )

        # ── Block bootstrap CI for each bar ───────────────────────────────────
        # ci_lower: 5th percentile of bootstrap mean distribution
        # ci_upper: 95th percentile of bootstrap mean distribution
        # If ci_lower > 0 → entire 90% CI positive → bullish
        # If ci_upper < 0 → entire 90% CI negative → bearish
        lookback = self.max_lookback  # 120 bars
        ci_lower = np.zeros(n)
        ci_upper = np.zeros(n)

        for i in range(lookback, n):
            window = log_ret[i - lookback:i]
            lo, hi = _block_bootstrap_ci(window, n_reps=200, block_size=6)
            if not (np.isnan(lo) or np.isnan(hi)):
                ci_lower[i] = lo
                ci_upper[i] = hi

        # ── Session and VIX filter ─────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        low_vix = vix_close < vix_cap

        # ── Entry signals ──────────────────────────────────────────────────────
        # buy CE: CI entirely above zero with minimum signal strength
        buy_ce = in_session & low_vix & (ci_lower > ci_min_signal)

        # buy PE: CI entirely below zero with minimum signal strength
        buy_pe = in_session & low_vix & (ci_upper < -ci_min_signal)

        # Mutual exclusion: never fire both on the same bar
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
