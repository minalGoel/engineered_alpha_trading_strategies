"""spectral_analysis_v1 — FFT Cycle Detection on NIFTY 5-second bars.

Mechanism: Institutional TWAP/VWAP execution and option market-maker delta-hedging
create periodic micro-oscillations in NIFTY at 30-180 second cycle lengths. When the
FFT of the last 20 minutes of 5-second closes reveals a dominant spectral peak with
signal-to-noise ratio > 3.5x the noise floor, we enter at confirmed phase extrema:
cycle trough (phase ≈ π) for CE, cycle peak (phase ≈ 0) for PE.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_spectral(
    close: np.ndarray,
    window: int,
    k_min: int,
    k_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute dominant period, spectral ratio, and cycle phase for each bar.

    Args:
        close: 1-D float64 array of close prices
        window: FFT window size in bars
        k_min: minimum frequency bin (corresponds to max period = window/k_min)
        k_max: maximum frequency bin (corresponds to min period = window/k_max)

    Returns:
        dominant_period: period in bars (0 if not computed)
        spectral_ratio: peak_power / mean_power (0 if not computed)
        cycle_phase: phase at current bar in [0, 2π) (π = neutral/trough)
    """
    n = len(close)
    dominant_period = np.zeros(n)
    spectral_ratio = np.zeros(n)
    # Default π = neutral (neither peak nor trough)
    cycle_phase = np.full(n, np.pi)

    half_win = window // 2
    x_lin = np.arange(window, dtype=np.float64)
    two_pi = 2.0 * np.pi

    for i in range(window, n):
        seg = close[i - window:i]

        # Linear detrend to remove drift within the window
        coeffs = np.polyfit(x_lin, seg, 1)
        detrended = seg - (coeffs[0] * x_lin + coeffs[1])

        # Real FFT: length window//2 + 1 complex coefficients
        fft_vals = np.fft.rfft(detrended)
        power = np.abs(fft_vals) ** 2

        # Restrict to valid frequency bins [k_min, k_max]
        k_lo = max(1, k_min)
        k_hi = min(len(power) - 1, k_max)
        if k_hi <= k_lo:
            continue

        valid_power = power[k_lo:k_hi + 1]
        dom_k_rel = int(np.argmax(valid_power))
        dom_k = dom_k_rel + k_lo

        if dom_k == 0:
            continue

        dominant_period[i] = window / dom_k

        # Spectral ratio: dominant power vs mean power over all non-DC bins
        mean_pow = np.mean(power[1:half_win + 1])
        if mean_pow > 1e-12:
            spectral_ratio[i] = power[dom_k] / mean_pow

        # Phase of dominant component at bar i (one step after the FFT window [i-N, i))
        # Sinusoidal component: A*cos(2π*k*t/N + φ) where φ = arg(FFT[k])
        # At t = N (current bar, one step past window end): phase = 2π*k + φ ≡ φ (mod 2π)
        phi = np.angle(fft_vals[dom_k])
        cycle_phase[i] = phi % two_pi

    return dominant_period, spectral_ratio, cycle_phase


def _period_stability(dominant_period: np.ndarray, lookback: int, tol: float) -> np.ndarray:
    """True if dominant_period has been stable within ±tol for the past `lookback` bars."""
    n = len(dominant_period)
    stable = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        window = dominant_period[i - lookback:i + 1]
        # All values must be non-zero (computed) and spread <= tol
        if window[-1] > 0 and np.min(window) > 0:
            if (np.max(window) - np.min(window)) <= tol:
                stable[i] = True
    return stable


class Strategy(BaseStrategy):
    name = "spectral_analysis_v1"
    underlying = "NIFTY"
    session_start_minutes = 580   # 09:40 IST — 240-bar (20 min) FFT warmup before signals
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 288            # 240-bar FFT window + 48 bars stability buffer
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("spectral_ratio_threshold", 3.5, 2.5, 6.0),
            # phase_tolerance in radians: default π/4 ≈ 0.785, range [π/8, π/2]
            TunableParam("phase_tolerance", 0.7854, 0.3927, 1.5708),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Parameters ───────────────────────────────────────────────────────
        srt = params.get("spectral_ratio_threshold", 3.5)
        phase_tol = params.get("phase_tolerance", 0.7854)   # radians, default π/4
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # ── Spot close (forward-fill nulls) ─────────────────────────────────
        close = (
            spot_df["close"]
            .fill_null(strategy="forward")
            .fill_null(0.0)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()

        # ── VIX close (backward-join, fill 15.0 if missing) ─────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Spectral analysis ────────────────────────────────────────────────
        fft_window = 240   # 20 minutes at 5s bars
        # Target cycles: 30s–3min → 6–36 bars
        # k_min = window / max_period = 240/36 = 6 (rounded to int)
        # k_max = window / min_period = 240/6  = 40
        k_min = 6
        k_max = 40

        dominant_period, spectral_ratio, cycle_phase = _compute_spectral(
            close, fft_window, k_min, k_max
        )

        # ── Period stability: stable ±2 bars for last 6 bars (30s) ──────────
        period_stable = _period_stability(dominant_period, lookback=6, tol=2.0)

        # ── 2-bar ROC (10-second directional confirmation) ───────────────────
        roc2 = np.zeros(n)
        roc2[2:] = close[2:] - close[:-2]

        # ── Phase zone detection ─────────────────────────────────────────────
        pi = np.pi
        two_pi = 2.0 * pi

        # Near trough (phase ≈ π): cosine at minimum → price locally low → buy CE
        near_trough = np.abs(cycle_phase - pi) < phase_tol

        # Near peak (phase ≈ 0 or 2π): cosine at maximum → price locally high → buy PE
        near_peak = (cycle_phase < phase_tol) | (cycle_phase > (two_pi - phase_tol))

        # ── Combined filters ─────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < 22.0
        cycle_ok = (
            (spectral_ratio >= srt)
            & (dominant_period >= 6)
            & (dominant_period <= 36)
            & period_stable
        )

        # ── Signals ──────────────────────────────────────────────────────────
        # Buy CE: cycle at trough + upward 10s confirmation
        buy_ce = in_session & vix_ok & cycle_ok & near_trough & (roc2 > 0)

        # Buy PE: cycle at peak + downward 10s confirmation
        buy_pe = in_session & vix_ok & cycle_ok & near_peak & (roc2 < 0)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,              # 60 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
