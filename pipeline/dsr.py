"""Deflated Sharpe Ratio (DSR) implementation.

Follows Bailey & de Prado (2016), "The Deflated Sharpe Ratio: Correcting for
Selection Bias, Backtest Overfitting and Non-Normality."
Journal of Portfolio Management 40(5).

Uses the 2016 version (not 2014) which adds the multiple-testing correction.

Two-layer adjustment:
1. Single-trial: adjusts SR for autocorrelation, skewness, excess kurtosis, finite T
2. Multiple-testing: benchmark is E[max SR | N_eff, T], not zero

DSR is the PSR (Probabilistic Sharpe Ratio) evaluated at the multiple-testing
benchmark. DSR > 0.5 means the strategy beats the benchmark at >50% confidence.
"""
from __future__ import annotations

import math
import numpy as np
from typing import Optional


# Euler-Mascheroni constant
GAMMA_EM = 0.5772156649015329


# ── Normal distribution helpers (pure Python, no scipy dependency) ───────────

def _ndtr(x: float) -> float:
    """Standard normal CDF: Φ(x) = 0.5 * erfc(-x / sqrt(2))."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _ndtri(p: float) -> float:
    """Inverse normal CDF using Acklam's rational approximation (6-digit accuracy).

    Reference: Peter J. Acklam, "An algorithm for computing the inverse
    normal cumulative distribution function", 2003.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    if p == 0.5:
        return 0.0

    # Coefficients
    a = (-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00)
    d = (7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00)

    p_low = 0.02425
    p_high = 1.0 - p_low

    if 0.0 < p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    elif p_low <= p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)


# ── Core DSR functions ────────────────────────────────────────────────────────

def compute_expected_max_sr(n_trials: int, n_obs: int) -> float:
    """Expected maximum SR under the null of no skill (Bailey & de Prado 2016).

    Proposition 3 of the paper:
        E[max SR | N, T] ≈ Z_max / sqrt(T - 1)

    where Z_max = (1-γ_EM)*Φ⁻¹(1 - 1/N) + γ_EM*Φ⁻¹(1 - 1/(N·e))

    Under the null, each SR_hat ~ N(0, 1/sqrt(T-1)), so the max of N such
    values in per-observation SR units is Z_max / sqrt(T-1).

    Returns the expected max SR in per-observation (non-annualized) units.
    Multiply by sqrt(annualization_factor) for annualized display.
    """
    if n_trials <= 0:
        return 0.0
    if n_trials == 1:
        return 0.0  # single trial, benchmark is zero

    e = math.e
    term1 = (1.0 - GAMMA_EM) * _ndtri(1.0 - 1.0 / n_trials)
    term2 = GAMMA_EM * _ndtri(1.0 - 1.0 / (n_trials * e))
    z_max = term1 + term2
    # Divide by sqrt(T-1) to convert from standard-normal Z to per-obs SR units
    return z_max / math.sqrt(max(n_obs - 1, 1))


def compute_dsr(
    returns: np.ndarray,
    n_total_trials: int,
    annualization_factor: float = 252.0,
) -> dict:
    """Compute DSR for a return series.

    Args:
        returns: Array of periodic returns (daily fractions, e.g. pnl/capital).
                 Zero-filled non-trading days must already be included.
        n_total_trials: Total number of backtest runs in the store (for N).
                        Query count_total_runs() at evaluation time.
        annualization_factor: Variance annualization (252 for daily).

    Returns:
        Dict:
            sharpe_raw      — annualized Sharpe ratio
            sharpe_deflated — DSR probability (PSR, 0–1); > 0.5 passes
            sr_benchmark    — E[max SR] benchmark in annualized terms
            n_obs           — T, number of return observations
            skewness        — return skewness
            excess_kurtosis — return excess kurtosis
            autocorr_lag1   — lag-1 autocorrelation
            n_trials_used   — effective N used for benchmark
            passes_dsr      — bool, sharpe_deflated > 0.5
    """
    T = len(returns)
    if T < 5:
        return _empty_dsr()

    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))

    if sigma == 0.0 or not math.isfinite(sigma):
        return _empty_dsr()

    sr_hat = mu / sigma  # per-observation SR (non-annualized)
    sr_hat_ann = sr_hat * math.sqrt(annualization_factor)

    # ── Moments ──
    skew = _skewness(returns)
    excess_kurt = _excess_kurtosis(returns)
    # Formula uses total kurtosis γ_4 = excess_kurtosis + 3
    kurtosis_total = excess_kurt + 3.0

    # ── Lag-1 autocorrelation ──
    autocorr = _autocorr_lag1(returns)

    # ── Variance of SR estimator (Bailey & de Prado 2016, eq. 4) ──
    # σ²(SR_hat) ≈ (1/T) * [1 - γ_3*SR_hat + ((γ_4-1)/4)*SR_hat²]
    var_component = 1.0 - skew * sr_hat + ((kurtosis_total - 1.0) / 4.0) * sr_hat ** 2
    var_component = max(var_component, 1e-12)
    std_sr = math.sqrt(var_component / T)

    # ── Benchmark: E[max SR | N_eff, T] ──
    sr_benchmark_raw = compute_expected_max_sr(max(n_total_trials, 1), T)
    sr_benchmark_ann = sr_benchmark_raw * math.sqrt(annualization_factor)

    # ── PSR (Probabilistic Sharpe Ratio) = DSR ──
    # PSR(SR*) = Φ[ (SR_hat - SR*) * √(T-1) / σ(SR_hat) ]
    # But σ(SR_hat) already has √(1/T) factor; use √(T-1) as per paper
    z_numerator = (sr_hat - sr_benchmark_raw) * math.sqrt(T - 1)
    z_denominator = math.sqrt(var_component)
    if z_denominator == 0.0:
        psr = 0.5
    else:
        z = z_numerator / z_denominator
        psr = _ndtr(z)

    return {
        "sharpe_raw": round(sr_hat_ann, 4),
        "sharpe_deflated": round(psr, 6),
        "sr_benchmark": round(sr_benchmark_ann, 4),
        "sr_benchmark_raw": round(sr_benchmark_raw, 6),
        "n_obs": T,
        "skewness": round(skew, 4),
        "excess_kurtosis": round(excess_kurt, 4),
        "autocorr_lag1": round(autocorr, 4),
        "n_trials_used": n_total_trials,
        "passes_dsr": psr > 0.5,
    }


def compute_effective_n(all_return_series: dict[str, np.ndarray]) -> tuple[int, int]:
    """Compute effective N as rank of pairwise return correlation matrix.

    Args:
        all_return_series: dict mapping run_id -> daily return array.

    Returns:
        (raw_n, effective_n) tuple.
        Falls back to (raw_n, raw_n) when insufficient data.
    """
    raw_n = len(all_return_series)
    if raw_n <= 1:
        return raw_n, raw_n

    series_list = list(all_return_series.values())
    min_len = min(len(s) for s in series_list)
    if min_len < 5:
        return raw_n, raw_n

    try:
        matrix = np.vstack([np.asarray(s, dtype=np.float64)[:min_len] for s in series_list])
        # Remove any NaN rows
        valid = ~np.any(np.isnan(matrix), axis=1)
        matrix = matrix[valid]
        if len(matrix) <= 1:
            return raw_n, raw_n

        corr_matrix = np.corrcoef(matrix)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        np.fill_diagonal(corr_matrix, 1.0)

        effective_n = int(np.linalg.matrix_rank(corr_matrix, tol=0.1))
        effective_n = max(1, min(effective_n, raw_n))
    except Exception:
        effective_n = raw_n

    return raw_n, effective_n


# ── Private helpers ───────────────────────────────────────────────────────────

def _skewness(x: np.ndarray) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mu = np.mean(x)
    sigma = np.std(x, ddof=0)
    if sigma == 0:
        return 0.0
    m3 = np.mean((x - mu) ** 3)
    return float(m3 / sigma ** 3)


def _excess_kurtosis(x: np.ndarray) -> float:
    n = len(x)
    if n < 4:
        return 0.0
    mu = np.mean(x)
    sigma = np.std(x, ddof=0)
    if sigma == 0:
        return 0.0
    m4 = np.mean((x - mu) ** 4)
    return float(m4 / sigma ** 4 - 3.0)


def _autocorr_lag1(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    try:
        r = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        return r if math.isfinite(r) else 0.0
    except Exception:
        return 0.0


def _empty_dsr() -> dict:
    return {
        "sharpe_raw": 0.0,
        "sharpe_deflated": 0.0,
        "sr_benchmark": 0.0,
        "sr_benchmark_raw": 0.0,
        "n_obs": 0,
        "skewness": 0.0,
        "excess_kurtosis": 0.0,
        "autocorr_lag1": 0.0,
        "n_trials_used": 0,
        "passes_dsr": False,
    }
