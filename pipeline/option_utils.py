"""
Option-specific utilities for 5-second index option trading.

Provides ATM strike identification, time-to-expiry, and Greek approximations.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl

# ── Strike step sizes ────────────────────────────────────────────────
STRIKE_STEP = {
    "NIFTY": 50,
    "BANKNIFTY": 100,
    "NSE:NIFTY50-INDEX": 50,
    "NSE:NIFTYBANK-INDEX": 100,
}


def get_strike_step(underlying: str) -> int:
    key = underlying.upper()
    for k, v in STRIKE_STEP.items():
        if key in k.upper() or k.upper() in key:
            return v
    raise KeyError(f"Unknown underlying for strike step: {underlying}")


# ── ATM strike ───────────────────────────────────────────────────────

def nearest_strike(spot_price: float, step: int) -> float:
    """Round spot price to the nearest strike (e.g. 24350 → 24350 for step=50).

    Uses math.floor(x + 0.5) instead of Python's round() to avoid banker's
    rounding (round-half-to-even), which causes inconsistent ATM strike
    assignment at exact midpoints.
    """
    return int(math.floor(spot_price / step + 0.5)) * step


def atm_strike_series(spot_close: pl.Series | np.ndarray, step: int) -> np.ndarray:
    """Vectorised ATM strike for every bar. Returns float64 array.

    Uses floor(x + 0.5) to avoid numpy's banker's rounding at midpoints.
    """
    if isinstance(spot_close, pl.Series):
        arr = spot_close.to_numpy().astype(np.float64)
    else:
        arr = np.asarray(spot_close, dtype=np.float64)
    return np.floor(arr / step + 0.5) * step


# ── Time to expiry ────────────────────────────────────────────────────

def time_to_expiry_hours(
    bar_ts: np.ndarray,          # datetime64[us] in IST (naive)
    expiry_date: np.ndarray,     # datetime64[D]
    market_close_hour: int = 15,
    market_close_min: int = 30,
) -> np.ndarray:
    """
    Compute hours remaining to expiry for each bar.

    Expiry is assumed to be at market_close_hour:market_close_min IST
    on the expiry date.

    Parameters
    ----------
    bar_ts      : array of datetime64[us]  (IST, naive)
    expiry_date : array of datetime64[D]

    Returns
    -------
    hours_remaining : float64 array  (can be negative after expiry)
    """
    # Convert expiry dates to expiry timestamps (15:30 IST on expiry day)
    expiry_ts = expiry_date.astype("datetime64[D]").astype("datetime64[us]")
    close_offset = np.timedelta64(market_close_hour * 60 + market_close_min, "m")
    expiry_ts = expiry_ts + close_offset

    delta_us = (expiry_ts - bar_ts).astype(np.float64)  # microseconds
    hours = delta_us / 3.6e9
    return hours


def is_expiry_day(session_date: np.ndarray, expiry_date: np.ndarray) -> np.ndarray:
    """Boolean mask: True when bar is on the expiry day."""
    return session_date == expiry_date


# ── Black-Scholes Greeks (quick approximations) ──────────────────────

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes call price.  T in years, sigma annualised."""
    if T <= 0:
        return max(S - K, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, opt_type: str = "CE") -> float:
    """Delta of a European option. Returns value in [0,1] for CE, [-1,0] for PE."""
    if T <= 0:
        if opt_type == "CE":
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    if opt_type == "CE":
        return _norm_cdf(d1)
    else:
        return _norm_cdf(d1) - 1.0


def implied_vol_from_price(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    opt_type: str = "CE",
    tol: float = 1e-4,
    max_iter: int = 50,
) -> float:
    """Newton-Raphson IV solver. Returns annualised vol or NaN if no convergence."""
    if T <= 0 or market_price <= 0:
        return float("nan")
    sigma = 0.3  # initial guess
    for _ in range(max_iter):
        if opt_type == "CE":
            price = bs_call_price(S, K, T, r, sigma)
        else:
            price = bs_put_price(S, K, T, r, sigma)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        # Vega
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        vega = S * sqrt_T * math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)
        if vega < 1e-10:
            break
        sigma -= diff / vega
        sigma = max(sigma, 0.01)
    return float("nan")


# ── Vectorised helpers ────────────────────────────────────────────────

def add_atm_strike_to_spot(spot_df: pl.DataFrame, underlying: str) -> pl.DataFrame:
    """Add 'atm_strike' column to a spot DataFrame."""
    step = get_strike_step(underlying)
    return spot_df.with_columns(
        (pl.col("close") / step + 0.5).floor().cast(pl.Float64).mul(step).alias("atm_strike")
    )
