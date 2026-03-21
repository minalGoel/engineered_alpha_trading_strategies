"""
hurst_exponent_v1 — Hurst Exponent Regime-Adaptive Strategy

Mechanism:
On NIFTY, the rolling 10-minute Hurst exponent (R/S method, 120 5-second bars) classifies
the current microstructure regime. When H > 0.55, institutional TWAP/VWAP algorithms drive
persistent flow — we trend-follow using EMA crossovers. When H < 0.45, market makers are
quoting both sides and price oscillates — we fade Bollinger Band extremes for mean-reversion
bounces. The Hurst computation uses 120 bars because shorter windows produce unreliable
estimates (high variance), while longer windows miss intraday regime shifts.

Original: equity strategy on NIFTY50 stocks, 1-min bars, 15-45 min hold.
Converted: NIFTY index, 5-second bars, 30-120s hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _hurst_rs(log_ret: np.ndarray, window: int) -> np.ndarray:
    """Rolling Hurst exponent via rescaled-range (R/S) method.

    Returns array of same length as log_ret.
    Values before index `window` are set to 0.5 (neutral / random walk).
    """
    n = len(log_ret)
    hurst = np.full(n, 0.5)
    log_window = np.log(window)
    for i in range(window, n):
        r = log_ret[i - window : i]
        s = np.std(r)
        if s <= 0.0:
            continue
        m = np.mean(r)
        devs = np.cumsum(r - m)
        R = devs.max() - devs.min()
        if R <= 0.0:
            continue
        hurst[i] = np.log(R / s) / log_window
    return hurst


def _ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    k = 2.0 / (period + 1)
    out = np.empty(len(close))
    out[0] = close[0]
    for i in range(1, len(close)):
        out[i] = close[i] * k + out[i - 1] * (1.0 - k)
    return out


def _bollinger(close: np.ndarray, window: int, mult: float):
    """Bollinger Bands: returns (mid, upper, lower)."""
    n = len(close)
    mid = np.empty(n)
    upper = np.empty(n)
    lower = np.empty(n)
    # initialise first window-1 bars
    mid[:window] = close[:window]
    upper[:window] = close[:window]
    lower[:window] = close[:window]
    for i in range(window, n):
        w = close[i - window : i]
        m = np.mean(w)
        s = np.std(w)
        mid[i] = m
        upper[i] = m + mult * s
        lower[i] = m - mult * s
    return mid, upper, lower


class Strategy(BaseStrategy):
    name = "hurst_exponent_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — need 120 bars (10 min) warmup from 09:20
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 144            # 120 (Hurst window) + 24 (BB window) buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("hurst_persist_threshold", 0.55, 0.50, 0.65),
            TunableParam("hurst_revert_threshold", 0.45, 0.35, 0.50),
            TunableParam("bb_std_mult", 2.0, 1.5, 2.5),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Parameters ──────────────────────────────────────────────────────────
        h_persist = float(params.get("hurst_persist_threshold", 0.55))
        h_revert = float(params.get("hurst_revert_threshold", 0.45))
        bb_mult = float(params.get("bb_std_mult", 2.0))
        stop_pts = float(params.get("stop_pts", 4.0))
        target_pts = float(params.get("target_pts", 6.0))

        # ── Raw data ─────────────────────────────────────────────────────────────
        close = (
            spot_df["close"]
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .to_numpy()
            .astype(np.float64)
        )
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Log returns (5-second) ───────────────────────────────────────────────
        log_ret = np.zeros(n)
        safe_prev = np.where(close[:-1] > 0.0, close[:-1], 1.0)
        log_ret[1:] = np.log(close[1:] / safe_prev)

        # ── Hurst exponent — 120 bars = 10 minutes ───────────────────────────────
        hurst_window = 120
        hurst = _hurst_rs(log_ret, hurst_window)
        # Clamp to [0, 1] to discard numerical outliers
        hurst = np.clip(hurst, 0.0, 1.0)
        # Track bars where Hurst is actually computed (valid)
        hurst_valid = np.zeros(n, dtype=bool)
        hurst_valid[hurst_window:] = True

        # ── EMA crossover for persistent (trending) regime ───────────────────────
        ema_fast = _ema(close, 12)   # 1-minute fast EMA
        ema_slow = _ema(close, 36)   # 3-minute slow EMA

        # ── Bollinger Bands for anti-persistent (mean-reverting) regime ──────────
        bb_window = 24               # 2-minute BB
        bb_mid, bb_upper, bb_lower = _bollinger(close, bb_window, bb_mult)

        # ── 1-bar price direction confirmation ──────────────────────────────────
        ret_1 = np.zeros(n)
        ret_1[1:] = close[1:] - close[:-1]

        # ── Session filter ───────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        valid = in_session & hurst_valid

        # ── Regime classification ────────────────────────────────────────────────
        persistent = hurst > h_persist      # institutional directional flow
        anti_persist = hurst < h_revert     # market-maker oscillation

        # ── Trend-following signals (persistent regime) ──────────────────────────
        buy_ce_trend = valid & persistent & (ema_fast > ema_slow) & (ret_1 > 0.0)
        buy_pe_trend = valid & persistent & (ema_fast < ema_slow) & (ret_1 < 0.0)

        # ── Mean-reversion signals (anti-persistent regime) ──────────────────────
        buy_ce_revert = valid & anti_persist & (close < bb_lower)  # oversold bounce
        buy_pe_revert = valid & anti_persist & (close > bb_upper)  # overbought fade

        # ── Combine regimes ──────────────────────────────────────────────────────
        buy_ce = buy_ce_trend | buy_ce_revert
        buy_pe = buy_pe_trend | buy_pe_revert

        # Eliminate simultaneous conflicting signals (should be rare)
        conflict = buy_ce & buy_pe
        buy_ce = buy_ce & ~conflict
        buy_pe = buy_pe & ~conflict

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts, dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # max 120 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
