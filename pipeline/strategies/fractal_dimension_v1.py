"""fractal_dimension_v1 — Higuchi Fractal Dimension Regime Filter on NIFTY.

When NIFTY's 5-second price series has Higuchi FD < 1.30 (low-complexity trending regime),
institutional TWAP flow creates persistent directional pressure. We enter in the direction
indicated by EMA crossover + VWAP position when FD has been trending for 3+ consecutive bars.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _higuchi_fd(x: np.ndarray, k_max: int) -> float:
    """Compute Higuchi Fractal Dimension of a 1D time series.

    Returns a value in [1.0, 2.0]: 1.0 = perfectly smooth trend, 2.0 = pure noise.
    Returns 1.5 (neutral) if computation is degenerate.
    """
    N = len(x)
    L = np.zeros(k_max)
    for k in range(1, k_max + 1):
        Lm_vals = []
        for m in range(1, k + 1):
            # Sub-series X_m = {x[m-1], x[m-1+k], x[m-1+2k], ...}
            idx = np.arange(m - 1, N, k)
            if len(idx) < 2:
                continue
            x_m = x[idx]
            diff_sum = np.sum(np.abs(np.diff(x_m)))
            # Normalisation: (N-1) / (floor((N-m)/k) * k^2)
            norm = (N - 1) / ((len(idx) - 1) * k * k)
            Lm_vals.append(diff_sum * norm)
        if Lm_vals:
            L[k - 1] = float(np.mean(Lm_vals))

    ks = np.arange(1, k_max + 1, dtype=float)
    valid = L > 0
    if np.sum(valid) < 2:
        return 1.5

    log_inv_k = np.log(1.0 / ks[valid])
    log_L = np.log(L[valid])

    xm = float(np.mean(log_inv_k))
    ym = float(np.mean(log_L))
    num = float(np.sum((log_inv_k - xm) * (log_L - ym)))
    den = float(np.sum((log_inv_k - xm) ** 2))

    if den == 0.0:
        return 1.5

    fd = num / den
    return float(np.clip(fd, 1.0, 2.0))


def _ema(x: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with seed = first value."""
    alpha = 2.0 / (period + 1)
    out = np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _compute_vwap(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                  volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP, reset each trading day."""
    n = len(close)
    typical = (high + low + close) / 3.0
    vwap = np.empty(n)
    cum_tp_vol = 0.0
    cum_vol = 0.0
    prev_day = -999999

    for i in range(n):
        if day_id[i] != prev_day:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        vol_i = volume[i] if volume[i] > 0 else 0.0
        cum_tp_vol += typical[i] * vol_i
        cum_vol += vol_i
        vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else close[i]

    return vwap


class Strategy(BaseStrategy):
    name = "fractal_dimension_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — need 10 min warmup for EMA_slow(60)
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 120            # 10 min warmup (60 bars EMA_slow + buffer)

    # Fixed algorithm parameters (not tunable — affect indicator validity)
    _FD_WINDOW = 48   # 4-minute rolling FD window (48 × 5s)
    _K_MAX = 6        # Higuchi k_max; requires FD_WINDOW >= 2 × K_MAX
    _EMA_FAST = 24    # 2-min EMA — matches lower hold-time bound
    _EMA_SLOW = 60    # 5-min EMA — proxies institutional TWAP sub-interval
    _VWAP_SLOPE_BARS = 12  # 1-min VWAP slope

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("fd_threshold", 1.30, 1.18, 1.42),
            TunableParam("stop_pts",     4.0,  2.5,  7.0),
            TunableParam("target_pts",   7.0,  4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        fd_threshold = float(params.get("fd_threshold", 1.30))
        stop_pts     = float(params.get("stop_pts",     4.0))
        target_pts   = float(params.get("target_pts",   7.0))

        # ── Raw arrays (forward-fill NaN before numpy) ──
        close  = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high   = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low    = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].to_numpy().astype(int)
        time_min = spot_df["time_minutes"].to_numpy().astype(int)

        # ── Rolling Higuchi FD ──
        fd = np.full(n, 1.5)   # neutral (noisy) until enough data
        fw = self._FD_WINDOW
        for i in range(fw - 1, n):
            fd[i] = _higuchi_fd(close[i - fw + 1: i + 1], self._K_MAX)

        # ── FD persistence: require FD < threshold for ≥3 consecutive bars ──
        fd_below = fd < fd_threshold
        fd_persistent = np.zeros(n, dtype=bool)
        run = 0
        for i in range(n):
            run = run + 1 if fd_below[i] else 0
            fd_persistent[i] = run >= 3

        # ── EMAs ──
        ema_fast = _ema(close, self._EMA_FAST)
        ema_slow = _ema(close, self._EMA_SLOW)

        # ── Session VWAP ──
        vwap = _compute_vwap(close, high, low, volume, day_id)

        # ── VWAP slope (1-min) ──
        vwap_slope = np.zeros(n)
        sb = self._VWAP_SLOPE_BARS
        vwap_slope[sb:] = vwap[sb:] - vwap[:-sb]

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Directional conditions ──
        bullish = (ema_fast > ema_slow) & (close > vwap) & (vwap_slope > 0.0)
        bearish = (ema_fast < ema_slow) & (close < vwap) & (vwap_slope < 0.0)

        buy_ce = in_session & fd_persistent & bullish
        buy_pe = in_session & fd_persistent & bearish

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
