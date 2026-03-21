"""
trend_strength_adaptive_v1 — ADX-Adaptive EMA Crossover on NIFTY

Mechanism: When NIFTY is in a strong 3-minute trend (high ADX), institutional order
flow is directionally persistent over the next 30-90 seconds. The adaptive EMA crossover
contracts its lookback as ADX rises — faster signal precisely when speed is reliable.
The +DI/-DI comparison confirms directional asymmetry; VWAP filter aligns with
session-level institutional benchmarking.

Hold: 15-120s. Underlying: NIFTY.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wilder's ADX, +DI, -DI.

    Returns (adx, plus_di, minus_di) arrays of length n.
    Uses Wilder's smoothing: initialised as sum of first `period` values,
    then each bar: smoothed = smoothed * (1 - 1/period) + raw * (1/period).
    ADX initialised as mean of first `period` DX values starting at bar 2*period-2.
    """
    n = len(close)
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # Directional movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        if up > dn and up > 0:
            plus_dm[i] = up
        if dn > up and dn > 0:
            minus_dm[i] = dn

    alpha = 1.0 / period
    s_tr = np.zeros(n)
    s_plus = np.zeros(n)
    s_minus = np.zeros(n)

    if period <= n:
        s_tr[period - 1] = np.sum(tr[:period])
        s_plus[period - 1] = np.sum(plus_dm[:period])
        s_minus[period - 1] = np.sum(minus_dm[:period])
        for i in range(period, n):
            s_tr[i] = s_tr[i - 1] * (1.0 - alpha) + tr[i]
            s_plus[i] = s_plus[i - 1] * (1.0 - alpha) + plus_dm[i]
            s_minus[i] = s_minus[i - 1] * (1.0 - alpha) + minus_dm[i]

    plus_di = np.where(s_tr > 0, 100.0 * s_plus / s_tr, 0.0)
    minus_di = np.where(s_tr > 0, 100.0 * s_minus / s_tr, 0.0)

    di_sum = plus_di + minus_di
    dx = np.where(di_sum > 0, 100.0 * np.abs(plus_di - minus_di) / di_sum, 0.0)

    # ADX: Wilder smooth of DX, initialised at bar (2*period - 2)
    adx = np.zeros(n)
    start = 2 * period - 2
    if start < n:
        adx[start] = np.mean(dx[period - 1 : start + 1])
        for i in range(start + 1, n):
            adx[i] = adx[i - 1] * (1.0 - alpha) + dx[i] * alpha

    return adx, plus_di, minus_di


def _compute_adaptive_ema(
    close: np.ndarray,
    adx: np.ndarray,
    base: int = 48,
    div: int = 3,
    min_lb: int = 12,
) -> np.ndarray:
    """Variable-alpha EMA: lookback = max(min_lb, base - floor(adx / div)).

    As ADX rises the lookback contracts, making the EMA faster.
    Alpha = 2 / (lookback + 1).
    """
    n = len(close)
    ema = np.empty(n)
    ema[0] = close[0]
    for i in range(1, n):
        lb = max(min_lb, base - int(adx[i] / div))
        alpha = 2.0 / (lb + 1)
        ema[i] = ema[i - 1] * (1.0 - alpha) + close[i] * alpha
    return ema


def _compute_vwap(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Session VWAP, reset each day."""
    n = len(close)
    typical = (high + low + close) / 3.0
    vwap = np.zeros(n)
    cum_tp_vol = 0.0
    cum_vol = 0.0
    prev_day = -1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        v = max(float(volume[i]), 1.0)
        cum_tp_vol += typical[i] * v
        cum_vol += v
        vwap[i] = cum_tp_vol / cum_vol
    return vwap


class Strategy(BaseStrategy):
    """ADX-Adaptive EMA Crossover — NIFTY 5s options.

    Entry when ADX(36) confirms a 3-minute trend AND adaptive EMA crossover
    aligns directionally AND VWAP confirms session flow direction.
    The EMA lookback contracts from ~36 bars (low ADX) to ~12 bars (high ADX),
    exploiting that fast signals are only reliable in genuinely trending regimes.
    """

    name = "trend_strength_adaptive_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    # ADX(36) needs 2×36 = 72 bars warmup; add buffer → 144 bars (12 min)
    max_lookback = 144

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_threshold", 18.0, 12.0, 30.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # --- Extract arrays (forward-fill NaN before numpy) ---
        close = (
            spot_df["close"].fill_null(strategy="forward").fill_null(0.0).to_numpy().astype(float)
        )
        high = (
            spot_df["high"].fill_null(strategy="forward").fill_null(0.0).to_numpy().astype(float)
        )
        low = (
            spot_df["low"].fill_null(strategy="forward").fill_null(0.0).to_numpy().astype(float)
        )
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # --- Parameters ---
        adx_threshold = float(params.get("adx_threshold", 18.0))
        stop_pts = float(params.get("stop_pts", 4.0))
        target_pts = float(params.get("target_pts", 7.0))

        # --- Indicators ---
        # ADX(36) = 3-minute trend strength
        adx, plus_di, minus_di = _compute_adx(high, low, close, period=36)

        # Adaptive fast EMA: lookback = max(12, 48 - floor(adx/3)) → range 12-48 bars
        ema_fast = _compute_adaptive_ema(close, adx, base=48, div=3, min_lb=12)

        # Adaptive slow EMA: lookback = max(24, 96 - floor(adx*2/3)) → range 24-96 bars
        ema_slow = _compute_adaptive_ema(close, adx, base=96, div=3, min_lb=24)
        # Note: slow = 2× fast. We achieve this by using base=96, same div=3 but adx*2:
        # max(24, 96 - floor(adx*2/3)) ≈ 2× max(12, 48 - floor(adx/3))
        # Simpler: compute directly
        ema_slow2 = np.empty(n)
        ema_slow2[0] = close[0]
        for i in range(1, n):
            lb_fast = max(12, 48 - int(adx[i] / 3))
            lb_slow = max(24, lb_fast * 2)
            alpha_s = 2.0 / (lb_slow + 1)
            ema_slow2[i] = ema_slow2[i - 1] * (1.0 - alpha_s) + close[i] * alpha_s
        ema_slow = ema_slow2

        # Session VWAP
        vwap = _compute_vwap(close, high, low, volume, day_id)

        # ADX rising: compare current ADX to 6 bars ago (30s window)
        adx_rising = np.zeros(n, dtype=bool)
        adx_rising[6:] = adx[6:] > adx[:-6]

        # --- Masks ---
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        # Warmup: need at least 2×36 = 72 bars for ADX to be non-zero
        warmed = np.zeros(n, dtype=bool)
        if n > 72:
            warmed[72:] = True

        # Bullish: +DI > -DI, ADX trending up, fast EMA above slow EMA, price above VWAP
        buy_ce = (
            in_session
            & warmed
            & (plus_di > minus_di)
            & (adx > adx_threshold)
            & adx_rising
            & (ema_fast > ema_slow)
            & (close > vwap)
        )

        # Bearish: -DI > +DI, ADX trending up, fast EMA below slow EMA, price below VWAP
        buy_pe = (
            in_session
            & warmed
            & (minus_di > plus_di)
            & (adx > adx_threshold)
            & adx_rising
            & (ema_fast < ema_slow)
            & (close < vwap)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
