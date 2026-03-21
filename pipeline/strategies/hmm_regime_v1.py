"""hmm_regime_v1 — Regime-Adaptive NIFTY Options Strategy

Detects trending vs. mean-reverting microstructure regimes on NIFTY using a
directional consistency measure (rolling sign autocorrelation proxy for HMM state).

In trending regime (institutional TWAP/VWAP flow): follow 2-min/6-min EMA crossover.
In mean-reverting regime (market-maker dominated): fade z-score extremes.
Choppy/mixed regime (mid-range consistency): no trade.

Adapted from Strategy_304.json (3-state HMM on NIFTY50 constituents, 1-min bars).
Full HMM inference replaced with directional consistency — computationally feasible at 5s.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average via single-pass alpha smoothing."""
    alpha = 2.0 / (period + 1)
    out = np.empty(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rolling_zscore(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score: (arr[i] - mean(window)) / std(window). Returns 0.0 at warmup."""
    n = len(arr)
    out = np.zeros(n)
    for i in range(window, n):
        w = arr[i - window : i + 1]
        mu = np.mean(w)
        sigma = np.std(w)
        if sigma > 1e-6:
            out[i] = (arr[i] - mu) / sigma
    return out


def _directional_consistency(ret: np.ndarray, window: int) -> np.ndarray:
    """Fraction of consecutive-bar pairs with the same return sign over rolling window.

    Value near 1.0 → strongly trending (same direction repeatedly).
    Value near 0.0 → strongly mean-reverting (alternating directions).
    Value near 0.5 → choppy/random.
    """
    n = len(ret)
    sign = np.sign(ret)
    out = np.full(n, 0.5)
    for i in range(window, n):
        s = sign[i - window + 1 : i + 1]   # `window` sign values
        same_dir = np.sum(s[1:] == s[:-1])  # count consecutive same-direction pairs
        out[i] = same_dir / (window - 1)
    return out


class Strategy(BaseStrategy):
    name = "hmm_regime_v1"
    underlying = "NIFTY"
    session_start_minutes = 560    # 09:20 IST
    session_end_minutes = 925      # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 144             # 12 min warmup: covers EMA(72) + dir_consistency(36) + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Regime classification thresholds
            TunableParam("trend_threshold", 0.60, 0.50, 0.72),
            TunableParam("reversion_threshold", 0.42, 0.30, 0.50),
            # Trend sub-strategy: minimum EMA divergence to confirm direction
            TunableParam("ema_signal_threshold", 0.0003, 0.0001, 0.0008),
            # Mean-reversion sub-strategy: z-score extreme level
            TunableParam("zscore_threshold", 1.5, 1.0, 2.5),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()

        trend_thr = params.get("trend_threshold", 0.60)
        rev_thr = params.get("reversion_threshold", 0.42)
        ema_sig_thr = params.get("ema_signal_threshold", 0.0003)
        zscore_thr = params.get("zscore_threshold", 1.5)

        # ── 5-second returns ──────────────────────────────────────────────────────────
        ret = np.zeros(n)
        denom = np.where(close[:-1] != 0, close[:-1], 1.0)
        ret[1:] = (close[1:] - close[:-1]) / denom

        # ── Regime classifier: directional consistency over 36 bars (3 min) ──────────
        dir_cons = _directional_consistency(ret, 36)

        # ── Trending sub-strategy: EMA(24) vs EMA(72) crossover ──────────────────────
        ema_fast = _ema(close, 24)   # 2-min fast EMA
        ema_slow = _ema(close, 72)   # 6-min slow EMA
        safe_slow = np.where(ema_slow != 0, ema_slow, 1.0)
        ema_diff_frac = (ema_fast - ema_slow) / safe_slow   # fractional divergence

        # ── Mean-reversion sub-strategy: 5-min rolling z-score ───────────────────────
        zscore = _rolling_zscore(close, 60)

        # ── Session filter ────────────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Regime masks ──────────────────────────────────────────────────────────────
        trending = dir_cons > trend_thr
        reverting = dir_cons < rev_thr

        # ── Trending regime entries: follow institutional directional flow ─────────────
        # Bullish: fast EMA above slow by threshold AND last bar moved up
        trend_bull = trending & (ema_diff_frac > ema_sig_thr) & (ret > 0)
        # Bearish: fast EMA below slow by threshold AND last bar moved down
        trend_bear = trending & (ema_diff_frac < -ema_sig_thr) & (ret < 0)

        # ── Mean-reversion regime entries: fade z-score extremes ─────────────────────
        # Price below 5-min mean → expect market-maker push back up
        rev_bull = reverting & (zscore < -zscore_thr)
        # Price above 5-min mean → expect market-maker fade back down
        rev_bear = reverting & (zscore > zscore_thr)

        # ── Combine and apply session filter ─────────────────────────────────────────
        buy_ce = in_session & (trend_bull | rev_bull)
        buy_pe = in_session & (trend_bear | rev_bear)

        # Resolve conflicts (both triggered simultaneously → skip bar)
        conflict = buy_ce & buy_pe
        buy_ce = buy_ce & ~conflict
        buy_pe = buy_pe & ~conflict

        # ── Per-bar stops/targets: wider for trending, tighter for reversion ──────────
        # Trending: stop=4 pts (8 spot pts), target=7 pts (14 spot pts) — ~1:1.75 R:R
        # Reverting: stop=3 pts (6 spot pts), target=5 pts (10 spot pts) — ~1:1.67 R:R
        stop_pts = np.where(trending, 4.0, 3.0).astype(np.float64)
        target_pts = np.where(trending, 7.0, 5.0).astype(np.float64)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_pts,
            target_points=target_pts,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
