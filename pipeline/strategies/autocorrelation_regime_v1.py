"""autocorrelation_regime_v1 — NIFTY 5-second options strategy.

Classifies NIFTY into momentum (autocorr > threshold) or mean-reversion
(autocorr < -threshold) regime using 5-minute rolling lag-1 autocorrelation
of 5-second returns. Trades directionally:
  - Momentum regime: follow the prevailing 1-minute impulse
  - Reversion regime: fade the 1-minute impulse back toward VWAP

Converted from Strategy_309 (equity 1-min autocorrelation regime).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    n = len(arr)
    ema = np.empty(n)
    if n == 0:
        return ema
    k = 2.0 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)
    return ema


def _compute_session_vwap(
    close: np.ndarray, volume: np.ndarray, day_id: np.ndarray
) -> np.ndarray:
    """Session-cumulative VWAP, reset at each new day."""
    n = len(close)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        v = volume[i]
        cum_pv += close[i] * v
        cum_v += v
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


def _rolling_lag1_autocorr(ret: np.ndarray, window: int) -> np.ndarray:
    """Rolling lag-1 Pearson autocorrelation over `window` bars.

    At each bar i (i >= window), computes corr(ret[i-window:i-1], ret[i-window+1:i]).
    Returns 0.0 where undefined or when std is too small.
    """
    n = len(ret)
    ac = np.zeros(n)
    for i in range(window, n):
        r = ret[i - window : i]        # length = window
        r1 = r[:-1]                    # lag-1 series
        r2 = r[1:]                     # lag-0 series
        m1, m2 = r1.mean(), r2.mean()
        s1 = r1.std()
        s2 = r2.std()
        if s1 > 1e-12 and s2 > 1e-12:
            ac[i] = np.mean((r1 - m1) * (r2 - m2)) / (s1 * s2)
    return ac


class Strategy(BaseStrategy):
    name = "autocorrelation_regime_v1"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — 5-min warmup after open for stable autocorr
    session_end_minutes = 920      # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120             # 10-min warmup (120 × 5s) for first stable estimate

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("autocorr_threshold", 0.20, 0.12, 0.35),
            TunableParam("momentum_ret_threshold", 0.0012, 0.0005, 0.0030),
            TunableParam("reversion_ret_threshold", 0.0015, 0.0008, 0.0040),
            TunableParam("stop_momentum", 3.0, 2.0, 6.0),
            TunableParam("target_momentum", 5.0, 3.0, 10.0),
            TunableParam("stop_reversion", 4.0, 2.0, 7.0),
            TunableParam("target_reversion", 7.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN in Polars before numpy) ──────────
        close = (
            spot_df["close"].fill_null(strategy="forward").fill_null(0.0).to_numpy()
        )
        volume = (
            spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        )
        day_id = spot_df["day_id"].fill_null(0).to_numpy()
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        ac_thr = params.get("autocorr_threshold", 0.20)
        mom_thr = params.get("momentum_ret_threshold", 0.0012)
        rev_thr = params.get("reversion_ret_threshold", 0.0015)
        stop_mom = params.get("stop_momentum", 3.0)
        tgt_mom = params.get("target_momentum", 5.0)
        stop_rev = params.get("stop_reversion", 4.0)
        tgt_rev = params.get("target_reversion", 7.0)

        # ── Indicators ────────────────────────────────────────────────────────

        # 1-bar (5s) returns
        ret_1 = np.zeros(n)
        denom = np.where(close[:-1] > 0.0, close[:-1], 1.0)
        ret_1[1:] = (close[1:] - close[:-1]) / denom

        # Rolling 5-min (60-bar) lag-1 autocorrelation — regime classifier
        autocorr = _rolling_lag1_autocorr(ret_1, window=60)

        # 12-bar (1-min) return for entry signal
        ret_12 = np.zeros(n)
        denom12 = np.where(close[:n - 12] > 0.0, close[:n - 12], 1.0)
        ret_12[12:] = (close[12:] - close[:n - 12]) / denom12

        # 12-bar EMA for trend direction
        ema_12 = _compute_ema(close, 12)

        # Session-cumulative VWAP
        vwap = _compute_session_vwap(close, volume, day_id)

        # ── Regime classification ─────────────────────────────────────────────
        is_momentum = autocorr > ac_thr        # positive autocorr → momentum
        is_reversion = autocorr < -ac_thr      # negative autocorr → reversion

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        # Momentum regime: follow 1-min directional impulse
        mom_bull = (
            in_session & is_momentum & (ret_12 > mom_thr) & (close > ema_12)
        )
        mom_bear = (
            in_session & is_momentum & (ret_12 < -mom_thr) & (close < ema_12)
        )

        # Reversion regime: fade 1-min impulse back toward VWAP
        rev_bull = (
            in_session & is_reversion & (ret_12 < -rev_thr) & (close < vwap)
        )
        rev_bear = (
            in_session & is_reversion & (ret_12 > rev_thr) & (close > vwap)
        )

        buy_ce = mom_bull | rev_bull
        buy_pe = mom_bear | rev_bear

        # ── Per-bar stop/target arrays ────────────────────────────────────────
        # Momentum: tight 3/5 (fast regime); Reversion: wider 4/7 (final flush room)
        is_mom_signal = mom_bull | mom_bear
        is_rev_signal = rev_bull | rev_bear

        stop_arr = np.where(
            is_mom_signal, stop_mom,
            np.where(is_rev_signal, stop_rev, 3.0)
        ).astype(np.float64)

        target_arr = np.where(
            is_mom_signal, tgt_mom,
            np.where(is_rev_signal, tgt_rev, 5.0)
        ).astype(np.float64)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_arr,
            target_points=target_arr,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
