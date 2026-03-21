"""Regime Detection Vol v1 — volatility regime transition strategy for NIFTY 5s options.

Detects low→high vol transitions (momentum) and high→low vol transitions (mean-reversion)
by comparing short-term (3-min) vs medium-term (10-min) realized vol. Approximates the
original HMM regime-probability with a simple vol ratio, enabling real-time 5-second
computation without forward-algorithm complexity.

Original: HMM 2-state on 120-bar 1-min NIFTY50 constituents.
Adapted: vol ratio proxy on NIFTY index 5s bars. Two signal legs preserved:
  - low→high transition + EMA alignment → momentum (buy CE or PE)
  - high→low transition + price vs VWAP → mean reversion (buy CE or PE)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation with loop (Numba-compatible pattern)."""
    n = len(arr)
    out = np.zeros(n)
    for i in range(window, n):
        out[i] = np.std(arr[i - window:i])
    return out


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    k = 2.0 / (period + 1)
    out = np.empty(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP, reset per day."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -999999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_v += volume[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


class Strategy(BaseStrategy):
    name = "regime_detection_vol_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — needs 120-bar rv_medium warmup after open
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 5
    max_lookback = 120            # 120 bars × 5s = 10 min warmup for rv_medium

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Vol ratio threshold above which we declare high-vol regime
            TunableParam("vol_ratio_threshold", 2.0, 1.4, 3.5),
            # Minimum VWAP displacement (fractional) to qualify for reversion entry
            TunableParam("reversion_gap_threshold", 0.0008, 0.0003, 0.0020),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays ──────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ──────────────────────────────────────────────────────
        vol_ratio_thr = params.get("vol_ratio_threshold", 2.0)
        rev_gap_thr = params.get("reversion_gap_threshold", 0.0008)

        # ── Log returns ─────────────────────────────────────────────────────
        log_ret = np.zeros(n)
        log_ret[1:] = np.log(close[1:] / np.where(close[:-1] > 0, close[:-1], 1.0))

        # ── Realized vol (annualization factor irrelevant for ratio) ─────────
        rv_short = _rolling_std(log_ret, 36)   # 3-min burst window
        rv_medium = _rolling_std(log_ret, 120)  # 10-min baseline

        # ── Vol ratio: proxy for HMM high-vol probability ────────────────────
        vol_ratio = np.ones(n)
        for i in range(120, n):
            if rv_medium[i] > 1e-10:
                vol_ratio[i] = rv_short[i] / rv_medium[i]
            else:
                vol_ratio[i] = 1.0

        # ── Regime state and transitions ─────────────────────────────────────
        high_vol = vol_ratio > vol_ratio_thr

        # Transition on current bar vs previous bar
        low_to_high = np.zeros(n, dtype=bool)
        high_to_low = np.zeros(n, dtype=bool)
        low_to_high[1:] = high_vol[1:] & ~high_vol[:-1]
        high_to_low[1:] = ~high_vol[1:] & high_vol[:-1]

        # ── EMA direction filter ─────────────────────────────────────────────
        ema_fast = _ema(close, 24)   # 2-min EMA
        ema_slow = _ema(close, 72)   # 6-min EMA

        # ── Session VWAP ─────────────────────────────────────────────────────
        vwap = _compute_vwap(close, volume, day_id)

        # ── Session filter ───────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Signal construction ──────────────────────────────────────────────
        # Momentum leg: vol burst in bullish direction
        ce_momentum = low_to_high & (ema_fast > ema_slow)
        # Momentum leg: vol burst in bearish direction
        pe_momentum = low_to_high & (ema_fast < ema_slow)

        # Reversion leg: vol collapse + price below VWAP (expect snap up)
        vwap_safe = np.where(vwap > 0, vwap, close)
        ce_reversion = high_to_low & (close < vwap_safe * (1.0 - rev_gap_thr))
        # Reversion leg: vol collapse + price above VWAP (expect snap down)
        pe_reversion = high_to_low & (close > vwap_safe * (1.0 + rev_gap_thr))

        buy_ce = in_session & (ce_momentum | ce_reversion)
        buy_pe = in_session & (pe_momentum | pe_reversion)

        # Mutual exclusion: if both fired on same bar, suppress
        both = buy_ce & buy_pe
        buy_ce = buy_ce & ~both
        buy_pe = buy_pe & ~both

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),    # 4 pts = ~8 spot pts; catches false regime signals
            target_points=np.full(n, 7.0),  # 7 pts = ~14 spot pts; ~60% of median vol burst move
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,              # 120s max hold; regime transitions play out within 2 min
            max_trades_per_day=self.max_trades_per_day,
        )
