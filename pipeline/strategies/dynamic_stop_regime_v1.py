"""dynamic_stop_regime_v1 — EMA crossover + VWAP with ATR-regime-adaptive stops on NIFTY.

Thesis: On NIFTY, when the 60-second EMA crosses above/below the 3-minute EMA with VWAP
confirmation, it signals short-term institutional order flow is accelerating. The key
adaptation is dynamic stop sizing: NIFTY's 10-minute ATR percentile classifies the current
microstructure volatility regime, and option premium stops are sized accordingly.
Low-vol regime (pctile < 30): 3pt stop, 5pt target.
Med-vol regime (30-70): 5pt stop, 8pt target.
High-vol regime (>70): 7pt stop, 12pt target.

Original: Strategy_318.json — EMA(10/25) crossover + VWAP on NIFTY50 stocks, 1-min bars.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr, dtype=np.float64)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Average True Range via EMA of True Range."""
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    return _ema(tr, period)


def _percentile_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling percentile rank in [0, 100]."""
    n = len(arr)
    out = np.full(n, 50.0, dtype=np.float64)
    for i in range(window, n):
        w = arr[i - window : i]
        out[i] = float(np.sum(w < arr[i])) / window * 100.0
    return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    cumsum = np.cumsum(arr)
    out[window:] = (cumsum[window:] - cumsum[:-window]) / window
    # For bars before window, use expanding mean
    for i in range(1, window):
        out[i] = np.mean(arr[:i+1])
    return out


class Strategy(BaseStrategy):
    name = "dynamic_stop_regime_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_lookback = 240            # 20 min warmup (240 × 5s) — needs 120 bars for percentile + EMA seed
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("atr_low_pctile",  30.0, 15.0, 45.0),
            TunableParam("atr_high_pctile", 70.0, 55.0, 85.0),
            TunableParam("stop_low_vol",     3.0,  2.0,  5.0),
            TunableParam("stop_med_vol",     5.0,  3.0,  8.0),
            TunableParam("stop_high_vol",    7.0,  5.0, 12.0),
            TunableParam("target_low_vol",   5.0,  3.0,  8.0),
            TunableParam("target_med_vol",   8.0,  5.0, 12.0),
            TunableParam("target_high_vol", 12.0,  8.0, 20.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before numpy) ──
        close   = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high    = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low     = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume  = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        # ── VWAP (cumulative per session day, reset at each new day_id) ──
        typical = (high + low + close) / 3.0
        vwap = np.empty(n, dtype=np.float64)
        cum_tpv = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_tpv = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            cum_tpv += typical[i] * volume[i]
            cum_vol += volume[i]
            vwap[i] = cum_tpv / cum_vol if cum_vol > 0.0 else close[i]

        # ── EMA crossover: fast=12 (60s), slow=36 (3 min) ──
        ema_fast = _ema(close, 12)
        ema_slow = _ema(close, 36)

        # Cross detection: bullish cross = fast crossed above slow THIS bar
        cross_up = np.zeros(n, dtype=bool)
        cross_dn = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if ema_fast[i - 1] <= ema_slow[i - 1] and ema_fast[i] > ema_slow[i]:
                cross_up[i] = True
            if ema_fast[i - 1] >= ema_slow[i - 1] and ema_fast[i] < ema_slow[i]:
                cross_dn[i] = True

        # ── ATR(36) = 3-minute ATR for regime classification ──
        atr_36 = _atr(high, low, close, 36)

        # ── ATR percentile rank over 120-bar window (10 min) ──
        atr_pctile = _percentile_rank(atr_36, 120)

        # ── Volume filter: volume > 0.8 × SMA(volume, 75) = 6.25 min ──
        vol_sma_75 = _rolling_mean(volume, 75)
        vol_ok = volume > (vol_sma_75 * 0.8)

        # ── Session and warmup filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed_up  = np.arange(n) >= self.max_lookback

        # ── Params ──
        atr_low  = params.get("atr_low_pctile",  30.0)
        atr_high = params.get("atr_high_pctile", 70.0)
        stop_low   = params.get("stop_low_vol",   3.0)
        stop_med   = params.get("stop_med_vol",   5.0)
        stop_hgh   = params.get("stop_high_vol",  7.0)
        tgt_low    = params.get("target_low_vol",  5.0)
        tgt_med    = params.get("target_med_vol",  8.0)
        tgt_hgh    = params.get("target_high_vol", 12.0)

        # ── Dynamic stops & targets by ATR regime ──
        stop_pts = np.where(
            atr_pctile < atr_low, stop_low,
            np.where(atr_pctile > atr_high, stop_hgh, stop_med)
        ).astype(np.float64)

        target_pts = np.where(
            atr_pctile < atr_low, tgt_low,
            np.where(atr_pctile > atr_high, tgt_hgh, tgt_med)
        ).astype(np.float64)

        # ── Entry signals ──
        base_filter = in_session & warmed_up & vol_ok

        buy_ce = base_filter & cross_up & (close > vwap)
        buy_pe = base_filter & cross_dn & (close < vwap)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_pts,
            target_points=target_pts,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
