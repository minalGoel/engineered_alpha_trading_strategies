"""
low_vol_compression_long_v1
============================
Volatility compression breakout on NIFTY 5-second bars.

Mechanism:
    When NIFTY's 2-minute realized vol (rvol_24) drops to below 55% of the
    30-minute session baseline (rvol_360), institutional algorithms are in a
    passive, non-directional mode. The first large trigger causes a cascade of
    stop orders and momentum algos activating simultaneously, producing a rapid
    15-25 spot point expansion. VWAP + EMA(20) direction at the expansion bar
    determines CE vs PE entry.

Original: Strategy_44.json — 1-min realized-vol compression on Nifty50 stocks.
Converted to 5-second NIFTY index options with compressed lookbacks and
simplified compression metric (ratio vs percentile rank).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation, zero-padded for warmup bars."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    for i in range(window, n):
        out[i] = np.std(arr[i - window : i])
    return out


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    alpha = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _session_vwap(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                  volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP, reset at each new day_id."""
    n = len(close)
    vwap = np.zeros(n, dtype=np.float64)
    cum_pv = 0.0
    cum_vol = 0.0
    prev_day = -1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        typical = (high[i] + low[i] + close[i]) / 3.0
        vol_i = max(float(volume[i]), 1.0)
        cum_pv += typical * vol_i
        cum_vol += vol_i
        vwap[i] = cum_pv / cum_vol
    return vwap


class Strategy(BaseStrategy):
    name = "low_vol_compression_long_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min pre-open noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 360            # 30 min warmup (360 bars × 5s) for rvol_360

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("compression_threshold", 0.55, 0.35, 0.75),
            TunableParam("range_expansion_mult", 1.5, 1.2, 2.2),
            TunableParam("stop_pts", 5.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN before numpy) ──────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ───────────────────────────────────────────────────────
        compression_threshold = params.get("compression_threshold", 0.55)
        range_expansion_mult = params.get("range_expansion_mult", 1.5)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── Log returns (5-second) ────────────────────────────────────────────
        log_ret = np.zeros(n, dtype=np.float64)
        safe_close = np.maximum(close, 1e-9)
        log_ret[1:] = np.log(safe_close[1:] / safe_close[:-1])

        # ── Realized vol indicators ───────────────────────────────────────────
        # Fast: 24-bar (2 min) — compression detector
        rvol_24 = _rolling_std(log_ret, 24)
        # Baseline: 360-bar (30 min) — session regime reference
        rvol_360 = _rolling_std(log_ret, 360)

        # Compression ratio: <threshold → index is anomalously quiet
        compression_ratio = np.where(rvol_360 > 1e-9, rvol_24 / rvol_360, 1.0)

        # ── Session VWAP ──────────────────────────────────────────────────────
        vwap = _session_vwap(close, high, low, volume, day_id)

        # ── EMA(20) — 100-second short-term trend ────────────────────────────
        ema_20 = _ema(close, 20)

        # ── Bar range expansion confirmation ─────────────────────────────────
        bar_range = high - low
        avg_range_10 = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            avg_range_10[i] = np.mean(bar_range[i - 10 : i])

        # Current bar range must exceed mult × recent average range
        range_expanding = (avg_range_10 > 0) & (bar_range > range_expansion_mult * avg_range_10)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Compression condition ─────────────────────────────────────────────
        # Only valid after 360-bar warmup
        warmup_done = np.arange(n) >= 360
        vol_compressed = warmup_done & (compression_ratio < compression_threshold)

        # ── Directional bias ──────────────────────────────────────────────────
        above_vwap = close > vwap
        above_ema = close > ema_20

        # ── Entry signals ─────────────────────────────────────────────────────
        base_condition = in_session & vol_compressed & range_expanding

        # Bullish: above VWAP AND above EMA → buy CE
        buy_ce = base_condition & above_vwap & above_ema

        # Bearish: below VWAP AND below EMA → buy PE
        buy_pe = base_condition & (~above_vwap) & (~above_ema)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts, dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
