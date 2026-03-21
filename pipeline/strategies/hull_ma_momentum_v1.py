"""Hull MA Momentum Strategy — 5-second NIFTY index options.

Mechanism: Hull Moving Average (HMA) slope flips on a 2-minute window detect
micro-momentum shifts driven by VWAP algo re-benchmarking and delta-hedging
flows on NIFTY. HMA's reduced lag (~50% vs EMA) catches the flip ~30-60 seconds
earlier, allowing entry at the start of the directional burst. VWAP alignment
ensures we enter only when institutional buy/sell pressure dominates.

Original: hull_ma_momentum_v1 equity strategy on NIFTY200 1-min bars.
Converted: 5s NIFTY index options, HMA(24)=2min trigger, VWAP context filter.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _wma(arr: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average — weights linearly increase to most recent bar."""
    n = len(arr)
    result = np.full(n, np.nan)
    if period < 1 or n < period:
        return result
    weights = np.arange(1, period + 1, dtype=np.float64)
    denom = weights.sum()
    for i in range(period - 1, n):
        result[i] = np.dot(arr[i - period + 1: i + 1], weights) / denom
    return result


def _hma(close: np.ndarray, n: int) -> np.ndarray:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half_n = max(1, n // 2)
    sqrt_n = max(1, int(round(n ** 0.5)))
    wma_half = _wma(close, half_n)
    wma_full = _wma(close, n)
    intermediate = 2.0 * wma_half - wma_full
    return _wma(intermediate, sqrt_n)


class Strategy(BaseStrategy):
    name = "hull_ma_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — allow HMA warmup + skip open noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup (covers HMA(24)+vol_sma_60 needs)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("hma_threshold", 0.5, 0.1, 2.0),   # min abs HMA slope (index pts)
            TunableParam("stop_pts", 5.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before extracting numpy arrays
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high_ = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low_ = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Session VWAP (cumulative per day from 09:15) ──────────────────────
        typical = (high_ + low_ + close) / 3.0
        vwap = np.full(n, np.nan)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            cum_tp_vol += typical[i] * volume[i]
            cum_vol += volume[i]
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else close[i]

        # ── HMA(24) — 2-minute momentum signal at 5s resolution ──────────────
        hma = _hma(close, 24)

        # HMA slope: difference between consecutive HMA values
        hma_slope = np.zeros(n)
        valid_hma = ~np.isnan(hma)
        hma_slope[1:] = np.where(
            valid_hma[1:] & valid_hma[:-1],
            hma[1:] - hma[:-1],
            0.0,
        )

        # ── HMA slope flip detection ──────────────────────────────────────────
        # flip_up[i]: slope was <= 0 at i-1 and > 0 at i
        # flip_dn[i]: slope was >= 0 at i-1 and < 0 at i
        slope_flip_up = np.zeros(n, dtype=bool)
        slope_flip_dn = np.zeros(n, dtype=bool)
        slope_flip_up[1:] = (hma_slope[1:] > 0) & (hma_slope[:-1] <= 0)
        slope_flip_dn[1:] = (hma_slope[1:] < 0) & (hma_slope[:-1] >= 0)

        # ── Choppiness filter: count flips in rolling 30-bar window (150s) ────
        flip_any = (slope_flip_up | slope_flip_dn).astype(np.float64)
        flip_count_30 = np.zeros(n)
        for i in range(30, n):
            flip_count_30[i] = flip_any[i - 30: i].sum()

        # ── Volume filter: 5-minute rolling average (60 bars × 5s) ───────────
        vol_sma_60 = np.full(n, np.nan)
        for i in range(60, n):
            vol_sma_60[i] = volume[i - 60: i].mean()

        # ── Parameters ────────────────────────────────────────────────────────
        hma_threshold = params.get("hma_threshold", 0.5)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── Filters ───────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        hma_valid = ~np.isnan(hma)
        vwap_valid = ~np.isnan(vwap)
        vol_ok = (volume >= vol_sma_60) & ~np.isnan(vol_sma_60)
        not_choppy = flip_count_30 < 5

        # Slope magnitude must clear threshold (filters micro-noise flips)
        slope_strong = np.abs(hma_slope) >= hma_threshold

        # ── Entry signals ─────────────────────────────────────────────────────
        # Buy CE: HMA just flipped bullish, price above VWAP, volume confirming
        buy_ce = (
            in_session
            & hma_valid
            & vwap_valid
            & slope_flip_up
            & slope_strong
            & (close > vwap)
            & vol_ok
            & not_choppy
        )

        # Buy PE: HMA just flipped bearish, price below VWAP, volume confirming
        buy_pe = (
            in_session
            & hma_valid
            & vwap_valid
            & slope_flip_dn
            & slope_strong
            & (close < vwap)
            & vol_ok
            & not_choppy
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
