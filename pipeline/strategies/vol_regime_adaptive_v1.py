"""
vol_regime_adaptive_v1 — Adaptive Volatility Regime Strategy

On NIFTY, intraday volatility clusters in two structurally distinct regimes visible at
5-second resolution. In LOW-vol regimes (ATR-14 below 30th percentile of rolling 240-bar
intraday distribution), passive market maker inventory cycling dominates and BB deviations
below VWAP attract limit-order replenishment within 15-30 seconds. In HIGH-vol regimes
(ATR-14 above 70th percentile), institutional TWAP/momentum algos work directional orders
creating EMA-aligned bursts that persist 30-90 seconds. Flat in ambiguous MED-vol transitions.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR using Wilder smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA using standard 2/(period+1) smoothing factor."""
    n = len(arr)
    ema = np.zeros(n)
    if n == 0:
        return ema
    k = 2.0 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1 - k)
    return ema


def _compute_bb(close: np.ndarray, period: int, n_std: float):
    """Bollinger Bands — returns (upper, mid, lower)."""
    n = len(close)
    upper = np.zeros(n)
    lower = np.zeros(n)
    mid = np.zeros(n)
    for i in range(period - 1, n):
        window = close[i - period + 1 : i + 1]
        m = np.mean(window)
        s = np.std(window)
        mid[i] = m
        upper[i] = m + n_std * s
        lower[i] = m - n_std * s
    # Backfill pre-warmup with last valid values (neutral: bands == mid == close)
    for i in range(period - 1):
        upper[i] = close[i]
        lower[i] = close[i]
        mid[i] = close[i]
    return upper, mid, lower


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP — resets each trading day."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_v += volume[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


def _rolling_vol_sma(volume: np.ndarray, period: int) -> np.ndarray:
    """Rolling simple mean of volume over `period` bars."""
    n = len(volume)
    sma = np.zeros(n)
    for i in range(period, n):
        sma[i] = np.mean(volume[i - period : i])
    return sma


def _atr_percentile_rank(atr: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling percentile rank of current ATR vs. past `window` bars.
    Returns 0-100; 50 = neutral (used for warm-up bars).
    """
    n = len(atr)
    rank = np.full(n, 50.0)
    for i in range(window, n):
        hist = atr[i - window : i]
        valid = hist[hist > 0.0]
        if len(valid) >= 10:
            rank[i] = float(np.sum(valid < atr[i])) / len(valid) * 100.0
    return rank


def _regime_stability(flag: np.ndarray, stability_bars: int) -> np.ndarray:
    """
    Returns True at bar i only if flag[i-stability_bars : i+1] are ALL True.
    Prevents regime-transition whipsaws.
    """
    n = len(flag)
    stable = np.zeros(n, dtype=bool)
    for i in range(stability_bars, n):
        if np.all(flag[i - stability_bars : i + 1]):
            stable[i] = True
    return stable


class Strategy(BaseStrategy):
    name = "vol_regime_adaptive_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST
    session_end_minutes = 910     # 15:10 IST
    max_trades_per_day = 8
    max_lookback = 240            # 20-min warmup for ATR percentile window

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("atr_low_pct", 30.0, 15.0, 45.0),
            TunableParam("atr_high_pct", 70.0, 55.0, 85.0),
            TunableParam("bb_std", 1.5, 1.0, 2.5),
            TunableParam("vol_surge_mult", 1.5, 1.0, 3.0),
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
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)

        # ── Parameters ────────────────────────────────────────────────────────
        atr_low_pct = float(params.get("atr_low_pct", 30.0))
        atr_high_pct = float(params.get("atr_high_pct", 70.0))
        bb_std = float(params.get("bb_std", 1.5))
        vol_surge_mult = float(params.get("vol_surge_mult", 1.5))
        stability_bars = 10  # hard-coded: 50s minimum regime stability

        # ── VIX filter ────────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy().astype(np.float64)

        vix_ok = vix_close < 28.0

        # ── Indicators ────────────────────────────────────────────────────────
        atr14 = _compute_atr(high, low, close, 14)
        atr_pct = _atr_percentile_rank(atr14, window=240)

        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)

        bb_upper, _bb_mid, bb_lower = _compute_bb(close, 20, bb_std)

        vwap = _compute_vwap(close, volume, day_id)

        vol_sma20 = _rolling_vol_sma(volume, 20)
        vol_surge = (vol_sma20 > 0.0) & (volume > vol_surge_mult * vol_sma20)

        # ── Regime detection (with stability filter) ──────────────────────────
        raw_low = atr_pct < atr_low_pct
        raw_high = atr_pct > atr_high_pct

        stable_low = _regime_stability(raw_low, stability_bars)
        stable_high = _regime_stability(raw_high, stability_bars)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── LOW-VOL mean-reversion signals ───────────────────────────────────
        # Buy CE: price below BB_lower AND below VWAP → bounce expected
        buy_ce_lowvol = (
            in_session
            & vix_ok
            & stable_low
            & (close < bb_lower)
            & (close < vwap)
        )
        # Buy PE: price above BB_upper AND above VWAP → fade expected
        buy_pe_lowvol = (
            in_session
            & vix_ok
            & stable_low
            & (close > bb_upper)
            & (close > vwap)
        )

        # ── HIGH-VOL momentum signals ─────────────────────────────────────────
        # Buy CE: EMA9 > EMA21 (uptrend) AND price above VWAP AND volume surge
        buy_ce_highvol = (
            in_session
            & vix_ok
            & stable_high
            & (ema9 > ema21)
            & (close > vwap)
            & vol_surge
        )
        # Buy PE: EMA9 < EMA21 (downtrend) AND price below VWAP AND volume surge
        buy_pe_highvol = (
            in_session
            & vix_ok
            & stable_high
            & (ema9 < ema21)
            & (close < vwap)
            & vol_surge
        )

        buy_ce = buy_ce_lowvol | buy_ce_highvol
        buy_pe = buy_pe_lowvol | buy_pe_highvol

        # ── Regime-conditional stops/targets (option premium points) ──────────
        # LOW-VOL: BB deviations ~10-20 NIFTY spot pts → option pts 5-10 at delta 0.5
        #   Stop 3 pts, target 5 pts (1:1.67 R:R). Bounce thesis fails if gap widens.
        # HIGH-VOL: momentum bursts 20-40 NIFTY spot pts → option pts 10-20
        #   Stop 5 pts, target 8 pts (1:1.6 R:R). Immediate reversal negates thesis.
        stop_pts = np.where(stable_low, 3.0, np.where(stable_high, 5.0, 4.0)).astype(np.float64)
        target_pts = np.where(stable_low, 5.0, np.where(stable_high, 8.0, 6.0)).astype(np.float64)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_pts,
            target_points=target_pts,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,            # 60 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
