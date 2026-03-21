"""VWAP Volume Divergence Strategy (5-second NIFTY index options).

When NIFTY drifts 1.5+ standard deviations below session VWAP while 5-second
bar volumes decline AND OBV rises (buyers absorbing at depressed price), the
sell-flow is exhausted and a 8-15 spot point snap-back toward VWAP is expected
within 30-90 seconds. Symmetric logic for bearish divergence above VWAP.

Original: equity VWAP+volume-divergence on NIFTY200 stocks, 1-min bars, 15-50 min hold.
Adapted: NIFTY index 5-second bars, 15-90 second hold, z-score deviation.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Cumulative session VWAP, resetting each trading day."""
    n = len(close)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_vol = 0.0
    prev_day = -999999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        typical = (high[i] + low[i] + close[i]) / 3.0
        cum_pv += typical * volume[i]
        cum_vol += volume[i]
        vwap[i] = cum_pv / cum_vol if cum_vol > 0.0 else close[i]
    return vwap


def _compute_obv(close: np.ndarray, volume: np.ndarray,
                 day_id: np.ndarray) -> np.ndarray:
    """On-Balance Volume, resetting each trading day."""
    n = len(close)
    obv = np.zeros(n)
    cum_obv = 0.0
    prev_day = -999999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_obv = 0.0
            prev_day = day_id[i]
        elif close[i] > close[i - 1]:
            cum_obv += volume[i]
        elif close[i] < close[i - 1]:
            cum_obv -= volume[i]
        obv[i] = cum_obv
    return obv


class Strategy(BaseStrategy):
    name = "vwap_volume_divergence_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip opening 15-min noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10-min warmup: enough VWAP dev history for z-score

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_zscore_threshold", 1.5, 1.0, 2.5),
            TunableParam("vol_ratio_threshold", 0.85, 0.60, 1.00),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
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

        # ── Extract spot arrays (forward-fill before numpy conversion) ──────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_id = spot_df["day_id"].fill_null(0).to_numpy()
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        zscore_thresh = params.get("vwap_dev_zscore_threshold", 1.5)
        vol_ratio_thresh = params.get("vol_ratio_threshold", 0.85)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # ── VWAP (cumulative, resets each session) ────────────────────────────
        vwap = _compute_vwap(high, low, close, volume, day_id)
        vwap_dev = close - vwap   # deviation in spot points

        # ── VWAP deviation z-score (120-bar = 10-min rolling window) ─────────
        # Normalises by current-session volatility so the threshold is regime-stable.
        DEV_WIN = 120
        dev_zscore = np.zeros(n)
        for i in range(DEV_WIN, n):
            window = vwap_dev[i - DEV_WIN:i]
            std = np.std(window)
            if std > 0.1:   # guard against near-zero std on flat tape
                dev_zscore[i] = vwap_dev[i] / std

        # ── Volume ratio: current bar vs 24-bar (2-min) SMA ──────────────────
        # vol_ratio < vol_ratio_thresh  →  selling/buying is losing volume = exhaustion.
        VOL_WIN = 24
        vol_sma = np.zeros(n)
        for i in range(VOL_WIN, n):
            vol_sma[i] = np.mean(volume[i - VOL_WIN:i])

        vol_ratio = np.zeros(n)
        for i in range(VOL_WIN, n):
            if vol_sma[i] > 0.0:
                vol_ratio[i] = volume[i] / vol_sma[i]
            else:
                vol_ratio[i] = 1.0   # neutral when volume history unavailable

        # ── OBV and its 24-bar slope ──────────────────────────────────────────
        # obv_slope > 0 while price < VWAP  →  buyers absorbing = bullish divergence.
        # obv_slope < 0 while price > VWAP  →  sellers absorbing = bearish divergence.
        obv = _compute_obv(close, volume, day_id)
        obv_slope = np.zeros(n)
        for i in range(VOL_WIN, n):
            obv_slope[i] = obv[i] - obv[i - VOL_WIN]

        # ── VIX filter (skip high-volatility, gap-prone sessions) ─────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")])
                      .sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        vix_ok = vix_close < 25.0

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Signals ───────────────────────────────────────────────────────────
        # Intrabar directional confirmation (green/red candle forming)
        bar_up = close > open_
        bar_dn = close < open_

        # Bullish divergence: price well below VWAP, volume declining, OBV rising
        buy_ce = (
            in_session
            & vix_ok
            & (dev_zscore < -zscore_thresh)
            & (vol_ratio < vol_ratio_thresh)
            & (obv_slope > 0.0)
            & bar_up
        )

        # Bearish divergence: price well above VWAP, volume declining, OBV falling
        buy_pe = (
            in_session
            & vix_ok
            & (dev_zscore > zscore_thresh)
            & (vol_ratio < vol_ratio_thresh)
            & (obv_slope < 0.0)
            & bar_dn
        )

        # Mutual exclusion (should not overlap, but guard anyway)
        both = buy_ce & buy_pe
        buy_ce = buy_ce & ~both
        buy_pe = buy_pe & ~both

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds — snap-back window
            max_trades_per_day=self.max_trades_per_day,
        )
