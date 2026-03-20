"""vix_filtered_index_trend_v133 — VIX Acceleration + EMA Crossover on NIFTY.

Differentiated from v124 (static VIX>21) and v126 (VIX 12-22 band) by using
VIX rate-of-change as the regime filter: detects when VIX is actively accelerating
(options demand surging), forcing market-maker delta-hedging that drives NIFTY
directional flow. Combines with 2-min/7-min EMA crossover and VWAP direction.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    result = np.empty(len(arr), dtype=np.float64)
    k = 2.0 / (period + 1)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1.0 - k)
    return result


def _session_vwap(
    typical: np.ndarray, volume: np.ndarray, day_id: np.ndarray
) -> np.ndarray:
    """Cumulative session VWAP, resets each new day_id."""
    n = len(typical)
    vwap = np.empty(n, dtype=np.float64)
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
        vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else typical[i]
    return vwap


class Strategy(BaseStrategy):
    name = "vix_filtered_index_trend_v133"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip opening 15-min noise
    session_end_minutes = 920     # 15:20 IST — flatten before EOD
    max_trades_per_day = 8
    max_lookback = 120            # 10-min warmup covers EMA(84) settling period

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_min", 17.0, 13.0, 25.0),
            TunableParam("vix_roc_threshold", 0.05, 0.01, 0.20),
            TunableParam("ema_gap_threshold", 0.0002, 0.0001, 0.001),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (forward-fill NaN in Polars first) ──────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ─────────────────────────────────────────────────────────
        vix_min = params.get("vix_min", 17.0)
        vix_roc_thr = params.get("vix_roc_threshold", 0.05)
        ema_gap_thr = params.get("ema_gap_threshold", 0.0002)

        # ── VIX aligned to spot bars (backward join_asof) ──────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_aligned = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_aligned["vix_close"].fill_null(15.0).to_numpy()

        # ── VIX rate-of-change over 24 bars (2 minutes) ────────────────────────
        vix_roc = np.zeros(n, dtype=np.float64)
        for i in range(24, n):
            base = vix_close[i - 24]
            if base > 0.0:
                vix_roc[i] = (vix_close[i] - base) / base

        # ── EMA fast (24 bars = 2 min) and slow (84 bars = 7 min) ─────────────
        ema_fast = _ema(close, 24)
        ema_slow = _ema(close, 84)
        ema_diff = ema_fast - ema_slow

        # ── Session VWAP ───────────────────────────────────────────────────────
        typical = (high + low + close) / 3.0
        vwap = _session_vwap(typical, volume, day_id)

        # ── EMA crossover detection with gap threshold ─────────────────────────
        bullish_cross = np.zeros(n, dtype=bool)
        bearish_cross = np.zeros(n, dtype=bool)
        for i in range(1, n):
            gap = ema_gap_thr * close[i]
            if ema_diff[i - 1] <= 0.0 and ema_diff[i] > gap:
                bullish_cross[i] = True
            elif ema_diff[i - 1] >= 0.0 and ema_diff[i] < -gap:
                bearish_cross[i] = True

        # ── Regime and session filters ─────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        vix_above_min = vix_close >= vix_min
        vix_rising = vix_roc >= vix_roc_thr

        # ── Entry signals ──────────────────────────────────────────────────────
        buy_ce = in_session & bullish_cross & vix_above_min & vix_rising & (close > vwap)
        buy_pe = in_session & bearish_cross & vix_above_min & vix_rising & (close < vwap)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 5.0),
            target_points=np.full(n, 8.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
