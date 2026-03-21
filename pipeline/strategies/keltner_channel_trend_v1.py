"""keltner_channel_trend_v1 — Keltner Channel Trend-Following on NIFTY 5s bars.

Mechanism: When NIFTY sustains above/below its 5-minute Keltner Channel (EMA(60) ± 2×ATR(60))
for 3 consecutive 5s bars, it signals institutional directional flow has overwhelmed the
ATR-calibrated volatility band. ADX(36) (3-min) confirms trend strength. We ride the
continuation for 30-120s targeting 7 option pts.

Converted from: trading_strategies/unique_strategies_all/Strategy_83.json (equity 1-min trend-following)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """EMA using multiplier method. Returns NaN for warmup bars."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = np.mean(arr[:period])
    k = 2.0 / (period + 1)
    for i in range(period, n):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ATR using Wilder smoothing. Returns NaN for warmup bars."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    out = np.full(n, np.nan)
    if n < period:
        return out
    out[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ADX using Wilder smoothing. Returns NaN for warmup bars."""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        move_up = high[i] - high[i - 1]
        move_down = low[i - 1] - low[i]
        plus_dm[i] = move_up if (move_up > move_down and move_up > 0.0) else 0.0
        minus_dm[i] = move_down if (move_down > move_up and move_down > 0.0) else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    def _wilder_smooth(arr: np.ndarray, p: int) -> np.ndarray:
        s = np.full(n, np.nan)
        if n < p + 1:
            return s
        s[p] = np.sum(arr[1 : p + 1])
        for i in range(p + 1, n):
            s[i] = s[i - 1] - s[i - 1] / p + arr[i]
        return s

    tr_s = _wilder_smooth(tr, period)
    pdm_s = _wilder_smooth(plus_dm, period)
    mdm_s = _wilder_smooth(minus_dm, period)

    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di = np.where(tr_s > 0.0, 100.0 * pdm_s / tr_s, 0.0)
        minus_di = np.where(tr_s > 0.0, 100.0 * mdm_s / tr_s, 0.0)
        denom = plus_di + minus_di
        dx = np.where(denom > 0.0, 100.0 * np.abs(plus_di - minus_di) / denom, 0.0)

    # ADX is Wilder smoothing of DX, starting at 2*period
    adx_out = np.full(n, np.nan)
    start = 2 * period
    if n <= start:
        return adx_out
    valid_dx = dx[period : start + 1]
    adx_out[start] = np.nanmean(valid_dx)
    for i in range(start + 1, n):
        adx_out[i] = (adx_out[i - 1] * (period - 1) + dx[i]) / period
    return adx_out


def _session_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Cumulative session VWAP, resetting each day."""
    n = len(close)
    vwap = close.copy()
    cum_tpv = 0.0
    cum_vol = 0.0
    prev_day = -1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_tpv = 0.0
            cum_vol = 0.0
            prev_day = int(day_id[i])
        tp = (high[i] + low[i] + close[i]) / 3.0
        vol = float(volume[i]) if volume[i] > 0 else 0.0
        cum_tpv += tp * vol
        cum_vol += vol
        vwap[i] = cum_tpv / cum_vol if cum_vol > 0.0 else close[i]
    return vwap


class Strategy(BaseStrategy):
    name = "keltner_channel_trend_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min opening noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 144            # 12 min warmup: covers EMA(60) + ADX(2×36=72)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_threshold", 22.0, 15.0, 35.0),
            TunableParam("kc_multiplier", 2.0, 1.5, 3.0),
            TunableParam("stop_pts", 5.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Extract and forward-fill spot OHLCV
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        day_id = spot_df["day_id"].fill_null(0).to_numpy().astype(np.int32)
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy()

        adx_threshold = params.get("adx_threshold", 22.0)
        kc_mult = params.get("kc_multiplier", 2.0)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── Keltner Channel (5-min EMA + 5-min ATR) ──
        kc_mid = _ema(close, 60)
        atr60 = _atr(high, low, close, 60)

        # Fill NaN with neutral values: mid→close, atr→0 (channel collapses, no breach possible)
        kc_mid_f = np.where(np.isnan(kc_mid), close, kc_mid)
        atr60_f = np.where(np.isnan(atr60), 0.0, atr60)

        kc_upper = kc_mid_f + kc_mult * atr60_f
        kc_lower = kc_mid_f - kc_mult * atr60_f

        # ── ADX(36) — 3-minute trend strength ──
        adx36 = _adx(high, low, close, 36)
        adx36_f = np.where(np.isnan(adx36), 0.0, adx36)

        # ── Session VWAP ──
        vwap = _session_vwap(high, low, close, volume, day_id)

        # ── Session time filter ──
        in_session = (
            (time_min >= self.session_start_minutes) &
            (time_min < self.session_end_minutes)
        )

        # ── 3-bar persistence: price must hold outside channel for 3 consecutive 5s bars ──
        above_upper = close > kc_upper
        below_lower = close < kc_lower

        persist_above = np.zeros(n, dtype=bool)
        persist_below = np.zeros(n, dtype=bool)
        for i in range(2, n):
            persist_above[i] = above_upper[i] and above_upper[i - 1] and above_upper[i - 2]
            persist_below[i] = below_lower[i] and below_lower[i - 1] and below_lower[i - 2]

        # ── Entry signals ──
        buy_ce = (
            in_session &
            persist_above &
            (adx36_f > adx_threshold) &
            (close > vwap)
        )

        buy_pe = (
            in_session &
            persist_below &
            (adx36_f > adx_threshold) &
            (close < vwap)
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
