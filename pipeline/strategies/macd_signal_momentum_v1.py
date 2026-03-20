"""MACD Histogram Acceleration Momentum — NIFTY 5-second index options.

Thesis: When MACD histogram (EMA_24 minus EMA_72, net of signal EMA_18) accelerates
for 3 consecutive 5-second bars, it signals an institutional TWAP/VWAP order on NIFTY
is mid-execution and the directional flow is compounding. VWAP alignment confirms we're
trading with the dominant session benchmark.

Original: MACD(12,26,9) histogram acceleration on 1-min NIFTY200 stocks, hold 15-40 min.
Converted: MACD(24,72,18) on 5-second NIFTY index, hold 30-90s.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA. Handles leading NaN by starting from first valid block."""
    result = np.full(len(values), np.nan)
    # Find first index where we have 'period' consecutive non-NaN values
    first_valid = 0
    while first_valid < len(values) and np.isnan(values[first_valid]):
        first_valid += 1
    start = first_valid + period - 1
    if start >= len(values):
        return result
    segment = values[first_valid : start + 1]
    if np.any(np.isnan(segment)):
        return result
    result[start] = np.mean(segment)
    k = 2.0 / (period + 1)
    for i in range(start + 1, len(values)):
        if np.isnan(values[i]) or np.isnan(result[i - 1]):
            result[i] = np.nan
        else:
            result[i] = values[i] * k + result[i - 1] * (1.0 - k)
    return result


class Strategy(BaseStrategy):
    name = "macd_signal_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — first 15 min skipped for MACD warmup
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 180            # 15 minutes — covers EMA(72) + EMA(18) warmup with margin

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("hist_threshold", 0.0, -0.3, 0.3),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        stop_pts = float(params.get("stop_pts", 4.0))
        target_pts = float(params.get("target_pts", 7.0))
        hist_threshold = float(params.get("hist_threshold", 0.0))

        # ── MACD: Fast EMA(24)=2min, Slow EMA(72)=6min ──────────────────────
        ema_fast = _ema(close, 24)
        ema_slow = _ema(close, 72)

        macd_line = np.where(
            np.isnan(ema_fast) | np.isnan(ema_slow),
            np.nan,
            ema_fast - ema_slow,
        )

        # Signal: EMA(18) of MACD line; feed nan-safe copy to avoid skewing initialization
        macd_for_signal = np.where(np.isnan(macd_line), np.nan, macd_line)
        signal_line = _ema(macd_for_signal, 18)

        # Histogram
        hist = np.where(
            np.isnan(macd_line) | np.isnan(signal_line),
            np.nan,
            macd_line - signal_line,
        )

        # ── Session VWAP (cumulative, resets each day) ───────────────────────
        vwap = np.full(n, np.nan)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = int(day_id[i])
            cum_tp_vol += close[i] * volume[i]
            cum_vol += volume[i]
            if cum_vol > 0.0:
                vwap[i] = cum_tp_vol / cum_vol

        # ── Volume SMA(60) — 5-minute rolling average ────────────────────────
        vol_sma = np.full(n, np.nan)
        for i in range(60, n):
            vol_sma[i] = np.mean(volume[i - 60 : i])

        # ── Histogram acceleration: 3-bar monotone change ────────────────────
        hist_acc_bull = np.zeros(n, dtype=bool)
        hist_acc_bear = np.zeros(n, dtype=bool)
        for i in range(2, n):
            h0 = hist[i]
            h1 = hist[i - 1]
            h2 = hist[i - 2]
            if np.isnan(h0) or np.isnan(h1) or np.isnan(h2):
                continue
            if h0 > h1 and h1 > h2 and h0 > hist_threshold:
                hist_acc_bull[i] = True
            if h0 < h1 and h1 < h2 and h0 < -hist_threshold:
                hist_acc_bear[i] = True

        # ── Filters ──────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        macd_positive = ~np.isnan(macd_line) & (macd_line > 0.0)
        macd_negative = ~np.isnan(macd_line) & (macd_line < 0.0)

        above_vwap = ~np.isnan(vwap) & (close > vwap)
        below_vwap = ~np.isnan(vwap) & (close < vwap)

        # Soft volume filter: at least 80% of 5-min average
        vol_ok = ~np.isnan(vol_sma) & (volume >= 0.8 * vol_sma)

        # ── Signals ──────────────────────────────────────────────────────────
        buy_ce = in_session & hist_acc_bull & macd_positive & above_vwap & vol_ok
        buy_pe = in_session & hist_acc_bear & macd_negative & below_vwap & vol_ok

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
