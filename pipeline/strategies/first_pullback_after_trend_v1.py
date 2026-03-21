"""first_pullback_after_trend_v1 — First EMA-pullback bounce after NIFTY trend establishment.

When NIFTY moves 0.3%+ from session open, large institutional TWAP/VWAP orders that
drove the move are only partially filled. The first time price retreats to the 3-min EMA
and then bounces back in the trend direction, those resting institutional orders absorb
the counter-trend selling and price resumes. We enter on the EMA re-cross (bounce bar).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    alpha = 2.0 / (period + 1.0)
    out = np.empty(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, 50.0)
    if n <= period:
        return out
    diff = np.diff(close)
    gains = np.where(diff > 0, diff, 0.0)
    losses = np.where(diff < 0, -diff, 0.0)
    avg_g = np.mean(gains[:period])
    avg_l = np.mean(losses[:period])
    for i in range(period, n):
        avg_g = (avg_g * (period - 1) + gains[i - 1]) / period
        avg_l = (avg_l * (period - 1) + losses[i - 1]) / period
        rs = avg_g / avg_l if avg_l > 1e-10 else 1e6
        out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _session_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    n = len(close)
    out = np.empty(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = day_id[0] - 1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        v = max(float(volume[i]), 0.0)
        cum_pv += close[i] * v
        cum_v += v
        out[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return out


def _day_open(open_px: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    n = len(open_px)
    out = np.empty(n)
    current_day = day_id[0] - 1
    current_open = open_px[0]
    for i in range(n):
        if day_id[i] != current_day:
            current_day = day_id[i]
            current_open = open_px[i]
        out[i] = current_open
    return out


class Strategy(BaseStrategy):
    name = "first_pullback_after_trend_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — 15 min after open allows trend to establish
    session_end_minutes = 900     # 15:00 IST
    max_trades_per_day = 3
    max_lookback = 240            # 20-min warmup for EMA(36) stability

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("trend_threshold_pct", 0.30, 0.15, 0.60),
            TunableParam("rsi_min", 35.0, 25.0, 50.0),
            TunableParam("rsi_max", 65.0, 55.0, 80.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_px = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        trend_threshold = params.get("trend_threshold_pct", 0.30) / 100.0
        rsi_min = params.get("rsi_min", 35.0)
        rsi_max = params.get("rsi_max", 65.0)

        # Indicators
        ema_36 = _ema(close, 36)
        rsi_36 = _rsi(close, 36)
        vwap = _session_vwap(close, volume, day_id)
        day_open_px = _day_open(open_px, day_id)

        # Return from session open (fractional)
        ret_from_open = (close - day_open_px) / np.where(day_open_px > 0, day_open_px, 1.0)

        # Session mask
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        # Per-day state machine for first-pullback detection
        # Bull: trend up → price drops through EMA → price re-crosses EMA upward → buy_ce
        # Bear: trend down → price rises through EMA → price re-crosses EMA downward → buy_pe
        current_day = -1
        bull_trend = False
        bear_trend = False
        bull_below_ema = False   # price has crossed below EMA (pullback started)
        bear_above_ema = False   # price has crossed above EMA (pullback started)
        bull_fired = False
        bear_fired = False

        warmup = 36  # need at least EMA(36) to be valid

        for i in range(warmup, n):
            if day_id[i] != current_day:
                current_day = day_id[i]
                bull_trend = False
                bear_trend = False
                bull_below_ema = False
                bear_above_ema = False
                bull_fired = False
                bear_fired = False

            if not in_session[i]:
                continue

            c = close[i]
            ema = ema_36[i]
            rsi = rsi_36[i]
            vw = vwap[i]
            rfopen = ret_from_open[i]

            # Establish trend (only once per direction per day)
            if not bull_trend and rfopen > trend_threshold:
                bull_trend = True
            if not bear_trend and rfopen < -trend_threshold:
                bear_trend = True

            # ── Bull trend: first pullback logic ──
            if bull_trend and not bull_fired:
                # Stage 1: detect pullback — price drops below EMA
                if not bull_below_ema and c < ema:
                    bull_below_ema = True

                # Stage 2: detect bounce — price re-crosses above EMA after pullback
                # Also require: still above VWAP and RSI not overextended
                if (bull_below_ema and
                        c > ema and
                        c > vw and
                        rsi_min <= rsi <= rsi_max):
                    buy_ce[i] = True
                    bull_fired = True

            # ── Bear trend: first pullback logic ──
            if bear_trend and not bear_fired:
                # Stage 1: detect pullback — price rises above EMA
                if not bear_above_ema and c > ema:
                    bear_above_ema = True

                # Stage 2: detect bounce — price re-crosses below EMA after pullback
                # Also require: still below VWAP and RSI not overextended
                if (bear_above_ema and
                        c < ema and
                        c < vw and
                        rsi_min <= rsi <= rsi_max):
                    buy_pe[i] = True
                    bear_fired = True

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 5),
            target_points=np.full(n, 8),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,        # 120-second max hold
            max_trades_per_day=3,
        )
