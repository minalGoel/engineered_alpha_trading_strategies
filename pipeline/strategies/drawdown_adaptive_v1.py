"""
drawdown_adaptive_v1 — Supertrend + VWAP with Session-Drawdown Regime Filter

Mechanism:
  On NIFTY, large institutional TWAP/VWAP algorithms create directional micro-trends
  detectable by Supertrend(2-min ATR). When NIFTY is in a bullish Supertrend AND
  above session VWAP, buy-side program flow is dominant and 30-60s continuation is
  high-probability. The drawdown-adaptive layer tracks NIFTY's session high-water
  mark: beyond 0.4% session drawdown, CE entries are suppressed (distribution regime);
  beyond 0.7%, all entries halt (liquidation/stress mode).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    multiplier: float,
) -> np.ndarray:
    """Returns trend direction array: +1 = bullish, -1 = bearish."""
    n = len(close)
    tr = np.zeros(n)
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.zeros(n)
    if n > period:
        atr[period] = np.mean(tr[1 : period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    hl2 = (high + low) / 2.0
    raw_upper = hl2 + multiplier * atr
    raw_lower = hl2 - multiplier * atr

    final_upper = raw_upper.copy()
    final_lower = raw_lower.copy()
    trend = np.ones(n, dtype=np.int8)

    for i in range(1, n):
        # Adjust lower band: only move up, never down (unless price breaks below)
        if raw_lower[i] > final_lower[i - 1] or close[i - 1] < final_lower[i - 1]:
            final_lower[i] = raw_lower[i]
        else:
            final_lower[i] = final_lower[i - 1]

        # Adjust upper band: only move down, never up (unless price breaks above)
        if raw_upper[i] < final_upper[i - 1] or close[i - 1] > final_upper[i - 1]:
            final_upper[i] = raw_upper[i]
        else:
            final_upper[i] = final_upper[i - 1]

        # Trend direction
        if trend[i - 1] == -1:
            trend[i] = np.int8(1) if close[i] > final_upper[i] else np.int8(-1)
        else:
            trend[i] = np.int8(-1) if close[i] < final_lower[i] else np.int8(1)

    return trend


def _compute_vwap(
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Session-cumulative VWAP, resets each day."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_vol = 0.0

    for i in range(n):
        if i == 0 or day_id[i] != day_id[i - 1]:
            cum_pv = close[i] * volume[i]
            cum_vol = float(volume[i])
        else:
            cum_pv += close[i] * volume[i]
            cum_vol += float(volume[i])

        vwap[i] = cum_pv / cum_vol if cum_vol > 0.0 else close[i]

    return vwap


def _compute_session_drawdown_pct(
    close: np.ndarray,
    time_min: np.ndarray,
    day_id: np.ndarray,
    session_start: int,
) -> np.ndarray:
    """
    Returns % below session high-water mark (0.0 = at/above session high).
    HWM is tracked from session_start onwards; resets each day.
    """
    n = len(close)
    dd_pct = np.zeros(n)
    session_high = np.zeros(n)

    for i in range(n):
        if i == 0 or day_id[i] != day_id[i - 1]:
            session_high[i] = close[i]
        else:
            if time_min[i] >= session_start:
                session_high[i] = max(session_high[i - 1], close[i])
            else:
                session_high[i] = close[i]

        if session_high[i] > 0.0:
            dd_pct[i] = max(0.0, (session_high[i] - close[i]) / session_high[i] * 100.0)

    return dd_pct


class Strategy(BaseStrategy):
    name = "drawdown_adaptive_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 120            # 24-bar ATR warmup (2 min) + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("supertrend_mult", 2.5, 1.5, 3.5),
            # % session drawdown beyond which CE entries are suppressed
            TunableParam("drawdown_pe_only_pct", 0.40, 0.20, 0.80),
            # % session drawdown beyond which ALL entries are suppressed
            TunableParam("drawdown_flat_pct", 0.70, 0.40, 1.20),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].fill_null(0).to_numpy().astype(int)
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy().astype(int)

        supertrend_mult = params.get("supertrend_mult", 2.5)
        drawdown_pe_only_pct = params.get("drawdown_pe_only_pct", 0.40)
        drawdown_flat_pct = params.get("drawdown_flat_pct", 0.70)

        # --- Indicators ---
        st_period = 24  # 2-minute ATR period (fixed — lookback, not tunable)
        trend = _compute_supertrend(high, low, close, st_period, supertrend_mult)
        vwap = _compute_vwap(close, volume, day_id)
        dd_pct = _compute_session_drawdown_pct(
            close, time_min, day_id, self.session_start_minutes
        )

        # Persistence: require 2 consecutive bars in same Supertrend direction
        # to filter single-bar 5s spikes
        trend_persist = np.zeros(n, dtype=bool)
        for i in range(1, n):
            trend_persist[i] = trend[i] == trend[i - 1]

        # --- Session filter ---
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        not_flat = dd_pct < drawdown_flat_pct

        # --- Entry signals ---
        # buy_ce: bullish Supertrend + above VWAP + NOT in distribution/liquidation regime
        buy_ce = (
            in_session
            & not_flat
            & (dd_pct < drawdown_pe_only_pct)  # CE suppressed in drawdown regime
            & (trend == 1)
            & trend_persist
            & (close > vwap)
        )

        # buy_pe: bearish Supertrend + below VWAP + not in full liquidation
        buy_pe = (
            in_session
            & not_flat
            & (trend == -1)
            & trend_persist
            & (close < vwap)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # stop=4 pts: ~8 NIFTY spot pts; if retraced this far, Supertrend thesis failed
            stop_points=np.full(n, 5),
            # target=7 pts: ~14 NIFTY spot pts; captures ~50% of expected 1-2 min trend move
            target_points=np.full(n, 8),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120s max hold
            max_trades_per_day=self.max_trades_per_day,
        )
