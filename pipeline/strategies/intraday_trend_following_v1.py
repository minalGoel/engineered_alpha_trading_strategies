"""Intraday Trend Following — EMA Pullback on NIFTY (5-second bars).

Original: EMA(20/50) pullback trend-following on NIFTY50 stocks, 1-min bars, 20-90 min hold.
Converted: EMA(24/120) pullback on NIFTY index, 5s bars, 15-90 second hold.

Mechanism: When institutional TWAP/VWAP algorithms sustain directional orders over 10-20 minutes,
they create persistent EMA(24) > EMA(120) alignment on NIFTY. Brief pullbacks to EMA(24) represent
short-term sellers exhausting against resting institutional bids. The bar that closes back above
EMA(24) signals institutional absorption has re-engaged; we ride the next 8-15 spot points.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with smoothing factor 2/(period+1)."""
    result = np.empty(len(arr))
    k = 2.0 / (period + 1)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] * k + result[i - 1] * (1.0 - k)
    return result


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI. Returns 50.0 for bars before warmup is complete."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = np.zeros(n)
    avg_l = np.zeros(n)
    avg_g[period] = np.mean(gain[1 : period + 1])
    avg_l[period] = np.mean(loss[1 : period + 1])
    for i in range(period + 1, n):
        avg_g[i] = (avg_g[i - 1] * (period - 1) + gain[i]) / period
        avg_l[i] = (avg_l[i - 1] * (period - 1) + loss[i]) / period
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi[period:] = 100.0 - (100.0 / (1.0 + rs[period:]))
    return rsi


def _session_vwap(spot_df: pl.DataFrame) -> np.ndarray:
    """Intraday VWAP reset each session day."""
    vwap_series = (
        spot_df.with_columns(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("tp")
        )
        .with_columns(
            (pl.col("tp") * pl.col("volume").cast(pl.Float64)).alias("tp_vol")
        )
        .with_columns(
            pl.col("tp_vol").cum_sum().over("day_id").alias("cum_tp_vol"),
            pl.col("volume").cast(pl.Float64).cum_sum().over("day_id").alias("cum_vol"),
        )
        .with_columns(
            (pl.col("cum_tp_vol") / pl.col("cum_vol").clip(lower_bound=1.0)).alias("vwap")
        )
        .select("vwap")
        ["vwap"]
        .fill_null(strategy="forward")
    )
    return vwap_series.to_numpy()


class Strategy(BaseStrategy):
    name = "intraday_trend_following_v1"
    underlying = "NIFTY"
    session_start_minutes = 575   # 09:35 IST — allow 20 min for trend to establish
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10-min warmup for EMA(120)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("trend_strength_pct", 0.001, 0.0003, 0.003),
            TunableParam("pullback_tolerance", 0.001, 0.0003, 0.002),
            TunableParam("rsi_overbought", 65.0, 55.0, 75.0),
            TunableParam("rsi_oversold", 35.0, 25.0, 45.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high  = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low   = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # Parameters
        trend_strength_pct = params.get("trend_strength_pct", 0.001)
        pullback_tol       = params.get("pullback_tolerance", 0.001)
        rsi_ob             = params.get("rsi_overbought", 65.0)
        rsi_os             = params.get("rsi_oversold", 35.0)
        stop_pts           = params.get("stop_pts", 3.0)
        target_pts         = params.get("target_pts", 5.0)

        # Indicators
        ema24  = _ema(close, 24)   # 2-min: pullback level and entry trigger
        ema120 = _ema(close, 120)  # 10-min: trend direction context
        rsi24  = _rsi(close, 24)   # 2-min RSI: avoid overbought/oversold entries
        vwap   = _session_vwap(spot_df)

        # Trend strength: relative separation of fast vs slow EMA
        ema120_safe = np.where(ema120 > 0, ema120, 1.0)
        trend_rel = (ema24 - ema120) / ema120_safe

        # Previous-bar arrays for pullback detection
        prev_low  = np.empty(n)
        prev_high = np.empty(n)
        prev_low[0]  = low[0]
        prev_high[0] = high[0]
        prev_low[1:]  = low[:-1]
        prev_high[1:] = high[:-1]

        # Session filter
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Bullish: EMA pullback bounce (buy CE) ──
        bull_trend    = trend_rel > trend_strength_pct                    # EMA24 meaningfully above EMA120
        bull_pullback = prev_low <= ema24 * (1.0 + pullback_tol)          # prev bar dipped to EMA24
        bull_bounce   = close > ema24                                      # current bar closed above EMA24
        bull_vwap     = close > vwap                                       # above session anchor
        bull_rsi      = rsi24 < rsi_ob                                     # not overbought

        buy_ce = in_session & bull_trend & bull_pullback & bull_bounce & bull_vwap & bull_rsi

        # ── Bearish: EMA rally rejection (buy PE) ──
        bear_trend    = trend_rel < -trend_strength_pct                    # EMA24 meaningfully below EMA120
        bear_pullback = prev_high >= ema24 * (1.0 - pullback_tol)          # prev bar rallied to EMA24
        bear_bounce   = close < ema24                                       # current bar closed below EMA24
        bear_vwap     = close < vwap                                        # below session anchor
        bear_rsi      = rsi24 > rsi_os                                      # not oversold

        buy_pe = in_session & bear_trend & bear_pullback & bear_bounce & bear_vwap & bear_rsi

        # Mutual exclusion (edge case: both fire on same bar)
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
            time_stop_bars=18,                   # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
