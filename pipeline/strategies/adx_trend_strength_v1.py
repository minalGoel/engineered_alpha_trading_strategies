"""ADX Trend Strength — 5-second NIFTY index options strategy.

Detects nascent institutional directional orders on NIFTY by monitoring ADX(60)
rising through 25 with +DI/-DI spread confirmation. Enters in the trend direction
while the institutional order is still partially unfilled (ADX still rising).
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int):
    """Compute Wilder's ADX, +DI, -DI.

    Returns (adx, plus_di, minus_di) as float64 arrays of length n.
    """
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0.0:
            plus_dm[i] = up
        elif down > up and down > 0.0:
            minus_dm[i] = down

    # Wilder smoothing: seed with sum of first `period` values
    atr = np.zeros(n)
    plus_dm_s = np.zeros(n)
    minus_dm_s = np.zeros(n)

    if n > period:
        atr[period] = np.sum(tr[1 : period + 1])
        plus_dm_s[period] = np.sum(plus_dm[1 : period + 1])
        minus_dm_s[period] = np.sum(minus_dm[1 : period + 1])

        for i in range(period + 1, n):
            atr[i] = atr[i - 1] - atr[i - 1] / period + tr[i]
            plus_dm_s[i] = plus_dm_s[i - 1] - plus_dm_s[i - 1] / period + plus_dm[i]
            minus_dm_s[i] = minus_dm_s[i - 1] - minus_dm_s[i - 1] / period + minus_dm[i]

    plus_di = np.where(atr > 0, 100.0 * plus_dm_s / atr, 0.0)
    minus_di = np.where(atr > 0, 100.0 * minus_dm_s / atr, 0.0)

    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.where(di_sum > 0, 100.0 * di_diff / di_sum, 0.0)

    adx = np.zeros(n)
    start = 2 * period
    if n > start:
        adx[start] = np.mean(dx[period : start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di


def _compute_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Compute session-cumulative VWAP, resetting each day."""
    n = len(close)
    vwap = np.zeros(n)
    cum_vol = 0.0
    cum_tpv = 0.0
    prev_day = -1

    for i in range(n):
        if day_id[i] != prev_day:
            cum_vol = 0.0
            cum_tpv = 0.0
            prev_day = day_id[i]
        tp = (high[i] + low[i] + close[i]) / 3.0
        cum_vol += volume[i]
        cum_tpv += tp * volume[i]
        vwap[i] = cum_tpv / cum_vol if cum_vol > 0 else close[i]

    return vwap


class Strategy(BaseStrategy):
    name = "adx_trend_strength_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — ADX(60) needs ~120-bar warmup
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 240            # 20 min (2× ADX period of 60 = safe warmup)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_threshold", 25.0, 18.0, 35.0),
            TunableParam("di_spread_min", 8.0, 4.0, 20.0),
            TunableParam("stop_pts", 5.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        adx_threshold = params.get("adx_threshold", 25.0)
        di_spread_min = params.get("di_spread_min", 8.0)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ADX(60) = 5-minute trend context
        adx60, plus_di60, minus_di60 = _compute_adx(high, low, close, 60)

        # VWAP from session open (cumulative, no scaling)
        vwap = _compute_vwap(high, low, close, volume, day_id)

        # Session filter
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ADX above threshold (trend established)
        adx_above = adx60 > adx_threshold

        # ADX still rising (institutional order still active / partially unfilled)
        adx_rising = np.zeros(n, dtype=bool)
        adx_rising[1:] = adx60[1:] > adx60[:-1]

        # Directional bias: DI spread confirms direction
        bull_di = (plus_di60 - minus_di60) > di_spread_min
        bear_di = (minus_di60 - plus_di60) > di_spread_min

        # VWAP alignment: price above/below session VWAP
        above_vwap = close > vwap
        below_vwap = close < vwap

        # Entry: all conditions met simultaneously
        buy_ce = in_session & adx_above & adx_rising & bull_di & above_vwap
        buy_pe = in_session & adx_above & adx_rising & bear_di & below_vwap

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
