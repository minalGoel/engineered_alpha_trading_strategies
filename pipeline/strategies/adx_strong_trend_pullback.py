"""ADX Strong Trend Pullback — 5-second NIFTY index options strategy.

When NIFTY is in a confirmed 3-minute micro-trend (ADX(36) > 25), waits for a shallow
pullback where the 5-second bar's low touches the 2-minute EMA but the close recovers
above it (buy CE) or the high touches the EMA but the close stays below it (buy PE).
This "touch-and-reclaim" confirms institutional absorption of retail profit-taking
and targets the next 10-20 spot point continuation within 30-90 seconds.

Differentiated from adx_trend_strength_v1 (ADX(60) + VWAP alignment, trend continuation):
this strategy specifically waits for the EMA pullback-and-reclaim entry trigger.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_adx(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
):
    """Wilder's ADX, +DI, -DI. Returns (adx, plus_di, minus_di), all float64, length n."""
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
        dn = low[i - 1] - low[i]
        if up > dn and up > 0.0:
            plus_dm[i] = up
        elif dn > up and dn > 0.0:
            minus_dm[i] = dn

    # Wilder smoothing: seed with first period sum, then rolling
    atr_s = np.zeros(n)
    pdm_s = np.zeros(n)
    mdm_s = np.zeros(n)

    if n > period:
        atr_s[period] = np.sum(tr[1: period + 1])
        pdm_s[period] = np.sum(plus_dm[1: period + 1])
        mdm_s[period] = np.sum(minus_dm[1: period + 1])

        for i in range(period + 1, n):
            atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
            pdm_s[i] = pdm_s[i - 1] - pdm_s[i - 1] / period + plus_dm[i]
            mdm_s[i] = mdm_s[i - 1] - mdm_s[i - 1] / period + minus_dm[i]

    plus_di = np.where(atr_s > 0, 100.0 * pdm_s / atr_s, 0.0)
    minus_di = np.where(atr_s > 0, 100.0 * mdm_s / atr_s, 0.0)

    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    dx = np.where(di_sum > 0, 100.0 * di_diff / di_sum, 0.0)

    adx = np.zeros(n)
    start = 2 * period
    if n > start:
        adx[start] = np.mean(dx[period: start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di


def _compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Standard EMA with Wilder-style seed (SMA of first period bars)."""
    n = len(close)
    ema = np.zeros(n)
    if n < period:
        return ema
    alpha = 2.0 / (period + 1)
    ema[period - 1] = np.mean(close[:period])
    for i in range(period, n):
        ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1]
    return ema


class Strategy(BaseStrategy):
    name = "adx_strong_trend_pullback"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — 6-min warmup for ADX(36) = 72 bars
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 144            # 12 min = 2× ADX period of 36 × 2 = safe warmup

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_threshold", 25.0, 18.0, 35.0),
            TunableParam("di_min_spread", 5.0, 2.0, 15.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        adx_threshold = params.get("adx_threshold", 25.0)
        di_min_spread = params.get("di_min_spread", 5.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # ADX(36) = 3-min trend strength — TRADE TRIGGER, not background context
        adx36, plus_di36, minus_di36 = _compute_adx(high, low, close, 36)

        # EMA(24) = 2-min dynamic support/resistance for pullback detection
        ema24 = _compute_ema(close, 24)

        # Session filter
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # Trend strength: ADX above threshold confirms institutional micro-trend
        trend_strong = adx36 > adx_threshold

        # Directional bias with minimum DI spread for confidence
        bull_di = (plus_di36 - minus_di36) > di_min_spread
        bear_di = (minus_di36 - plus_di36) > di_min_spread

        # EMA(24) valid (past warmup period)
        ema_valid = ema24 > 0.0

        # Pullback touch-and-reclaim: bar's low touched EMA but close recovered above
        # This is the core signal: selling absorbed at EMA, institutional buyers stepped in
        ema_bull_touch = (low <= ema24) & (close > ema24)

        # Mirror for bearish: high touched EMA resistance but close rejected below
        ema_bear_touch = (high >= ema24) & (close < ema24)

        # Final entry signals
        buy_ce = in_session & trend_strong & bull_di & ema_valid & ema_bull_touch
        buy_pe = in_session & trend_strong & bear_di & ema_valid & ema_bear_touch

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,             # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
