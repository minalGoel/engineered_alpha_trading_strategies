"""ADX Strong Trend Pullback — gemini_12_of_20

Thesis: In strong trending markets (high ADX), pullbacks to the EMA(20)
offer high-probability continuation entries.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_ema(data, period):
    n = len(data)
    ema = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return np.nan_to_num(ema, nan=data[0] if n > 0 else 0.0)
    ema[period - 1] = np.mean(data[:period])
    mult = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = data[i] * mult + ema[i - 1] * (1.0 - mult)
    # Fill early bars
    for i in range(period - 1):
        ema[i] = ema[period - 1]
    return ema


class Strategy(BaseStrategy):
    name = "adx_strong_trend_pullback"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_min", default=25.0, low=20.0, high=40.0),
            TunableParam("adx_exit", default=20.0, low=10.0, high=25.0),
            TunableParam("target_atr_mult", default=2.0, low=1.0, high=4.0),
            TunableParam("stop_atr_mult", default=1.5, low=0.5, high=3.0),
            TunableParam("trailing_atr_mult", default=1.0, low=0.5, high=2.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_min = params.get("adx_min", 25.0)
        adx_exit_thresh = params.get("adx_exit", 20.0)
        target_atr = params.get("target_atr_mult", 2.0)
        stop_atr = params.get("stop_atr_mult", 1.5)
        trail_atr = params.get("trailing_atr_mult", 1.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        # ATR(14)
        atr14 = _compute_atr(high, low, close, 14)

        # EMA(20)
        ema20 = _compute_ema(close, 20)

        # ADX(14) with +DI/-DI
        adx_period = 14
        plus_dm = np.zeros(n, dtype=np.float64)
        minus_dm = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            up = high[i] - high[i - 1]
            dn = low[i - 1] - low[i]
            plus_dm[i] = up if (up > dn and up > 0) else 0.0
            minus_dm[i] = dn if (dn > up and dn > 0) else 0.0

        smooth_plus = np.zeros(n, dtype=np.float64)
        smooth_minus = np.zeros(n, dtype=np.float64)
        if n >= adx_period + 1:
            smooth_plus[adx_period] = np.mean(plus_dm[1:adx_period + 1])
            smooth_minus[adx_period] = np.mean(minus_dm[1:adx_period + 1])
            for i in range(adx_period + 1, n):
                smooth_plus[i] = (smooth_plus[i - 1] * (adx_period - 1) + plus_dm[i]) / adx_period
                smooth_minus[i] = (smooth_minus[i - 1] * (adx_period - 1) + minus_dm[i]) / adx_period

        safe_atr = np.where(atr14 > 1e-10, atr14, 1e-10)
        plus_di = 100.0 * smooth_plus / safe_atr
        minus_di = 100.0 * smooth_minus / safe_atr
        dx = np.where((plus_di + minus_di) > 1e-10,
                       100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)

        adx = np.zeros(n, dtype=np.float64)
        adx_start = 2 * adx_period
        if n > adx_start:
            adx[adx_start] = np.mean(dx[adx_period:adx_start + 1])
            for i in range(adx_start + 1, n):
                adx[i] = (adx[i - 1] * (adx_period - 1) + dx[i]) / adx_period

        # Long: ADX > thresh, +DI > -DI, low touches EMA(20) but close above
        long_entry = (
            (adx > adx_min)
            & (plus_di > minus_di)
            & (low <= ema20)
            & (close > ema20)
        )

        # Short: ADX > thresh, -DI > +DI, high touches EMA(20) but close below
        short_entry = (
            (adx > adx_min)
            & (minus_di > plus_di)
            & (high >= ema20)
            & (close < ema20)
        )

        # Signal exit: ADX drops below exit threshold
        signal_exit = adx < adx_exit_thresh

        # Trailing stop in ATR terms: convert to pct using current close
        # Use a rough median close for trailing pct
        median_close = np.median(close) if n > 0 else 1.0
        trailing_pct = (trail_atr * np.median(atr14[atr14 > 0]) / median_close) if np.any(atr14 > 0) else 0.005

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit,
            signal_exit_short=signal_exit,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_atr_mult=target_atr,
            stop_loss_atr_mult=stop_atr,
            trailing_stop_pct=trailing_pct,
            time_stop_bars=200,
        )
