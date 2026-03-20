"""RSI Reversal — GPT_4_of_10

Thesis: Extreme RSI readings (< 30 oversold, > 70 overbought) signal temporary
exhaustion due to behavioral overreaction, often followed by a price revert.
Uses EMA200 as trend confirmation.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Compute RSI from close prices."""
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)  # neutral default
    if n < period + 1:
        return rsi

    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # Wilder's smoothing (EMA-like)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)

    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _compute_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA from numpy array."""
    n = len(arr)
    ema = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return np.nan_to_num(ema, nan=arr[0] if n > 0 else 0.0)

    ema[period - 1] = np.mean(arr[:period])
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
    # Backfill warmup with first valid value
    first_valid = ema[period - 1]
    ema[:period - 1] = first_valid
    return ema


class Strategy(BaseStrategy):
    name = "rsi_reversal_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_oversold", default=30.0, low=15.0, high=40.0),
            TunableParam("rsi_overbought", default=70.0, low=60.0, high=85.0),
            TunableParam("vix_max", default=25.0, low=15.0, high=35.0),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.005),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rsi_os = params.get("rsi_oversold", 30.0)
        rsi_ob = params.get("rsi_overbought", 70.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.002)
        target_pct = params.get("target_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)

        # ── Indicators ──
        rsi = _compute_rsi(close, 14)
        ema200 = _compute_ema(close, 200)
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)

        # ── Filters ──
        vix_ok = vix < vix_max
        vol_ok = volume > avg_vol_20

        # ── Entry: RSI extreme + EMA200 trend confirmation ──
        # Long: RSI < oversold AND close below EMA200 (deep oversold in downtrend → bounce)
        long_entry = (rsi < rsi_os) & (close < ema200) & vix_ok & vol_ok
        # Short: RSI > overbought AND close above EMA200 (overextended in uptrend → pullback)
        short_entry = (rsi > rsi_ob) & (close > ema200) & vix_ok & vol_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=15,
        )
