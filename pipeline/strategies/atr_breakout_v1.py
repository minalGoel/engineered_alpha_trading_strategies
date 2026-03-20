"""ATR Breakout — GPT_6_of_10

Thesis: When the current bar's range exceeds 1.5x ATR(14), a volatility
breakout is occurring. The direction is determined by whether the close
is in the upper or lower half of the bar.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Compute ATR using Wilder's method."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)

    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))

    if n < period:
        return atr

    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return atr


class Strategy(BaseStrategy):
    name = "atr_breakout_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("atr_mult", default=1.5, low=1.0, high=3.0),
            TunableParam("vol_mult", default=1.5, low=1.0, high=3.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.01),
            TunableParam("target_pct", default=0.01, low=0.005, high=0.02),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        atr_mult = params.get("atr_mult", 1.5)
        vol_mult = params.get("vol_mult", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.005)
        target_pct = params.get("target_pct", 0.01)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        # ── ATR(14) ──
        atr14 = _compute_atr(high, low, close, 14)

        # ── Bar range ──
        bar_range = high - low

        # ── Volume filter ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol_20

        # ── Breakout: range > atr_mult * ATR ──
        # Safe: only signal where ATR > 0 (post warmup)
        atr_safe = np.where(atr14 > 0, atr14, 1e10)
        breakout = bar_range > atr_mult * atr_safe

        # Direction: close in upper half → bullish, lower half → bearish
        mid = (high + low) / 2.0
        long_entry = breakout & (close > mid) & vol_ok
        short_entry = breakout & (close <= mid) & vol_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=10,
        )
