"""Mean Reversion Hull MA — gemini_15_of_20

Thesis: The Hull Moving Average (HMA) responds quickly to price changes.
When price deviates >1% from a rising/falling HMA, fade the deviation
expecting reversion to the HMA.
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


def _compute_wma(data, period):
    """Weighted Moving Average."""
    n = len(data)
    wma = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return np.nan_to_num(wma, nan=data[0] if n > 0 else 0.0)
    weights = np.arange(1, period + 1, dtype=np.float64)
    w_sum = weights.sum()
    for i in range(period - 1, n):
        wma[i] = np.dot(data[i - period + 1:i + 1], weights) / w_sum
    # Fill early bars
    first_valid = wma[period - 1]
    for i in range(period - 1):
        wma[i] = first_valid
    return wma


class Strategy(BaseStrategy):
    name = "mean_reversion_hull_ma"
    is_long_only = False
    session_start = 630   # 10:30
    session_end = 870     # 14:30
    max_trades_per_day = 15

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("deviation_pct", default=1.0, low=0.3, high=2.0),
            TunableParam("stop_pct", default=0.4, low=0.2, high=0.8),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        dev_pct = params.get("deviation_pct", 1.0)
        stop_pct = params.get("stop_pct", 0.4)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)

        atr14 = _compute_atr(high, low, close, 14)

        # HMA(50) = WMA(2*WMA(close,25) - WMA(close,50), int(sqrt(50)))
        wma25 = _compute_wma(close, 25)
        wma50 = _compute_wma(close, 50)
        hull_input = 2.0 * wma25 - wma50
        hull_period = int(np.sqrt(50))  # 7
        hma = _compute_wma(hull_input, hull_period)
        hma = np.nan_to_num(hma, nan=close[0] if n > 0 else 0.0)

        # HMA direction (rising/falling)
        hma_rising = np.zeros(n, dtype=np.bool_)
        hma_falling = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            hma_rising[i] = hma[i] > hma[i - 1]
            hma_falling[i] = hma[i] < hma[i - 1]

        # Deviation from HMA
        safe_hma = np.where(np.abs(hma) > 1e-10, hma, 1e-10)
        pct_from_hma = (close - hma) / safe_hma * 100.0

        # Long: close < HMA - dev%, HMA rising (expect reversion up)
        long_entry = (pct_from_hma < -dev_pct) & hma_rising

        # Short: close > HMA + dev%, HMA falling (expect reversion down)
        short_entry = (pct_from_hma > dev_pct) & hma_falling

        # Signal exit: touch HMA
        signal_exit_long = close >= hma
        signal_exit_short = close <= hma

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=hma.copy(),
            use_target_indicator=True,
            stop_loss_pct=stop_pct / 100.0,
            time_stop_bars=30,
        )
