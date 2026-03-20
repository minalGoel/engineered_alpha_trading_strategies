# AUDIT FIX: Added session window filter to long_entry and short_entry
"""Volume Spike Exhaustion Fade — gemini_19_of_20

Thesis: Extremely high volume bars with small bodies (doji-like) signal
exhaustion. Fade the direction of the preceding move expecting a reversal.
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


class Strategy(BaseStrategy):
    name = "volume_spike_exhaustion_fade"
    is_long_only = False
    session_start = 630   # 10:30
    session_end = 870     # 14:30
    max_trades_per_day = 20

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_spike_mult", default=5.0, low=3.0, high=8.0),
            TunableParam("body_ratio_max", default=0.3, low=0.1, high=0.5),
            TunableParam("target_pct", default=0.25, low=0.1, high=0.5),
            TunableParam("stop_pct", default=0.15, low=0.05, high=0.3),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_spike_mult = params.get("vol_spike_mult", 5.0)
        body_ratio_max = params.get("body_ratio_max", 0.3)
        target_pct = params.get("target_pct", 0.25)
        stop_pct = params.get("stop_pct", 0.15)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)

        atr14 = _compute_atr(high, low, close, 14)

        # Volume SMA(50)
        vol_sma50 = np.full(n, 1.0, dtype=np.float64)
        for i in range(49, n):
            vol_sma50[i] = np.mean(volume[i - 49:i + 1])
        # Fill early bars
        if n >= 50:
            for i in range(49):
                vol_sma50[i] = vol_sma50[49]
        vol_sma50 = np.where(vol_sma50 > 1e-10, vol_sma50, 1.0)

        # Volume spike
        vol_spike = volume > (vol_spike_mult * vol_sma50)

        # Body/range ratio (exhaustion candle = small body relative to range)
        body = np.abs(close - opn)
        bar_range = high - low
        safe_range = np.where(bar_range > 1e-10, bar_range, 1e-10)
        body_ratio = body / safe_range

        exhaustion = vol_spike & (body_ratio < body_ratio_max)

        # Previous close for direction
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]

        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Long: down exhaustion (close < prev close) — fade the sell
        long_entry = exhaustion & (close < prev_close) & in_session

        # Short: up exhaustion (close > prev close) — fade the buy
        short_entry = exhaustion & (close > prev_close) & in_session

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct / 100.0,
            stop_loss_pct=stop_pct / 100.0,
            time_stop_bars=10,
        )
