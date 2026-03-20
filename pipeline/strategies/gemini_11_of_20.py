"""Gap Momentum Continuation — gemini_11_of_20

Thesis: Stocks that gap significantly and then break the opening range high
continue in the gap direction, driven by momentum chasers.
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
    name = "gap_momentum_continuation"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 690     # 11:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_pct", default=2.0, low=1.0, high=4.0),
            TunableParam("vol_mult", default=1.5, low=1.0, high=3.0),
            TunableParam("target_pct", default=1.2, low=0.5, high=2.5),
            TunableParam("stop_pct", default=0.6, low=0.3, high=1.0),
            TunableParam("be_pct", default=0.6, low=0.3, high=1.0),
            TunableParam("vix_max", default=25.0, low=15.0, high=35.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_pct = params.get("gap_pct", 2.0)
        vol_mult = params.get("vol_mult", 1.5)
        target_pct = params.get("target_pct", 1.2)
        stop_pct = params.get("stop_pct", 0.6)
        be_pct = params.get("be_pct", 0.6)
        vix_max = params.get("vix_max", 25.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_id = df["day_id"].to_numpy()

        atr14 = _compute_atr(high, low, close, 14)

        # Compute previous day close per bar
        prev_day_close = np.full(n, np.nan, dtype=np.float64)
        prev_close_val = np.nan
        last_day = -1
        day_first_close = {}  # day_id -> first bar's previous close
        # First pass: find the last close of each day
        day_last_close = {}
        for i in range(n):
            d = day_id[i]
            day_last_close[d] = close[i]
        # Second pass: assign prev day close
        sorted_days = sorted(set(day_id))
        day_to_prev_close = {}
        for idx, d in enumerate(sorted_days):
            if idx == 0:
                day_to_prev_close[d] = np.nan
            else:
                day_to_prev_close[d] = day_last_close[sorted_days[idx - 1]]
        for i in range(n):
            prev_day_close[i] = day_to_prev_close.get(day_id[i], np.nan)
        prev_day_close = np.nan_to_num(prev_day_close, nan=close[0])

        # Gap percentage
        gap = (opn - prev_day_close) / np.where(prev_day_close > 1e-10, prev_day_close, 1e-10) * 100.0

        # 15-bar rolling high of close (range high)
        range_period = 15
        range_high = np.full(n, -np.inf, dtype=np.float64)
        range_low = np.full(n, np.inf, dtype=np.float64)
        for i in range(n):
            start = max(0, i - range_period)
            range_high[i] = np.max(high[start:i + 1])
            range_low[i] = np.min(low[start:i + 1])

        # Previous bar range high/low for breakout confirmation
        prev_range_high = np.roll(range_high, 1)
        prev_range_high[0] = range_high[0]
        prev_range_low = np.roll(range_low, 1)
        prev_range_low[0] = range_low[0]

        # Volume SMA(20) for relative volume
        vol_sma20 = np.full(n, 1.0, dtype=np.float64)
        for i in range(19, n):
            vol_sma20[i] = np.mean(volume[i - 19:i + 1])
        vol_sma20 = np.where(vol_sma20 > 1e-10, vol_sma20, 1.0)
        rel_vol = volume / vol_sma20

        # Entries
        vix_ok = vix < vix_max
        long_entry = (gap > gap_pct) & (close > prev_range_high) & (rel_vol > vol_mult) & vix_ok
        short_entry = (gap < -gap_pct) & (close < prev_range_low) & (rel_vol > vol_mult) & vix_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct / 100.0,
            stop_loss_pct=stop_pct / 100.0,
            breakeven_pct=be_pct / 100.0,
            time_stop_bars=180,
        )
