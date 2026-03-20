"""Open=High / Open=Low — Grok_3_of_10

Thesis: When the first bar's open equals its high (bearish) or low (bullish),
it signals strong institutional conviction at the open. Trade in that direction.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "open_high_low_v1"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 915     # 15:15
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", default=21.0, low=14.0, high=30.0),
            TunableParam("rel_vol_thresh", default=1.5, low=1.0, high=3.0),
            TunableParam("target_atr_mult", default=1.5, low=0.5, high=3.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_max = params.get("vix_max", 21.0)
        rv_thresh = params.get("rel_vol_thresh", 1.5)
        tgt_atr = params.get("target_atr_mult", 1.5)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr14 = _compute_atr(high, low, close, 14)

        # Relative volume
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        rel_vol = volume / avg_vol_20

        # Detect open=high and open=low on first bar of each day, propagate to first 15 min
        open_eq_low = np.zeros(n, dtype=np.bool_)
        open_eq_high = np.zeros(n, dtype=np.bool_)

        for d in np.unique(day_ids):
            day_idx = np.where(day_ids == d)[0]
            if len(day_idx) == 0:
                continue
            first = day_idx[0]
            tol = 0.0001 * open_[first] if open_[first] > 0 else 0.01
            is_open_low = abs(open_[first] - low[first]) < tol
            is_open_high = abs(open_[first] - high[first]) < tol

            # Apply to first 15 bars of the day
            first_15 = day_idx[(time_mins[day_idx] >= 555) & (time_mins[day_idx] <= 569)]
            if is_open_low:
                open_eq_low[first_15] = True
            if is_open_high:
                open_eq_high[first_15] = True

        # Filters
        time_ok = time_mins <= 569  # first 15 min only
        vix_ok = vix < vix_max
        vol_ok = rel_vol > rv_thresh

        # Open=Low → bullish (long), Open=High → bearish (short)
        long_entry = open_eq_low & time_ok & vix_ok & vol_ok
        short_entry = open_eq_high & time_ok & vix_ok & vol_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=0.005,  # stop at open price ≈ small %
            target_atr_mult=tgt_atr,
            time_stop_bars=60,
        )
