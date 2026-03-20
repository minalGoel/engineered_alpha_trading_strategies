# AUDIT FIX: long_entry was not filtered by session window (session_start=560, session_end=915).
# Added time_ok filter using df["time_minutes"] to restrict entries to session hours.
"""EMA200 Dip Reversal — Grok_6_of_10

Thesis: Bounces from the 200-period EMA after multi-day support formation.
Long-only strategy buying dips near EMA200 with volume confirmation.

Assumptions:
- "multi_day_support" interpreted as: price has been near EMA200 on at least
  2 of the last 3 days (close within 0.5% of EMA200 on those days)
- 5min aggregation simplified to using 200-bar EMA on 1min (≈200 minutes)
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_ema(arr, period):
    n = len(arr)
    ema = np.full(n, np.nan, dtype=np.float64)
    if n < period:
        return np.nan_to_num(ema, nan=arr[0] if n > 0 else 0.0)
    ema[period-1] = np.mean(arr[:period])
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
    ema[:period-1] = ema[period-1]
    return ema


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
    name = "ema200_dip_reversal_v1"
    is_long_only = True
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 7
    assumptions = ["multi_day_support = close within 0.5% of EMA200 on 2+ of last 3 days",
                   "uses 200-bar EMA on 1min instead of 5min aggregation"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", default=19.0, low=12.0, high=28.0),
            TunableParam("rel_vol_thresh", default=1.6, low=1.0, high=3.0),
            TunableParam("dip_atr_mult", default=0.5, low=0.2, high=1.5),
            TunableParam("target_atr_mult", default=1.0, low=0.5, high=2.5),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_max = params.get("vix_max", 19.0)
        rv_thresh = params.get("rel_vol_thresh", 1.6)
        dip_mult = params.get("dip_atr_mult", 0.5)
        tgt_atr = params.get("target_atr_mult", 1.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_ids = df["day_id"].to_numpy()

        ema200 = _compute_ema(close, 200)
        atr20 = _compute_atr(high, low, close, 20)

        # Relative volume
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        rel_vol = volume / avg_vol_20

        # Multi-day support: check if close was within 0.5% of EMA200 on prev days
        # Simplified: for each bar, check if any bar in the previous 2 day-ids had
        # close within 0.5% of ema200
        multi_day = np.zeros(n, dtype=np.bool_)
        unique_days = np.unique(day_ids)
        day_near_ema = {}  # day_id → bool
        for d in unique_days:
            day_idx = np.where(day_ids == d)[0]
            near = np.any(np.abs(close[day_idx] - ema200[day_idx]) < 0.005 * close[day_idx])
            day_near_ema[d] = near

        for i, d in enumerate(unique_days):
            day_idx = np.where(day_ids == d)[0]
            count = 0
            for prev_d in unique_days[max(0, i-3):i]:
                if day_near_ema.get(prev_d, False):
                    count += 1
            if count >= 2:
                multi_day[day_idx] = True

        # Filters
        vix_ok = vix < vix_max
        vol_ok = rel_vol > rv_thresh

        # Session window filter
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Dip near EMA200: close > ema200 - dip_mult * ATR
        safe_atr = np.where(atr20 > 0, atr20, 1e10)
        near_ema = close > (ema200 - dip_mult * safe_atr)
        above_ema = close > ema200  # needs to be near but above or at

        long_entry = near_ema & multi_day & vix_ok & vol_ok & time_ok
        short_entry = np.zeros(n, dtype=np.bool_)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr20,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=0.005,
            target_atr_mult=tgt_atr,
            time_stop_bars=30,
        )
