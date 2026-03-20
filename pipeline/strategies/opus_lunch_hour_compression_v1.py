"""Lunch Hour Compression Breakout — Opus_36

Thesis: The lunch hour (12:00-13:30) is typically a low-volatility
consolidation period. A breakout from the lunch range at 13:30,
aligned with the morning trend, tends to continue into the close.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "opus_lunch_hour_compression_v1"
    is_long_only = False
    session_start = 720   # 12:00
    session_end = 900     # 15:00
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("target_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        target_pct = params.get("target_pct", 0.003)
        stop_pct = params.get("stop_loss_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        day_id = df["day_id"].to_numpy().astype(np.int64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Compute lunch high/low per day (720-810 minutes) ──
        day_lunch_high = {}
        day_lunch_low = {}
        for i in range(n):
            d = day_id[i]
            if 720 <= time_mins[i] < 810:
                if d not in day_lunch_high:
                    day_lunch_high[d] = high[i]
                    day_lunch_low[d] = low[i]
                else:
                    day_lunch_high[d] = max(day_lunch_high[d], high[i])
                    day_lunch_low[d] = min(day_lunch_low[d], low[i])

        lunch_high = np.zeros(n, dtype=np.float64)
        lunch_low = np.full(n, 1e10, dtype=np.float64)
        for i in range(n):
            d = day_id[i]
            if d in day_lunch_high:
                lunch_high[i] = day_lunch_high[d]
                lunch_low[i] = day_lunch_low[d]

        # ── Morning direction: compare close at 720 vs open at 555 ──
        day_morning_open = {}
        day_morning_close = {}
        for i in range(n):
            d = day_id[i]
            if time_mins[i] <= 560 and d not in day_morning_open:
                day_morning_open[d] = open_[i]
            if 715 <= time_mins[i] <= 725:
                day_morning_close[d] = close[i]

        morning_direction = np.zeros(n, dtype=np.float64)
        for i in range(n):
            d = day_id[i]
            if d in day_morning_open and d in day_morning_close:
                if day_morning_open[d] > 1e-10:
                    morning_direction[i] = (day_morning_close[d] - day_morning_open[d]) / day_morning_open[d]

        # ── Average volume ──
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Filters ──
        time_ok = (time_mins >= 810) & (time_mins <= self.session_end)
        has_lunch_range = lunch_high > 0

        # ── Entries ──
        long_entry = (
            (close > lunch_high)
            & (time_mins >= 810)
            & (morning_direction > 0)
            & (volume > avg_vol)
            & has_lunch_range
            & time_ok
        )
        short_entry = (
            (close < lunch_low)
            & (lunch_low < 1e9)
            & (time_mins >= 810)
            & (morning_direction < 0)
            & (volume > avg_vol)
            & has_lunch_range
            & time_ok
        )

        # ── Signal exit: close back inside lunch range ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if close[i] < lunch_high[i] and close[i - 1] >= lunch_high[i - 1] and lunch_high[i] > 0:
                sig_exit_long[i] = True
            if close[i] > lunch_low[i] and close[i - 1] <= lunch_low[i - 1] and lunch_low[i] < 1e9:
                sig_exit_short[i] = True

        # Time stop: by 900 (15:00) — compute bars from entry session start
        # Use 90 bars as approximate (810 to 900 = 90 min)
        time_stop = 90

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=time_stop,
        )
