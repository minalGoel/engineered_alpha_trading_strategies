"""Earnings Surprise Momentum v1 — cursor_opus46max_121

Thesis: Post-Earnings Announcement Drift (PEAD).  Stocks gapping >2% at open
on result days drift further in gap direction for 60-90 minutes.  Proxy
earnings gap using opening bar return and volume surge.
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
    name = "cursor_opus46max_121"
    is_long_only = False
    session_start = 575   # 09:35
    session_end = 720     # 12:00 (drift fades by midday)
    max_trades_per_day = 5
    assumptions = ["Earnings surprise proxied via gap size and volume surge"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_pct_thresh", default=2.0, low=1.0, high=4.0),
            TunableParam("vol_surge_thresh", default=2.0, low=1.5, high=4.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.01),
            TunableParam("target_pct", default=0.008, low=0.004, high=0.015),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_thresh = params.get("gap_pct_thresh", 2.0)
        vol_surge = params.get("vol_surge_thresh", 2.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        target_pct = params.get("target_pct", 0.008)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Gap percentage: open vs previous day close
        prev_close = np.zeros(n, dtype=np.float64)
        day_open = np.zeros(n, dtype=np.float64)
        prev_day_close = close[0]
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                prev_close[i] = prev_day_close
                day_open[i] = opn[i]
                if i > 0:
                    prev_day_close = close[i-1]
            else:
                prev_close[i] = prev_close[i-1]
                day_open[i] = day_open[i-1]

        safe_prev = np.where(prev_close > 1e-10, prev_close, 1e-10)
        gap_pct = (day_open - prev_close) / safe_prev * 100.0

        # Volume surge relative to 20-bar average
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # Intraday continuation: close vs day open
        safe_day_open = np.where(day_open > 1e-10, day_open, 1e-10)
        continuation = (close - day_open) / safe_day_open * 10000.0

        # Entry only in first 45 min (555-600 = 09:15-10:00, using 575-600)
        early_session = (time_mins >= 575) & (time_mins <= 600)

        # Opening range: first 5 bars of day
        or_high = np.zeros(n, dtype=np.float64)
        or_low = np.full(n, 1e10, dtype=np.float64)
        bars_in_day = 0
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                bars_in_day = 0
            bars_in_day += 1
            if bars_in_day <= 5:
                if i == 0 or day_id[i] != day_id[max(0, i-1)]:
                    or_high[i] = high[i]
                    or_low[i] = low[i]
                else:
                    or_high[i] = max(or_high[i-1], high[i])
                    or_low[i] = min(or_low[i-1], low[i])
            else:
                or_high[i] = or_high[i-1]
                or_low[i] = or_low[i-1]

        # Long: positive gap > thresh, volume surge, continuation positive, breakout above OR high
        long_entry = ((gap_pct > gap_thresh) & (vol_ratio > vol_surge) &
                      (continuation > 0) & (close > or_high) & early_session)

        # Short: negative gap, volume surge, continuation negative
        short_entry = ((gap_pct < -gap_thresh) & (vol_ratio > vol_surge) &
                       (continuation < 0) & (close < or_low) & early_session)

        # Signal exit: volume surge drops below 1.0
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if vol_ratio[i] < 1.0 and vol_ratio[i-1] >= 1.0:
                sig_exit_long[i] = True
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=0.002,
            trailing_activate_pct=0.003,
            time_stop_bars=90,
        )
