"""Previous-Day Reversal — Opus_40

Thesis: After a strong previous-day move (>1%), the next day tends
to reverse. Enter on first confirming bar (green for long after
down day, red for short after up day) with RSI and VIX filters.
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


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[1:period + 1])
    avg_loss = np.mean(loss[1:period + 1])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


class Strategy(BaseStrategy):
    name = "opus_monday_reversal_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 660     # 11:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("prev_day_ret_thresh", default=0.01, low=0.005, high=0.02),
            TunableParam("rsi_long_thresh", default=40.0, low=25.0, high=50.0),
            TunableParam("rsi_short_thresh", default=60.0, low=50.0, high=75.0),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.0035, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        prev_day_thresh = params.get("prev_day_ret_thresh", 0.01)
        rsi_long = params.get("rsi_long_thresh", 40.0)
        rsi_short = params.get("rsi_short_thresh", 60.0)
        vix_max = params.get("vix_max", 22.0)
        stp_pct = params.get("stop_loss_pct", 0.0035)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        atr = _compute_atr(high, low, close, 20)
        rsi = _compute_rsi(close, 14)

        # ── Compute previous day return ──
        prev_day_return = np.zeros(n, dtype=np.float64)
        prev_day = -1
        day_open = 0.0
        day_close = 0.0
        last_day_ret = 0.0

        # First pass: collect day open/close
        day_opens = {}
        day_closes = {}
        for i in range(n):
            d = day_id[i]
            if d not in day_opens:
                day_opens[d] = open_[i]
            day_closes[d] = close[i]

        unique_days = sorted(day_opens.keys())
        day_ret_map = {}
        for idx, d in enumerate(unique_days):
            if idx == 0:
                day_ret_map[d] = 0.0
            else:
                prev_d = unique_days[idx - 1]
                prev_o = day_opens[prev_d]
                prev_c = day_closes[prev_d]
                if prev_o > 0:
                    day_ret_map[d] = (prev_c - prev_o) / prev_o
                else:
                    day_ret_map[d] = 0.0

        for i in range(n):
            prev_day_return[i] = day_ret_map.get(day_id[i], 0.0)

        # Target pct: 0.5 * |prev_day_return|, clamped
        target_pct_arr = np.clip(0.5 * np.abs(prev_day_return), 0.002, 0.01)
        # Use median as scalar target
        active_mask = np.abs(prev_day_return) > prev_day_thresh
        if np.any(active_mask):
            tgt_pct = float(np.median(target_pct_arr[active_mask]))
        else:
            tgt_pct = 0.005

        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)
        vix_ok = vix < vix_max
        green_bar = close > open_
        red_bar = close < open_

        # Long: prev day was down big, first green bar with RSI < 40
        long_entry = ((prev_day_return < -prev_day_thresh) & green_bar &
                      (rsi < rsi_long) & vix_ok & time_ok)

        # Short: prev day was up big, first red bar with RSI > 60
        short_entry = ((prev_day_return > prev_day_thresh) & red_bar &
                       (rsi > rsi_short) & vix_ok & time_ok)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stp_pct,
            time_stop_bars=90,
        )
