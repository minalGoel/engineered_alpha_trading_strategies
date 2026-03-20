# AUDIT FIX: Added session time filter — entries were firing outside session_start=570/session_end=870
"""ORB Failure Trade v1 — cursor_opus46max_094

Thesis: Failed ORB breakouts (price breaks then reverses within 15 bars)
are high-probability reversal signals. Trapped breakout traders' liquidation
creates reliable flow for the contrarian entry back inside the range.
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


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss > 0:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss > 0:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i + 1] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_094"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("failure_window", default=15.0, low=8.0, high=25.0),
            TunableParam("rsi_long", default=40.0, low=30.0, high=50.0),
            TunableParam("rsi_short", default=60.0, low=50.0, high=70.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.007),
            TunableParam("vix_low", default=13.0, low=10.0, high=16.0),
            TunableParam("vix_high", default=22.0, low=18.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        fail_win = int(params.get("failure_window", 15.0))
        rsi_long = params.get("rsi_long", 40.0)
        rsi_short = params.get("rsi_short", 60.0)
        stop_pct = params.get("stop_loss_pct", 0.004)
        target_pct = params.get("target_pct", 0.004)
        vix_lo = params.get("vix_low", 13.0)
        vix_hi = params.get("vix_high", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_min = df["time_minutes"].to_numpy()
        day_id = df["day_id"].to_numpy()

        # ── ORB 15-min ──
        orb_high = np.zeros(n, dtype=np.float64)
        orb_low = np.zeros(n, dtype=np.float64)
        orb_computed = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cur_h = high[i]
                cur_l = low[i]
            if time_min[i] < 570:
                cur_h = max(cur_h, high[i])
                cur_l = min(cur_l, low[i])
            orb_high[i] = cur_h
            orb_low[i] = cur_l
            if time_min[i] >= 570:
                orb_computed[i] = True

        # ── Detect failed breakouts ──
        # Track if breakout occurred above/below, and then price returned inside range
        breakout_up_bar = np.full(n, -1, dtype=np.int64)  # last bar that broke above
        breakout_down_bar = np.full(n, -1, dtype=np.int64)  # last bar that broke below
        was_above = np.zeros(n, dtype=np.bool_)
        was_below = np.zeros(n, dtype=np.bool_)

        for i in range(1, n):
            if not orb_computed[i]:
                continue
            if day_id[i] != day_id[i - 1]:
                continue
            # Track breakout occurrence
            if close[i] > orb_high[i]:
                breakout_up_bar[i] = i
                was_above[i] = True
            elif was_above[i - 1] and day_id[i] == day_id[i - 1]:
                breakout_up_bar[i] = breakout_up_bar[i - 1]

            if close[i] < orb_low[i]:
                breakout_down_bar[i] = i
                was_below[i] = True
            elif was_below[i - 1] and day_id[i] == day_id[i - 1]:
                breakout_down_bar[i] = breakout_down_bar[i - 1]

        # ── Failed upside breakout: was above, now back below, within window ──
        failed_up = np.zeros(n, dtype=np.bool_)
        failed_down = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            if not orb_computed[i]:
                continue
            # Failed up: breakout occurred recently, now close < orb_high
            if breakout_up_bar[i - 1] > 0 and close[i] < orb_high[i]:
                bars_since = i - breakout_up_bar[i - 1]
                if 0 < bars_since <= fail_win:
                    # 2-bar confirmation back inside
                    if i >= 2 and close[i - 1] < orb_high[i] and day_id[i] == day_id[i - 1]:
                        failed_up[i] = True
            # Failed down: breakout occurred recently, now close > orb_low
            if breakout_down_bar[i - 1] > 0 and close[i] > orb_low[i]:
                bars_since = i - breakout_down_bar[i - 1]
                if 0 < bars_since <= fail_win:
                    if i >= 2 and close[i - 1] > orb_low[i] and day_id[i] == day_id[i - 1]:
                        failed_down[i] = True

        rsi = _compute_rsi(close, 14)
        vix_ok = (vix >= vix_lo) & (vix <= vix_hi)
        time_ok = (time_min >= 570) & (time_min <= 870)

        # ── Failed down => go long (trapped shorts liquidating) ──
        long_entry = failed_down & (rsi > rsi_long) & (close > orb_low) & vix_ok & time_ok
        # ── Failed up => go short (trapped longs liquidating) ──
        short_entry = failed_up & (rsi < rsi_short) & (close < orb_high) & vix_ok & time_ok

        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=90,
        )
