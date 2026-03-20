"""DeMark TD Sequential — cursor_opus46max_147

Thesis: TD Sequential counts consecutive bars closing above/below their
close 4 bars earlier. A completed 9-count signals trend exhaustion.
Fade the trend at count 9 with RSI confirmation.
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


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_147"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 915     # 15:15
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("comparison_offset", default=4.0, low=3.0, high=6.0),
            TunableParam("completion_count", default=9.0, low=7.0, high=13.0),
            TunableParam("rsi_buy_thresh", default=35.0, low=20.0, high=45.0),
            TunableParam("rsi_sell_thresh", default=65.0, low=55.0, high=80.0),
            TunableParam("stop_bps", default=12.0, low=6.0, high=20.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        offset = int(params.get("comparison_offset", 4.0))
        target_count = int(params.get("completion_count", 9.0))
        rsi_buy = params.get("rsi_buy_thresh", 35.0)
        rsi_sell = params.get("rsi_sell_thresh", 65.0)
        stop_bps = params.get("stop_bps", 12.0)
        vix_max = params.get("vix_max", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr = _compute_atr(high, low, close, 14)
        rsi = _compute_rsi(close, 14)

        # TD Sequential counting
        buy_count = np.zeros(n, dtype=np.int32)   # counts bars closing < close[offset]
        sell_count = np.zeros(n, dtype=np.int32)   # counts bars closing > close[offset]

        for i in range(offset, n):
            # Buy setup: close < close[offset bars ago]
            if close[i] < close[i - offset]:
                buy_count[i] = buy_count[i-1] + 1 if buy_count[i-1] > 0 or True else 1
                sell_count[i] = 0
            elif close[i] > close[i - offset]:
                sell_count[i] = sell_count[i-1] + 1 if sell_count[i-1] > 0 or True else 1
                buy_count[i] = 0
            else:
                buy_count[i] = 0
                sell_count[i] = 0

        # Fix: reset properly
        for i in range(offset, n):
            if close[i] < close[i - offset]:
                if i > offset and close[i-1] >= close[i-1 - offset]:
                    buy_count[i] = 1
                else:
                    buy_count[i] = buy_count[i-1] + 1
                sell_count[i] = 0
            elif close[i] > close[i - offset]:
                if i > offset and close[i-1] <= close[i-1 - offset]:
                    sell_count[i] = 1
                else:
                    sell_count[i] = sell_count[i-1] + 1
                buy_count[i] = 0
            else:
                buy_count[i] = 0
                sell_count[i] = 0

        # Perfection rule: low of bar 8 or 9 <= low of bar 6
        buy_perfected = np.zeros(n, dtype=np.bool_)
        sell_perfected = np.zeros(n, dtype=np.bool_)
        for i in range(target_count, n):
            if buy_count[i] == target_count:
                # Check: low of bar count=8 or count=9 <= low of bar count=6
                if i >= 3:
                    if low[i] <= low[i-3] or low[i-1] <= low[i-3]:
                        buy_perfected[i] = True
                    else:
                        buy_perfected[i] = True  # allow imperfect too
            if sell_count[i] == target_count:
                if i >= 3:
                    if high[i] >= high[i-3] or high[i-1] >= high[i-3]:
                        sell_perfected[i] = True
                    else:
                        sell_perfected[i] = True

        # Volume declining on last 3 bars (exhaustion)
        vol_declining = np.zeros(n, dtype=np.bool_)
        for i in range(3, n):
            vol_declining[i] = volume[i] < volume[i-3]

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 565) & (time_mins <= 900)

        # Long: buy setup completed (9 down bars), RSI oversold, bounce
        long_entry_cond = buy_perfected & (rsi < rsi_buy) & vix_ok & time_ok
        # Confirmation: close > close[1] on bar after setup
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if long_entry_cond[i-1] and close[i] > close[i-1]:
                long_entry[i] = True
            if sell_perfected[i-1] and rsi[i-1] > rsi_sell and vix_ok[i] and time_ok[i]:
                if close[i] < close[i-1]:
                    short_entry[i] = True

        # Signal exit: close crosses above close[offset] (trend reversal per DeMark)
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(offset, n):
            if close[i] >= close[i - offset]:
                sig_exit_long[i] = True
            if close[i] <= close[i - offset]:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_bps / 10000.0,
            target_pct=0.002,
            trailing_stop_pct=0.0005,
            trailing_activate_pct=0.001,
            time_stop_bars=15,
        )
