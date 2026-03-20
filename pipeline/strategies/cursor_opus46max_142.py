"""Simplified Elliott Wave 3 — cursor_opus46max_142

Thesis: Detect Wave 1 (impulse), Wave 2 (38-62% retracement), then trade
the Wave 3 breakout. Wave 3 is the strongest wave with highest volume.
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
    name = "cursor_opus46max_142"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 915     # 15:15
    max_trades_per_day = 8
    assumptions = [
        "Elliott Wave identification simplified to impulse + retracement + breakout",
    ]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("wave1_min_bps", default=15.0, low=8.0, high=30.0),
            TunableParam("wave2_retrace_lo", default=0.382, low=0.25, high=0.45),
            TunableParam("wave2_retrace_hi", default=0.618, low=0.55, high=0.75),
            TunableParam("vol_surge", default=1.5, low=1.2, high=2.5),
            TunableParam("vix_max", default=25.0, low=15.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        w1_min = params.get("wave1_min_bps", 15.0)
        w2_lo = params.get("wave2_retrace_lo", 0.382)
        w2_hi = params.get("wave2_retrace_hi", 0.618)
        vol_surge = params.get("vol_surge", 1.5)
        vix_max = params.get("vix_max", 25.0)

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

        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        for i in range(20, n):
            if vix[i] > vix_max or time_mins[i] < 565 or time_mins[i] > 900:
                continue

            # Look for bullish Wave 1+2+3 pattern in last 20 bars
            # Find the low in [i-20..i-10], high in [i-10..i-5], then retracement
            window_low_idx = i - 20
            window_mid = i - 10
            window_recent = i - 5

            if window_low_idx < 0:
                continue

            # Bullish: find min in first half, max in second half
            seg1 = close[window_low_idx:window_mid]
            seg2 = close[window_mid:window_recent]

            if len(seg1) == 0 or len(seg2) == 0:
                continue

            a_rel = np.argmin(seg1)
            b_rel = np.argmax(seg2)
            a_idx = window_low_idx + a_rel
            b_idx = window_mid + b_rel

            if b_idx <= a_idx:
                continue

            wave1 = close[b_idx] - close[a_idx]
            safe_a = max(close[a_idx], 1e-10)
            wave1_bps = wave1 / safe_a * 10000.0

            if wave1_bps > w1_min:
                # Find retracement low between b_idx and now
                seg3 = close[b_idx:i]
                if len(seg3) < 2:
                    continue
                c_rel = np.argmin(seg3)
                c_idx = b_idx + c_rel
                retrace = (close[b_idx] - close[c_idx]) / wave1 if wave1 > 1e-10 else 0

                if w2_lo <= retrace <= w2_hi:
                    # Wave 3 breakout: close > wave1 high
                    if close[i] > close[b_idx] and volume[i] > vol_surge * avg_vol[i]:
                        if rsi[i] > 50 and rsi[i] > rsi[i-1]:
                            long_entry[i] = True

            # Bearish: find max in first half, min in second half
            a_rel_b = np.argmax(seg1)
            b_rel_b = np.argmin(seg2)
            a_idx_b = window_low_idx + a_rel_b
            b_idx_b = window_mid + b_rel_b

            if b_idx_b <= a_idx_b:
                continue

            wave1_b = close[a_idx_b] - close[b_idx_b]
            safe_ab = max(close[a_idx_b], 1e-10)
            wave1_bps_b = wave1_b / safe_ab * 10000.0

            if wave1_bps_b > w1_min:
                seg3_b = close[b_idx_b:i]
                if len(seg3_b) < 2:
                    continue
                c_rel_b = np.argmax(seg3_b)
                c_idx_b = b_idx_b + c_rel_b
                retrace_b = (close[c_idx_b] - close[b_idx_b]) / wave1_b if wave1_b > 1e-10 else 0

                if w2_lo <= retrace_b <= w2_hi:
                    if close[i] < close[b_idx_b] and volume[i] > vol_surge * avg_vol[i]:
                        if rsi[i] < 50 and rsi[i] < rsi[i-1]:
                            short_entry[i] = True

        # Signal exit: RSI extreme
        sig_exit_long = rsi > 80
        sig_exit_short = rsi < 20

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=1.5,
            target_pct=0.003,
            trailing_stop_pct=0.001,
            trailing_activate_pct=0.0015,
            time_stop_bars=30,
        )
