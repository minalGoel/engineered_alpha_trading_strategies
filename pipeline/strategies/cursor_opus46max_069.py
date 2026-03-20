"""Opening Drive Post Gap v1 — cursor_opus46max_069

Thesis: First 5-min opening drive aligned with gap + heavy volume predicts
20-40 min continuation. Enter at 5th bar close if drive confirms gap.
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


def _ema(data, period):
    n = len(data)
    out = np.zeros(n, dtype=np.float64)
    out[0] = data[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_069"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 660     # 11:00
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min", default=0.003, low=0.002, high=0.006),
            TunableParam("vol_surge_min", default=2.0, low=1.3, high=3.0),
            TunableParam("body_ratio_min", default=0.6, low=0.4, high=0.8),
            TunableParam("vix_min", default=12.0, low=8.0, high=16.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_min = params.get("gap_min", 0.003)
        vol_surge_min = params.get("vol_surge_min", 2.0)
        body_ratio_min = params.get("body_ratio_min", 0.6)
        vix_min = params.get("vix_min", 12.0)
        vix_max = params.get("vix_max", 25.0)
        target_pct = params.get("target_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high_ = df["high"].to_numpy().astype(np.float64)
        low_ = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr10 = _compute_atr(high_, low_, close, 10)
        ema5 = _ema(close, 5)

        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)

        # Per-day: gap, opening drive analysis
        gap_pct = np.zeros(n, dtype=np.float64)
        drive_bar = np.zeros(n, dtype=np.bool_)  # True on the 5th bar of each day
        unique_days = np.unique(day_ids)
        prev_close = np.nan

        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) == 0:
                continue
            day_open = open_[idx[0]]
            if not np.isnan(prev_close) and prev_close > 0:
                gap_pct[idx] = (day_open - prev_close) / prev_close
            if len(idx) >= 5:
                bar5 = idx[4]  # 5th bar (0-indexed)
                f5_close = close[bar5]
                f5_range = np.max(high_[idx[:5]]) - np.min(low_[idx[:5]])
                f5_vol = np.sum(volume[idx[:5]])
                avg_f5_vol = np.mean(avg_vol[idx[:5]]) * 5
                vol_surge = f5_vol / max(avg_f5_vol, 1.0)
                body = abs(f5_close - day_open)
                body_ratio = body / max(f5_range, 1e-8)

                # Drive aligned with gap
                drive_aligned = ((gap_pct[bar5] > 0 and f5_close > day_open) or
                                 (gap_pct[bar5] < 0 and f5_close < day_open))

                if (abs(gap_pct[bar5]) > gap_min and drive_aligned
                        and vol_surge > vol_surge_min and body_ratio > body_ratio_min):
                    drive_bar[bar5] = True
            prev_close = close[idx[-1]]

        vix_ok = (vix >= vix_min) & (vix <= vix_max)

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if drive_bar[i] and vix_ok[i]:
                if gap_pct[i] > 0:
                    long_entry[i] = True
                elif gap_pct[i] < 0:
                    short_entry[i] = True

        # Signal exit: price crosses below EMA5 (long) or above (short)
        signal_exit_long = close < ema5
        signal_exit_short = close > ema5

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr10,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_atr_mult=0.5,
            trailing_stop_pct=0.001,
            trailing_activate_pct=0.002,
            time_stop_bars=40,
        )
