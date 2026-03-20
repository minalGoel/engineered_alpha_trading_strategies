"""Gap Fade Large v1 — cursor_opus46max_063

Thesis: Large gaps (>1.5%) fade 55-60% when not news-driven.
MACD cross confirmation, target 40% of excess gap.
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


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff = close[i] - close[i-1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    if avg_loss > 1e-10:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 1e-10:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_063"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 840     # 14:00
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min", default=0.015, low=0.01, high=0.025),
            TunableParam("rsi_low", default=25.0, low=15.0, high=35.0),
            TunableParam("rsi_high", default=75.0, low=65.0, high=85.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.007, low=0.003, high=0.012),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_min = params.get("gap_min", 0.015)
        rsi_low = params.get("rsi_low", 25.0)
        rsi_high = params.get("rsi_high", 75.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.007)

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

        atr14 = _compute_atr(high_, low_, close, 14)
        rsi14 = _compute_rsi(close, 14)

        # MACD
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)
        macd_line = ema12 - ema26
        macd_signal = _ema(macd_line, 9)

        # Gap
        gap_pct = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        prev_close = np.nan
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) == 0:
                continue
            day_open = open_[idx[0]]
            if not np.isnan(prev_close) and prev_close > 0:
                gap_pct[idx] = (day_open - prev_close) / prev_close
            prev_close = close[idx[-1]]

        # Volume climax: > 3x avg
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_climax = volume > 3.0 * avg_vol

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 560) & (time_mins <= 630)

        # MACD cross confirmation
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # Large gap down: MACD crosses above signal and no volume climax
            if (gap_pct[i] < -gap_min and rsi14[i] < rsi_low
                    and vix_ok[i] and time_ok[i] and not vol_climax[i]
                    and macd_line[i-1] <= macd_signal[i-1] and macd_line[i] > macd_signal[i]):
                long_entry[i] = True
            # Large gap up: MACD crosses below signal
            if (gap_pct[i] > gap_min and rsi14[i] > rsi_high
                    and vix_ok[i] and time_ok[i] and not vol_climax[i]
                    and macd_line[i-1] >= macd_signal[i-1] and macd_line[i] < macd_signal[i]):
                short_entry[i] = True

        # Signal exit: volume climax in gap direction
        signal_exit_long = vol_climax & (close < open_)
        signal_exit_short = vol_climax & (close > open_)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=0.7,
            target_pct=0.006,
            time_stop_bars=150,
        )
