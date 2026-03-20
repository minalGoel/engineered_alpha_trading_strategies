"""Pairs Trading IT v1 — cursor_opus46max_052

Thesis: TCS/INFOSYS-like IT pair spread mean-reverts intraday.
Adapted: stock vs index spread z-score with RSI filter.
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
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    else:
        rsi[period] = 100.0
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 1e-10:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        else:
            rsi[i] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_052"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", default=1.8, low=1.2, high=3.0),
            TunableParam("spread_lookback", default=90.0, low=45.0, high=180.0),
            TunableParam("rsi_low", default=25.0, low=15.0, high=35.0),
            TunableParam("rsi_high", default=75.0, low=65.0, high=85.0),
            TunableParam("vix_max", default=20.0, low=14.0, high=26.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.007),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_entry = params.get("zscore_entry", 1.8)
        lb = int(params.get("spread_lookback", 90.0))
        rsi_low = params.get("rsi_low", 25.0)
        rsi_high = params.get("rsi_high", 75.0)
        vix_max = params.get("vix_max", 20.0)
        stop_pct = params.get("stop_loss_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Spread
        log_close = np.log(np.clip(close, 1e-8, None))
        log_index = np.log(np.clip(index_close, 1e-8, None))
        spread = log_close - log_index

        # Z-score via EMA-based mean
        spread_mean = np.zeros(n, dtype=np.float64)
        alpha_ema = 2.0 / (lb + 1)
        spread_mean[0] = spread[0]
        for i in range(1, n):
            spread_mean[i] = alpha_ema * spread[i] + (1 - alpha_ema) * spread_mean[i-1]

        zscore = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            seg = spread[i - lb:i + 1]
            std = np.std(seg)
            if std > 1e-10:
                zscore[i] = (spread[i] - spread_mean[i]) / std

        # RSI of spread
        spread_rsi = _compute_rsi(spread, 14)

        # Was below/above for 2+ bars then first uptick/downtick
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 560) & (time_mins <= 840)  # 09:20 - 14:00

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        bars_below = 0
        bars_above = 0
        for i in range(1, n):
            if zscore[i-1] < -zs_entry:
                bars_below += 1
            else:
                bars_below = 0
            if zscore[i-1] > zs_entry:
                bars_above += 1
            else:
                bars_above = 0

            if (bars_below >= 2 and zscore[i] > zscore[i-1]
                    and spread_rsi[i] < rsi_low and vix_ok[i] and time_ok[i]):
                long_entry[i] = True
            if (bars_above >= 2 and zscore[i] < zscore[i-1]
                    and spread_rsi[i] > rsi_high and vix_ok[i] and time_ok[i]):
                short_entry[i] = True

        signal_exit_long = zscore > 0
        signal_exit_short = zscore < 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=120,
        )
