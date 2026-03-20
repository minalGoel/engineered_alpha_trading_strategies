"""Multi-Timeframe Confirmation — Opus_45

Thesis: 5-minute EMA trend (proxied by EMA(45)/EMA(105) on 1-min)
plus 1-minute RSI pullback. Enter when RSI crosses back from extreme
in direction of higher-timeframe trend.
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


def _compute_ema(arr, period):
    """EMA with standard multiplier."""
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    k = 2.0 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)
    return ema


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
    name = "opus_multi_timeframe_confirmation_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_long_thresh", default=25.0, low=15.0, high=35.0),
            TunableParam("rsi_short_thresh", default=75.0, low=65.0, high=85.0),
            TunableParam("target_atr_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("stop_atr_mult", default=1.0, low=0.5, high=1.5),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rsi_long = params.get("rsi_long_thresh", 25.0)
        rsi_short = params.get("rsi_short_thresh", 75.0)
        tgt_mult = params.get("target_atr_mult", 1.5)
        stp_mult = params.get("stop_atr_mult", 1.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Proxies for 5-min EMA(9) and EMA(21): 1-min EMA(45) and EMA(105)
        ema45 = _compute_ema(close, 45)
        ema105 = _compute_ema(close, 105)

        rsi7 = _compute_rsi(close, 7)
        atr = _compute_atr(high, low, close, 20)

        uptrend = ema45 > ema105
        downtrend = ema45 < ema105

        # RSI crosses back from extreme
        rsi_cross_up = np.zeros(n, dtype=np.bool_)
        rsi_cross_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if rsi7[i - 1] < rsi_long and rsi7[i] >= rsi_long:
                rsi_cross_up[i] = True
            if rsi7[i - 1] > rsi_short and rsi7[i] <= rsi_short:
                rsi_cross_down[i] = True

        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Long: uptrend, RSI was < 25 and crosses above, close > VWAP
        long_entry = uptrend & rsi_cross_up & (close > vwap) & time_ok

        # Short: downtrend, RSI was > 75 and crosses below, close < VWAP
        short_entry = downtrend & rsi_cross_down & (close < vwap) & time_ok

        # Signal exit: trend flips (EMA45 crosses EMA105)
        exit_long = np.zeros(n, dtype=np.bool_)
        exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # Uptrend ended: ema45 crosses below ema105
            if ema45[i - 1] >= ema105[i - 1] and ema45[i] < ema105[i]:
                exit_long[i] = True
            # Downtrend ended: ema45 crosses above ema105
            if ema45[i - 1] <= ema105[i - 1] and ema45[i] > ema105[i]:
                exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=exit_long,
            signal_exit_short=exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_atr_mult=tgt_mult,
            stop_loss_atr_mult=stp_mult,
            time_stop_bars=60,
        )
