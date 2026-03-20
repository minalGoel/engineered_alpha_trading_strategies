"""Multi-Indicator Consensus v1 — cursor_opus46max_154

Thesis: Requiring consensus from 5 diverse indicators (VWAP, RSI, MACD,
Stochastic, Volume ROC) filters noise. Enter when >= 4/5 agree for 2 bars.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


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
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


def _compute_ema(arr, period):
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = arr[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)
    return ema


def _compute_stochastic_k(high_arr, low_arr, close_arr, period, smooth):
    """Stochastic %K with smoothing."""
    n = len(close_arr)
    raw_k = np.full(n, 50.0, dtype=np.float64)
    for i in range(period - 1, n):
        hh = np.max(high_arr[i - period + 1:i + 1])
        ll = np.min(low_arr[i - period + 1:i + 1])
        if hh - ll > 0:
            raw_k[i] = (close_arr[i] - ll) / (hh - ll) * 100.0
    # Smooth with SMA
    stoch_k = np.full(n, 50.0, dtype=np.float64)
    for i in range(period + smooth - 2, n):
        stoch_k[i] = np.mean(raw_k[i - smooth + 1:i + 1])
    return stoch_k


class Strategy(BaseStrategy):
    name = "cursor_opus46max_154"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("consensus_thresh", default=4.0, low=3.0, high=5.0),
            TunableParam("rsi_long_thresh", default=35.0, low=25.0, high=40.0),
            TunableParam("rsi_short_thresh", default=65.0, low=60.0, high=75.0),
            TunableParam("stoch_long_thresh", default=25.0, low=15.0, high=35.0),
            TunableParam("stoch_short_thresh", default=75.0, low=65.0, high=85.0),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        consensus_t = int(params.get("consensus_thresh", 4.0))
        rsi_lt = params.get("rsi_long_thresh", 35.0)
        rsi_st = params.get("rsi_short_thresh", 65.0)
        stoch_lt = params.get("stoch_long_thresh", 25.0)
        stoch_st = params.get("stoch_short_thresh", 75.0)
        target_pct = params.get("target_pct", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Indicators
        rsi = _compute_rsi(close, 14)
        ema12 = _compute_ema(close, 12)
        ema26 = _compute_ema(close, 26)
        macd = ema12 - ema26
        macd_signal = _compute_ema(macd, 9)
        stoch_k = _compute_stochastic_k(high, low, close, 14, 3)

        # Volume ROC
        vol_roc = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            if volume[i - 10] > 0:
                vol_roc[i] = (volume[i] / volume[i - 10] - 1.0) * 100.0

        # Build signals: each is +1 (bullish), -1 (bearish), or 0 (neutral)
        vwap_sig = np.where(close > vwap, 1, np.where(close < vwap, -1, 0)).astype(np.float64)
        rsi_sig = np.where(rsi < rsi_lt, 1, np.where(rsi > rsi_st, -1, 0)).astype(np.float64)
        macd_sig = np.where(macd > macd_signal, 1, -1).astype(np.float64)
        stoch_sig = np.where(stoch_k < stoch_lt, 1, np.where(stoch_k > stoch_st, -1, 0)).astype(np.float64)
        vroc_sig = np.where(vol_roc > 50.0, 1, 0).astype(np.float64)

        consensus = vwap_sig + rsi_sig + macd_sig + stoch_sig + vroc_sig

        # 2-bar confirmation
        long_raw = consensus >= consensus_t
        short_raw = consensus <= -consensus_t

        long_conf = np.zeros(n, dtype=np.bool_)
        short_conf = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if long_raw[i] and long_raw[i - 1]:
                long_conf[i] = True
            if short_raw[i] and short_raw[i - 1]:
                short_conf[i] = True

        # Filters
        time_ok = (time_mins >= 565) & (time_mins <= 910)
        vix_ok = vix < 25.0

        long_entry = long_conf & time_ok & vix_ok
        short_entry = short_conf & time_ok & vix_ok

        # Signal exit: consensus crosses zero
        sig_exit_long = consensus <= 0
        sig_exit_short = consensus >= 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=0.0015,
            trailing_activate_pct=0.0025,
            time_stop_bars=60,
        )
