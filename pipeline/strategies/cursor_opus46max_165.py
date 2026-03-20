"""Weighted Indicator v1 — cursor_opus46max_165

Thesis: Dynamically weight 5 indicators (VWAP, RSI, MACD, EMA, OBV)
based on recent performance. Use equal weights as baseline (no historical
performance data available). Weighted composite score for entry.
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


class Strategy(BaseStrategy):
    name = "cursor_opus46max_165"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 7

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("score_thresh", default=0.60, low=0.40, high=0.80),
            TunableParam("rsi_long_thresh", default=35.0, low=25.0, high=40.0),
            TunableParam("rsi_short_thresh", default=65.0, low=60.0, high=75.0),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("stop_loss_pct", default=0.0022, low=0.0015, high=0.003),
            TunableParam("trailing_stop_pct", default=0.0012, low=0.0008, high=0.002),
            TunableParam("trailing_activate_pct", default=0.0022, low=0.0015, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        score_thresh = params.get("score_thresh", 0.60)
        rsi_lt = params.get("rsi_long_thresh", 35.0)
        rsi_st = params.get("rsi_short_thresh", 65.0)
        target_pct = params.get("target_pct", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.0022)
        trail_pct = params.get("trailing_stop_pct", 0.0012)
        trail_act = params.get("trailing_activate_pct", 0.0022)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
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

        # Ind1: VWAP signal
        ind1 = np.sign(close - vwap)

        # Ind2: RSI signal
        rsi = _compute_rsi(close, 14)
        ind2 = np.where(rsi < rsi_lt, 1, np.where(rsi > rsi_st, -1, 0)).astype(np.float64)

        # Ind3: MACD signal
        ema12 = _compute_ema(close, 12)
        ema26 = _compute_ema(close, 26)
        macd = ema12 - ema26
        macd_signal = _compute_ema(macd, 9)
        ind3 = np.sign(macd - macd_signal)

        # Ind4: EMA 9/21 signal
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)
        ind4 = np.sign(ema9 - ema21)

        # Ind5: OBV signal
        obv = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i] > close[i - 1]:
                obv[i] = obv[i - 1] + volume[i]
            elif close[i] < close[i - 1]:
                obv[i] = obv[i - 1] - volume[i]
            else:
                obv[i] = obv[i - 1]
        obv_sma20 = np.zeros(n, dtype=np.float64)
        for i in range(19, n):
            obv_sma20[i] = np.mean(obv[i - 19:i + 1])
        ind5 = np.sign(obv - obv_sma20)

        # Equal-weighted score (divide by 5 to normalize)
        weighted_score = (ind1 + ind2 + ind3 + ind4 + ind5) / 5.0

        # Count how many indicators agree
        bull_count = ((ind1 > 0).astype(np.int32) + (ind2 > 0).astype(np.int32) +
                      (ind3 > 0).astype(np.int32) + (ind4 > 0).astype(np.int32) +
                      (ind5 > 0).astype(np.int32))
        bear_count = ((ind1 < 0).astype(np.int32) + (ind2 < 0).astype(np.int32) +
                      (ind3 < 0).astype(np.int32) + (ind4 < 0).astype(np.int32) +
                      (ind5 < 0).astype(np.int32))

        # 2-bar confirmation
        long_raw = (weighted_score > score_thresh) & (bull_count >= 3) & (close > vwap)
        short_raw = (weighted_score < -score_thresh) & (bear_count >= 3) & (close < vwap)

        long_conf = np.zeros(n, dtype=np.bool_)
        short_conf = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if long_raw[i] and long_raw[i - 1]:
                long_conf[i] = True
            if short_raw[i] and short_raw[i - 1]:
                short_conf[i] = True

        # Volume filter
        avg_vol = df["volume"].rolling_mean(15).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 570) & (time_mins <= 910)
        vix_ok = vix < 25.0

        long_entry = long_conf & vol_ok & time_ok & vix_ok
        short_entry = short_conf & vol_ok & time_ok & vix_ok

        # Signal exit: score crosses zero
        sig_exit_long = weighted_score <= 0
        sig_exit_short = weighted_score >= 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=60,
        )
