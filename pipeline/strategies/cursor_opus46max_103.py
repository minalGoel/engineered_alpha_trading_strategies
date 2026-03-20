"""Tick Direction Momentum v1 — cursor_opus46max_103

Thesis: Consecutive upticks aggregated over 1-min bars reveal net buying
pressure predicting short-term continuation.  Proxy tick direction with
close-to-close direction and bar close position ratio.
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


def _ema(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    out[0] = arr[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_103"
    is_long_only = False
    session_start = 575   # 09:35
    session_end = 920     # 15:20
    max_trades_per_day = 25
    assumptions = ["Tick-level data proxied via close-to-close direction ratio"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("tick_ema_long", default=0.60, low=0.55, high=0.70),
            TunableParam("tick_ema_short", default=0.40, low=0.30, high=0.45),
            TunableParam("momentum_thresh", default=0.05, low=0.02, high=0.10),
            TunableParam("price_roc_thresh", default=5.0, low=2.0, high=10.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.0008, low=0.0004, high=0.0015),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        tick_ema_long = params.get("tick_ema_long", 0.60)
        tick_ema_short = params.get("tick_ema_short", 0.40)
        mom_thresh = params.get("momentum_thresh", 0.05)
        roc_thresh = params.get("price_roc_thresh", 5.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.0008)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Proxy tick_ratio: use (close - low) / (high - low) as bar-level uptick ratio
        bar_range = high - low
        bar_range = np.where(bar_range < 1e-10, 1e-10, bar_range)
        tick_ratio = (close - low) / bar_range  # 0 to 1

        # EMA(tick_ratio, 5)
        tick_ema = _ema(tick_ratio, 5)

        # Tick momentum: tick_ema - tick_ema[3]
        tick_momentum = np.zeros(n, dtype=np.float64)
        for i in range(3, n):
            tick_momentum[i] = tick_ema[i] - tick_ema[i-3]

        # Price ROC over 3 bars (in bps)
        price_roc = np.zeros(n, dtype=np.float64)
        for i in range(3, n):
            if close[i-3] > 1e-10:
                price_roc[i] = (close[i] - close[i-3]) / close[i-3] * 10000.0

        # Volume filter
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        vix_ok = vix < vix_max

        long_entry = ((tick_ema > tick_ema_long) & (tick_momentum > mom_thresh) &
                      (price_roc > roc_thresh) & (close > vwap) &
                      (tick_ratio > 0.55) & vol_ok & vix_ok)
        short_entry = ((tick_ema < tick_ema_short) & (tick_momentum < -mom_thresh) &
                       (price_roc < -roc_thresh) & (close < vwap) &
                       (tick_ratio < 0.45) & vol_ok & vix_ok)

        # Signal exit: tick_ema crosses 0.50
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if tick_ema[i] <= 0.50 and tick_ema[i-1] > 0.50:
                sig_exit_long[i] = True
            if tick_ema[i] >= 0.50 and tick_ema[i-1] < 0.50:
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=0.0015,
            trailing_stop_pct=0.0004,
            trailing_activate_pct=0.0008,
            time_stop_bars=5,
        )
