"""Quality Momentum v1 — cursor_opus46max_156

Thesis: High-quality companies with strong intraday EMA momentum are
less likely to reverse. Long quality names with momentum; short low-quality
names with negative momentum. EMA(9)/EMA(21) crossover with widening gap.
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
    name = "cursor_opus46max_156"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("mom30_thresh_bps", default=20.0, low=10.0, high=40.0),
            TunableParam("gap_widen_bars", default=5.0, low=3.0, high=10.0),
            TunableParam("stop_atr_mult", default=1.0, low=0.5, high=1.5),
            TunableParam("target_atr_mult", default=1.5, low=1.0, high=2.5),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        mom_thresh = params.get("mom30_thresh_bps", 20.0)
        gap_bars = int(params.get("gap_widen_bars", 5.0))
        stop_mult = params.get("stop_atr_mult", 1.0)
        target_mult = params.get("target_atr_mult", 1.5)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ATR
        atr = _compute_atr(high, low, close, 14)

        # EMAs
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)

        # 30-bar momentum in bps
        mom30 = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            if close[i - 30] > 0:
                mom30[i] = (close[i] / close[i - 30] - 1.0) * 10000.0

        # EMA gap widening
        ema_gap = ema9 - ema21
        gap_widening_long = np.zeros(n, dtype=np.bool_)
        gap_widening_short = np.zeros(n, dtype=np.bool_)
        for i in range(gap_bars, n):
            if ema_gap[i] > ema_gap[i - gap_bars] and ema_gap[i] > 0:
                gap_widening_long[i] = True
            if ema_gap[i] < ema_gap[i - gap_bars] and ema_gap[i] < 0:
                gap_widening_short[i] = True

        # Volume filter
        avg_vol = df["volume"].rolling_mean(15).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 570) & (time_mins <= 910)

        # Long: EMA9 > EMA21, momentum > thresh, above VWAP, gap widening
        long_entry = (ema9 > ema21) & (mom30 > mom_thresh) & (close > vwap) & \
                     gap_widening_long & vol_ok & time_ok

        # Short: EMA9 < EMA21, momentum < -thresh, below VWAP, gap widening (short)
        short_entry = (ema9 < ema21) & (mom30 < -mom_thresh) & (close < vwap) & \
                      gap_widening_short & vol_ok & time_ok

        # Signal exit: EMA cross against position
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if ema9[i - 1] >= ema21[i - 1] and ema9[i] < ema21[i]:
                sig_exit_long[i] = True
            if ema9[i - 1] <= ema21[i - 1] and ema9[i] > ema21[i]:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_mult,
            target_atr_mult=target_mult,
            trailing_stop_pct=0.0,
            trailing_activate_pct=0.0,
            time_stop_bars=75,
        )
