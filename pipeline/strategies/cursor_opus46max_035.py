"""Hull MA Momentum v1 — cursor_opus46max_035

Thesis: Hull Moving Average (HMA) with period 16 provides nearly zero-lag
smoothed price signal. HMA slope flips signal momentum shifts earlier than
EMA/SMA. Enter on 2-bar confirmed slope flip with VWAP alignment.
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


def _wma(data, period):
    """Weighted Moving Average."""
    n = len(data)
    wma = np.zeros(n, dtype=np.float64)
    weights = np.arange(1, period + 1, dtype=np.float64)
    w_sum = weights.sum()
    for i in range(period - 1, n):
        wma[i] = np.sum(data[i - period + 1:i + 1] * weights) / w_sum
    return wma


def _hma(data, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half_period = max(int(period / 2), 1)
    sqrt_period = max(int(np.sqrt(period)), 1)

    wma_half = _wma(data, half_period)
    wma_full = _wma(data, period)
    diff = 2.0 * wma_half - wma_full
    return _wma(diff, sqrt_period)


class Strategy(BaseStrategy):
    name = "cursor_opus46max_035"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("hma_period", default=16.0, low=10.0, high=24.0),
            TunableParam("confirm_bars", default=2.0, low=1.0, high=4.0),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("target_pct", default=0.0035, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        hma_period = int(params.get("hma_period", 16.0))
        confirm_bars = int(params.get("confirm_bars", 2.0))
        stop_pct = params.get("stop_loss_pct", 0.002)
        tgt_pct = params.get("target_pct", 0.0035)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── HMA ──
        hma = _hma(close, hma_period)

        # ── HMA slope ──
        hma_slope = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            hma_slope[i] = hma[i] - hma[i-1]

        # ── Slope flip detection with confirmation ──
        pos_slope_count = np.zeros(n, dtype=np.int32)
        neg_slope_count = np.zeros(n, dtype=np.int32)
        was_negative = np.zeros(n, dtype=np.bool_)
        was_positive = np.zeros(n, dtype=np.bool_)

        for i in range(1, n):
            if hma_slope[i] > 0:
                pos_slope_count[i] = pos_slope_count[i-1] + 1
                neg_slope_count[i] = 0
            elif hma_slope[i] < 0:
                neg_slope_count[i] = neg_slope_count[i-1] + 1
                pos_slope_count[i] = 0

            # Was the slope negative before this positive run?
            if pos_slope_count[i] == 1 and (i < 2 or hma_slope[i-1] <= 0):
                was_negative[i] = True
            elif pos_slope_count[i] > 1 and i > 0:
                was_negative[i] = was_negative[i-1]

            if neg_slope_count[i] == 1 and (i < 2 or hma_slope[i-1] >= 0):
                was_positive[i] = True
            elif neg_slope_count[i] > 1 and i > 0:
                was_positive[i] = was_positive[i-1]

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── HMA flip count in last 30 bars ──
        flip_arr = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if (hma_slope[i] > 0 and hma_slope[i-1] <= 0) or (hma_slope[i] < 0 and hma_slope[i-1] >= 0):
                flip_arr[i] = 1
        flip_count_30 = np.zeros(n, dtype=np.int32)
        for i in range(n):
            start = max(0, i - 29)
            flip_count_30[i] = np.sum(flip_arr[start:i+1])

        # ── Time filter ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            (hma_slope > 0) & (pos_slope_count >= confirm_bars) & was_negative &
            (close > hma) & (close > vwap) & vol_ok &
            (flip_count_30 <= 5) & time_ok
        )
        short_entry = (
            (hma_slope < 0) & (neg_slope_count >= confirm_bars) & was_positive &
            (close < hma) & (close < vwap) & vol_ok &
            (flip_count_30 <= 5) & time_ok
        )

        # ── Signal exits: HMA slope crosses zero ──
        signal_exit_long = hma_slope < 0
        signal_exit_short = hma_slope > 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=tgt_pct,
            trailing_stop_pct=0.0012,
            trailing_activate_pct=0.002,
            time_stop_bars=30,
        )
