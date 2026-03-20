"""Acceleration Momentum v1 — cursor_opus46max_029

Thesis: Momentum acceleration (second derivative of price) captures the
rate at which momentum is changing. Entry on positive acceleration captures
the sweet spot before peak velocity. ROC_accel positive for 3+ bars confirms.
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


def _linreg_slope(data, period):
    """Rolling linear regression slope."""
    n = len(data)
    slope = np.zeros(n, dtype=np.float64)
    x = np.arange(period, dtype=np.float64)
    x_mean = np.mean(x)
    x_var = np.sum((x - x_mean) ** 2)
    if x_var < 1e-10:
        return slope
    for i in range(period - 1, n):
        y = data[i - period + 1:i + 1]
        y_mean = np.mean(y)
        slope[i] = np.sum((x - x_mean) * (y - y_mean)) / x_var
    return slope


class Strategy(BaseStrategy):
    name = "cursor_opus46max_029"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("accel_confirm_bars", default=3.0, low=2.0, high=5.0),
            TunableParam("roc_max", default=1.0, low=0.5, high=1.5),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.004),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        accel_bars = int(params.get("accel_confirm_bars", 3.0))
        roc_max = params.get("roc_max", 1.0)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        tgt_pct = params.get("target_pct", 0.004)

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

        # ── ROC(10) ──
        roc10 = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            if close[i-10] > 0:
                roc10[i] = (close[i] - close[i-10]) / close[i-10] * 100.0

        # ── ROC acceleration: ROC_10[t] - ROC_10[t-5] ──
        roc_accel = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            roc_accel[i] = roc10[i] - roc10[i-5]

        # ── ROC accel slope (linreg over 5 bars) ──
        accel_slope = _linreg_slope(roc_accel, 5)

        # ── Sustained acceleration ──
        accel_pos_count = np.zeros(n, dtype=np.int32)
        accel_neg_count = np.zeros(n, dtype=np.int32)
        for i in range(n):
            if roc_accel[i] > 0:
                accel_pos_count[i] = (accel_pos_count[i-1] + 1) if i > 0 else 1
            if roc_accel[i] < 0:
                accel_neg_count[i] = (accel_neg_count[i-1] + 1) if i > 0 else 1

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── Time filter ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            (roc10 > 0) & (roc10 < roc_max) &
            (roc_accel > 0) & (accel_slope > 0) &
            (accel_pos_count >= accel_bars) &
            (close > vwap) & vol_ok & time_ok
        )
        short_entry = (
            (roc10 < 0) & (roc10 > -roc_max) &
            (roc_accel < 0) & (accel_slope < 0) &
            (accel_neg_count >= accel_bars) &
            (close < vwap) & vol_ok & time_ok
        )

        # ── Signal exits: acceleration flips sign ──
        signal_exit_long = roc_accel < 0
        signal_exit_short = roc_accel > 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=tgt_pct,
            trailing_stop_pct=0.0015,
            trailing_activate_pct=0.002,
            time_stop_bars=30,
        )
