"""Deviation from EMA v1 — cursor_opus46max_015

Thesis: Price acts like a rubber band around EMA(50). When deviation exceeds
1.5 ATR and EMA slope is flat, enter for reversion. Target is EMA(50) touch.
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


def _compute_ema(close, period):
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = close[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_015"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 900     # 15:00
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("dev_atr_thresh", default=1.5, low=1.0, high=2.5),
            TunableParam("dev_atr_stop", default=2.5, low=2.0, high=3.5),
            TunableParam("vix_max", default=25.0, low=16.0, high=30.0),
            TunableParam("breakeven_pct", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        dev_thresh = params.get("dev_atr_thresh", 1.5)
        dev_stop = params.get("dev_atr_stop", 2.5)
        vix_max = params.get("vix_max", 25.0)
        be_pct = params.get("breakeven_pct", 0.003)

        n = len(df)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── EMA(50) ──
        ema50 = _compute_ema(close, 50)

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)
        safe_atr = np.clip(atr, 1e-10, None)

        # ── Deviation in ATR units ──
        ema_dev = close - ema50
        ema_dev_atr = ema_dev / safe_atr

        # ── EMA slope (10-bar change) ──
        ema_slope = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            ema_slope[i] = ema50[i] - ema50[i - 10]

        # ── Snap back confirmation ──
        snapping_back = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            snapping_back[i] = abs(ema_dev_atr[i]) < abs(ema_dev_atr[i - 1])

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── EMA slope near flat ──
        safe_close = np.where(close > 0, close, 1.0)
        slope_flat = np.abs(ema_slope) < 0.01 * safe_close

        # ── Filters ──
        time_ok = (time_mins >= 600) & (time_mins <= 870)
        vix_ok = vix < vix_max

        # ── Entries ──
        long_entry = (
            (ema_dev_atr < -dev_thresh)
            & slope_flat
            & vol_ok
            & snapping_back
            & vix_ok & time_ok
        )
        short_entry = (
            (ema_dev_atr > dev_thresh)
            & slope_flat
            & vol_ok
            & snapping_back
            & vix_ok & time_ok
        )

        # ── Signal exit: deviation extends to stop level ──
        sig_exit_long = ema_dev_atr < -dev_stop
        sig_exit_short = ema_dev_atr > dev_stop

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=ema50.copy(),
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=35,
        )
