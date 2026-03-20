# AUDIT FIX: Rolling max/min included current bar's high/low, making close > orb_high impossible.
# Fixed by shifting orb_high/orb_low by 1 (exclude current bar from the lookback window).
"""ORB Momentum v32 — Gemini31Pro Strategy 032

Thesis: Opening range breakout with institutional order flow.
Uses HIGHEST(high,21)/LOWEST(low,21), VWAP confirmation,
volume > SMA(volume,10). Target 3.0 * ATR(10). Stop 1.5 * ATR(14).
VIX > 15. Time stop 15 bars.
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


def _rolling_max(arr, period):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        out[i] = np.max(arr[i - period + 1:i + 1])
    return out


def _rolling_min(arr, period):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        out[i] = np.min(arr[i - period + 1:i + 1])
    return out


class Strategy(BaseStrategy):
    name = "gemini31pro_orb_momentum_v32"
    is_long_only = False
    session_start = 570
    session_end = 915
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("target_atr_mult", default=3.0, low=1.5, high=4.5),
            TunableParam("vix_min", default=15.0, low=10.0, high=20.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        target_mult = params.get("target_atr_mult", 3.0)
        vix_min = params.get("vix_min", 15.0)
        be_pct = params.get("breakeven_pct", 0.005)

        n = len(df)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ORB: HIGHEST(high, 21) and LOWEST(low, 21) (shifted 1 bar to exclude current bar) ──
        # Shift by 1 so that we compare close[i] against the max/min of the PREVIOUS 21 bars.
        # Without the shift, close > orb_high is impossible because high[i] >= close[i].
        orb_high_raw = _rolling_max(high, 21)
        orb_low_raw = _rolling_min(low, 21)
        orb_high = np.roll(orb_high_raw, 1)
        orb_high[0] = orb_high_raw[0]
        orb_low = np.roll(orb_low_raw, 1)
        orb_low[0] = orb_low_raw[0]
        orb_high = np.nan_to_num(orb_high, nan=1e10)
        orb_low = np.nan_to_num(orb_low, nan=-1e10)

        # ── Volume filter: volume > SMA(volume, 10) ──
        avg_vol = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── ATR(14) for stops, ATR(10) for target ──
        atr_14 = _compute_atr(high, low, close, 14)
        atr_10 = _compute_atr(high, low, close, 10)

        # ── Filters ──
        vix_ok = vix > vix_min
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (close > orb_high) & (close > vwap) & vol_ok & vix_ok & time_ok
        short_entry = (close < orb_low) & (close < vwap) & vol_ok & vix_ok & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr_14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=1.5,
            target_atr_mult=target_mult,
            breakeven_pct=be_pct,
            time_stop_bars=15,
        )
