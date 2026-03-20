"""RSI Extreme Reversion v1 — cursor_opus46max_014

Thesis: RSI(2) on 1-min bars reaches extreme levels (<5 or >95) and reverts
quickly. Enter when RSI(2) crosses back through 10/90 from extreme, confirmed
by RSI(14) being in neutral zone (30-70).
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


class Strategy(BaseStrategy):
    name = "cursor_opus46max_014"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi2_extreme_long", default=5.0, low=2.0, high=10.0),
            TunableParam("rsi2_extreme_short", default=95.0, low=90.0, high=98.0),
            TunableParam("rsi2_confirm_long", default=10.0, low=5.0, high=20.0),
            TunableParam("rsi2_confirm_short", default=90.0, low=80.0, high=95.0),
            TunableParam("vix_max", default=25.0, low=16.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        r2_el = params.get("rsi2_extreme_long", 5.0)
        r2_es = params.get("rsi2_extreme_short", 95.0)
        r2_cl = params.get("rsi2_confirm_long", 10.0)
        r2_cs = params.get("rsi2_confirm_short", 90.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.002)

        n = len(df)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── RSI(2) and RSI(14) ──
        rsi2 = _compute_rsi(close, 2)
        rsi14 = _compute_rsi(close, 14)

        # ── ATR ──
        atr = _compute_atr(high, low, close, 10)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.8 * avg_vol

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 885)
        vix_ok = vix < vix_max
        rsi14_neutral = (rsi14 > 30) & (rsi14 < 70)

        # ── Entry: RSI(2) crosses from extreme back through confirm ──
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if rsi2[i - 1] < r2_el and rsi2[i] >= r2_cl:
                long_entry[i] = True
            if rsi2[i - 1] > r2_es and rsi2[i] <= r2_cs:
                short_entry[i] = True

        long_entry = long_entry & rsi14_neutral & (close > vwap * 0.99) & vol_ok & vix_ok & time_ok
        short_entry = short_entry & rsi14_neutral & (close < vwap * 1.01) & vol_ok & vix_ok & time_ok

        # ── Signal exit: RSI(2) crosses 60/40 or new extreme ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if rsi2[i] >= 60 and rsi2[i - 1] < 60:
                sig_exit_long[i] = True
            if rsi2[i] < 3:
                sig_exit_long[i] = True
            if rsi2[i] <= 40 and rsi2[i - 1] > 40:
                sig_exit_short[i] = True
            if rsi2[i] > 97:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=15,
        )
