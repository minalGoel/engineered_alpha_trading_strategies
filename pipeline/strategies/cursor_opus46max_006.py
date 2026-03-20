"""VWAP Anchored Support v1 — cursor_opus46max_006

Thesis: Anchored VWAP from prior significant events acts as persistent
institutional memory. When price approaches and bounces off this level,
it signals support/resistance. We approximate using the previous day's
final VWAP as the anchor.
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
    name = "cursor_opus46max_006"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("dist_thresh", default=0.15, low=0.05, high=0.3),
            TunableParam("vol_ratio_thresh", default=1.3, low=0.8, high=2.0),
            TunableParam("vix_max", default=25.0, low=16.0, high=30.0),
            TunableParam("target_pct", default=0.005, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.005),
            TunableParam("trailing_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("trailing_activate", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        dist_thresh = params.get("dist_thresh", 0.15)
        vol_thresh = params.get("vol_ratio_thresh", 1.3)
        vix_max = params.get("vix_max", 25.0)
        tgt_pct = params.get("target_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        trail_pct = params.get("trailing_pct", 0.002)
        trail_act = params.get("trailing_activate", 0.003)

        n = len(df)

        # ── Session VWAP ──
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
        day_id = df["day_id"].to_numpy()

        # ── Anchored VWAP: use prev day's final VWAP as anchor level ──
        anchored_vwap = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_id)
        prev_day_vwap = 0.0
        for d in unique_days:
            mask = day_id == d
            idx = np.where(mask)[0]
            anchored_vwap[idx] = prev_day_vwap if prev_day_vwap > 0 else vwap[idx[0]]
            prev_day_vwap = vwap[idx[-1]]

        # ── Distance to anchored VWAP ──
        safe_anchor = np.where(anchored_vwap > 0, anchored_vwap, 1.0)
        dist_pct = (close - anchored_vwap) / safe_anchor * 100.0

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        # ── Volume ratio ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_thresh * avg_vol

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Crossover of anchored VWAP ──
        cross_above = np.zeros(n, dtype=np.bool_)
        cross_below = np.zeros(n, dtype=np.bool_)
        consec_above = np.zeros(n, dtype=np.int32)
        consec_below = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if close[i] > anchored_vwap[i]:
                consec_above[i] = consec_above[i - 1] + 1
                consec_below[i] = 0
            elif close[i] < anchored_vwap[i]:
                consec_below[i] = consec_below[i - 1] + 1
                consec_above[i] = 0
            if close[i] > anchored_vwap[i] and close[i - 1] <= anchored_vwap[i - 1]:
                cross_above[i] = True
            if close[i] < anchored_vwap[i] and close[i - 1] >= anchored_vwap[i - 1]:
                cross_below[i] = True

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 900)
        vix_ok = vix < vix_max
        rsi_neutral = (rsi > 40) & (rsi < 60)

        # ── Entries: bounce off anchored VWAP with 2 consec bars ──
        long_entry = (
            (np.abs(dist_pct) < dist_thresh)
            & (close > anchored_vwap)
            & (consec_above >= 2)
            & vol_ok & rsi_neutral & vix_ok & time_ok
        )
        short_entry = (
            (np.abs(dist_pct) < dist_thresh)
            & (close < anchored_vwap)
            & (consec_below >= 2)
            & vol_ok & rsi_neutral & vix_ok & time_ok
        )

        # ── Signal exit: re-cross anchored VWAP ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if close[i] < anchored_vwap[i] and close[i - 1] >= anchored_vwap[i - 1]:
                sig_exit_long[i] = True
            if close[i] > anchored_vwap[i] and close[i - 1] <= anchored_vwap[i - 1]:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=90,
        )
