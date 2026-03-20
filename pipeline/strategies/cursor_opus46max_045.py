"""VWAP Breakout v1 — cursor_opus46max_045

Thesis: Decisive VWAP break after 3+ tests is high-conviction. Multiple
tests weaken opposing limit orders (battering ram). Entry on 2nd
consecutive bar closing beyond VWAP after 3+ touches. Stop at VWAP.
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


class Strategy(BaseStrategy):
    name = "cursor_opus46max_045"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("touch_count_min", default=3.0, low=2.0, high=5.0),
            TunableParam("touch_pct", default=0.05, low=0.02, high=0.10),
            TunableParam("break_pct", default=0.1, low=0.05, high=0.2),
            TunableParam("vol_mult", default=2.0, low=1.2, high=3.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        touch_min = int(params.get("touch_count_min", 3.0))
        touch_pct = params.get("touch_pct", 0.05) / 100.0
        break_pct = params.get("break_pct", 0.1) / 100.0
        vol_mult = params.get("vol_mult", 2.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.003)
        tgt_pct = params.get("target_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

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

        # ── VWAP touch count per session ──
        touch_count = np.zeros(n, dtype=np.int32)
        prev_day = -1
        running_touches = 0
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                running_touches = 0
            if vwap[i] > 0 and abs(close[i] - vwap[i]) / vwap[i] < touch_pct:
                running_touches += 1
            touch_count[i] = running_touches

        # ── Side before break: mode of sign(close - VWAP) over last 20 bars ──
        side_before = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            signs = np.sign(close[i-20:i] - vwap[i-20:i])
            side_before[i] = np.sign(np.sum(signs))

        # ── Consecutive bars above/below VWAP with clear break ──
        above_vwap_count = np.zeros(n, dtype=np.int32)
        below_vwap_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if vwap[i] > 0 and close[i] > vwap[i] * (1.0 + break_pct):
                above_vwap_count[i] = above_vwap_count[i-1] + 1
            else:
                above_vwap_count[i] = 0
            if vwap[i] > 0 and close[i] < vwap[i] * (1.0 - break_pct):
                below_vwap_count[i] = below_vwap_count[i-1] + 1
            else:
                below_vwap_count[i] = 0

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Filters ──
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 600) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            (above_vwap_count == 2) & (touch_count >= touch_min) &
            (side_before < 0) & vol_ok & vix_ok & time_ok
        )
        short_entry = (
            (below_vwap_count == 2) & (touch_count >= touch_min) &
            (side_before > 0) & vol_ok & vix_ok & time_ok
        )

        # ── Signal exit: close near VWAP for 2 bars ──
        near_vwap_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if vwap[i] > 0 and abs(close[i] - vwap[i]) / vwap[i] < touch_pct:
                near_vwap_count[i] = near_vwap_count[i-1] + 1
            else:
                near_vwap_count[i] = 0

        signal_exit_long = near_vwap_count >= 2
        signal_exit_short = near_vwap_count >= 2

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=tgt_pct,
            trailing_stop_pct=0.002,
            trailing_activate_pct=0.003,
            time_stop_bars=60,
        )
