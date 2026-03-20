"""Volume Momentum v1 — cursor_opus46max_028

Thesis: Volume-weighted momentum measures how much volume participated in
a move. When ROC(10) is positive AND cumulative volume on up-bars exceeds
down-bars by >60:40 over 20 bars, momentum has institutional backing.
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
    name = "cursor_opus46max_028"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_ratio_long", default=0.60, low=0.55, high=0.70),
            TunableParam("vol_ratio_short", default=0.40, low=0.30, high=0.45),
            TunableParam("roc_thresh", default=0.15, low=0.05, high=0.30),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vr_long = params.get("vol_ratio_long", 0.60)
        vr_short = params.get("vol_ratio_short", 0.40)
        roc_thresh = params.get("roc_thresh", 0.15)
        stop_pct = params.get("stop_loss_pct", 0.002)
        tgt_pct = params.get("target_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
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

        # ── Up/down volume ratio over 20 bars ──
        is_up = close > opn
        up_vol = np.where(is_up, volume, 0.0)
        dn_vol = np.where(~is_up, volume, 0.0)

        up_vol_20 = np.zeros(n, dtype=np.float64)
        dn_vol_20 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - 19)
            up_vol_20[i] = np.sum(up_vol[start:i+1])
            dn_vol_20[i] = np.sum(dn_vol[start:i+1])

        total_vol_20 = up_vol_20 + dn_vol_20
        total_vol_20 = np.where(total_vol_20 > 0, total_vol_20, 1.0)
        vol_ratio = up_vol_20 / total_vol_20

        # ── ROC(10) ──
        roc10 = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            if close[i-10] > 0:
                roc10[i] = (close[i] - close[i-10]) / close[i-10] * 100.0

        # ── Volume ratio confirmation: increasing ──
        vr_increasing = np.zeros(n, dtype=np.bool_)
        vr_decreasing = np.zeros(n, dtype=np.bool_)
        for i in range(5, n):
            vr_increasing[i] = vol_ratio[i] > vol_ratio[i-5]
            vr_decreasing[i] = vol_ratio[i] < vol_ratio[i-5]

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── Time filter ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            (vol_ratio > vr_long) & (roc10 > roc_thresh) &
            (close > vwap) & vol_ok & vr_increasing & time_ok
        )
        short_entry = (
            (vol_ratio < vr_short) & (roc10 < -roc_thresh) &
            (close < vwap) & vol_ok & vr_decreasing & time_ok
        )

        # ── Signal exits: vol_ratio neutralized ──
        signal_exit_long = (vol_ratio >= 0.45) & (vol_ratio <= 0.55)
        signal_exit_short = (vol_ratio >= 0.45) & (vol_ratio <= 0.55)

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
            trailing_activate_pct=0.0025,
            time_stop_bars=40,
        )
