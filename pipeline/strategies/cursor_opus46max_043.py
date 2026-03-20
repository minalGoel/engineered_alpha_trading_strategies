"""Previous Day High/Low Breakout v1 — cursor_opus46max_043

Thesis: PDH/PDL are key psychological levels. Breaking above PDH triggers
short covering + breakout buying + delta-hedging. Volume > 2x and break
before 12:00 have significantly higher success rate. 3-bar confirmation.
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
    name = "cursor_opus46max_043"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 840     # 14:00
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=2.0, low=1.2, high=3.0),
            TunableParam("confirm_bars", default=3.0, low=2.0, high=5.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_pct", default=0.007, low=0.004, high=0.010),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 2.0)
        confirm = int(params.get("confirm_bars", 3.0))
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.003)
        tgt_pct = params.get("target_pct", 0.007)

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

        # ── Previous day high/low ──
        pdh = np.zeros(n, dtype=np.float64)
        pdl = np.zeros(n, dtype=np.float64)

        unique_days = []
        day_data = {}
        prev_day = -1
        for i in range(n):
            d = day_id[i]
            if d != prev_day:
                unique_days.append(d)
                day_data[d] = {"high": high[i], "low": low[i]}
                prev_day = d
            else:
                day_data[d]["high"] = max(day_data[d]["high"], high[i])
                day_data[d]["low"] = min(day_data[d]["low"], low[i])

        day_to_idx = {d: idx for idx, d in enumerate(unique_days)}

        for i in range(n):
            d = day_id[i]
            didx = day_to_idx.get(d, 0)
            if didx > 0:
                prev_d = unique_days[didx - 1]
                pdh[i] = day_data[prev_d]["high"]
                pdl[i] = day_data[prev_d]["low"]

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Consecutive bars above PDH / below PDL ──
        above_pdh_count = np.zeros(n, dtype=np.int32)
        below_pdl_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if pdh[i] > 0 and close[i] > pdh[i]:
                above_pdh_count[i] = above_pdh_count[i-1] + 1
            else:
                above_pdh_count[i] = 0
            if pdl[i] > 0 and close[i] < pdl[i]:
                below_pdl_count[i] = below_pdl_count[i-1] + 1
            else:
                below_pdl_count[i] = 0

        # ── Filters ──
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 570) & (time_mins <= 840)

        # ── Entries on confirm-th bar ──
        long_entry = (
            (above_pdh_count == confirm) & (close > vwap) &
            vol_ok & vix_ok & time_ok & (pdh > 0)
        )
        short_entry = (
            (below_pdl_count == confirm) & (close < vwap) &
            vol_ok & vix_ok & time_ok & (pdl > 0)
        )

        # ── Signal exit: close back below PDH / above PDL for 2 bars ──
        back_below = np.zeros(n, dtype=np.int32)
        back_above = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if pdh[i] > 0 and close[i] < pdh[i]:
                back_below[i] = back_below[i-1] + 1
            else:
                back_below[i] = 0
            if pdl[i] > 0 and close[i] > pdl[i]:
                back_above[i] = back_above[i-1] + 1
            else:
                back_above[i] = 0

        signal_exit_long = back_below >= 2
        signal_exit_short = back_above >= 2

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=tgt_pct,
            trailing_stop_pct=0.0025,
            trailing_activate_pct=0.004,
            time_stop_bars=90,
        )
