"""Pivot Point Breakout v1 — cursor_opus46max_042

Thesis: Classic pivot points (P, R1, R2, S1, S2) from previous day OHLC
are widely-watched Schelling points. Break through R1/S1 with volume
confirms move beyond yesterday's equilibrium. Target is R2/S2; stop at
Pivot (P).
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
    name = "cursor_opus46max_042"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("confirm_bars", default=2.0, low=1.0, high=3.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vol_mult = params.get("vol_mult", 1.5)
        confirm = int(params.get("confirm_bars", 2.0))
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
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

        # ── Compute pivot levels per day from previous day ──
        pivot = np.zeros(n, dtype=np.float64)
        r1 = np.zeros(n, dtype=np.float64)
        r2 = np.zeros(n, dtype=np.float64)
        s1 = np.zeros(n, dtype=np.float64)
        s2 = np.zeros(n, dtype=np.float64)

        # Get previous day OHLC
        unique_days = []
        day_data = {}
        prev_day = -1
        for i in range(n):
            d = day_id[i]
            if d != prev_day:
                unique_days.append(d)
                day_data[d] = {"high": high[i], "low": low[i], "close": close[i]}
                prev_day = d
            else:
                day_data[d]["high"] = max(day_data[d]["high"], high[i])
                day_data[d]["low"] = min(day_data[d]["low"], low[i])
                day_data[d]["close"] = close[i]

        day_to_idx = {d: idx for idx, d in enumerate(unique_days)}

        for i in range(n):
            d = day_id[i]
            didx = day_to_idx.get(d, 0)
            if didx > 0:
                prev_d = unique_days[didx - 1]
                pd = day_data[prev_d]
                p = (pd["high"] + pd["low"] + pd["close"]) / 3.0
                pivot[i] = p
                r1[i] = 2.0 * p - pd["low"]
                r2[i] = p + (pd["high"] - pd["low"])
                s1[i] = 2.0 * p - pd["high"]
                s2[i] = p - (pd["high"] - pd["low"])

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Consecutive bars above R1 / below S1 ──
        above_r1_count = np.zeros(n, dtype=np.int32)
        below_s1_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if r1[i] > 0 and close[i] > r1[i] and close[i] > opn[i]:
                above_r1_count[i] = above_r1_count[i-1] + 1
            else:
                above_r1_count[i] = 0
            if s1[i] > 0 and close[i] < s1[i] and close[i] < opn[i]:
                below_s1_count[i] = below_s1_count[i-1] + 1
            else:
                below_s1_count[i] = 0

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            (above_r1_count == confirm) & (close > vwap) &
            vol_ok & time_ok & (r1 > 0)
        )
        short_entry = (
            (below_s1_count == confirm) & (close < vwap) &
            vol_ok & time_ok & (s1 > 0)
        )

        # ── Signal exit: close back below R1 / above S1 for 2 bars ──
        back_below_r1 = np.zeros(n, dtype=np.int32)
        back_above_s1 = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if r1[i] > 0 and close[i] < r1[i]:
                back_below_r1[i] = back_below_r1[i-1] + 1
            else:
                back_below_r1[i] = 0
            if s1[i] > 0 and close[i] > s1[i]:
                back_above_s1[i] = back_above_s1[i-1] + 1
            else:
                back_above_s1[i] = 0

        signal_exit_long = back_below_r1 >= 2
        signal_exit_short = back_above_s1 >= 2

        # ── Target: R2 for longs, S2 for shorts (use pct approximation) ──
        # Approximate target as distance from R1 to R2
        valid_r = r2[r2 > 0] - r1[r1 > 0][:len(r2[r2 > 0])] if np.any(r1 > 0) else np.array([0.0])
        valid_c = close[close > 0]
        if len(valid_r) > 0 and len(valid_c) > 0:
            tgt_pct = np.median(np.abs(valid_r)) / np.median(valid_c)
        else:
            tgt_pct = 0.005

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=tgt_pct,
            time_stop_bars=60,
        )
