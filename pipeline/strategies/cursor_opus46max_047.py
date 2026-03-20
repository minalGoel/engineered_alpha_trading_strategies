"""Premarket Range Breakout v1 — cursor_opus46max_047

Thesis: The pre-market session (09:00-09:08 IST call auction) establishes an
indicative price range. Breaking out of this range within the first 30 minutes
with volume and VIX confirmation signals surprise information flow. We proxy
the pre-market range using the first 15 bars (09:15-09:30) high/low since
actual call auction data is not available in standard 1-min OHLCV feeds.
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
    name = "cursor_opus46max_047"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 840     # 14:00
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("pm_bars", default=15.0, low=10.0, high=20.0),
            TunableParam("confirm_bars", default=3.0, low=2.0, high=5.0),
            TunableParam("vol_mult", default=2.0, low=1.2, high=3.0),
            TunableParam("vix_max", default=22.0, low=18.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("target_pct", default=0.008, low=0.005, high=0.012),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        pm_bars = int(params.get("pm_bars", 15.0))
        confirm = int(params.get("confirm_bars", 3.0))
        vol_mult = params.get("vol_mult", 2.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.004)
        tgt_pct = params.get("target_pct", 0.008)

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

        # ── Pre-market range proxy: first pm_bars bars of each day ──
        pm_high = np.zeros(n, dtype=np.float64)
        pm_low = np.zeros(n, dtype=np.float64)
        pm_mid = np.zeros(n, dtype=np.float64)

        prev_day = -1
        bar_in_day = 0
        cur_pm_high = 0.0
        cur_pm_low = 0.0
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                bar_in_day = 0
                cur_pm_high = high[i]
                cur_pm_low = low[i]
            else:
                bar_in_day += 1

            if bar_in_day < pm_bars:
                cur_pm_high = max(cur_pm_high, high[i])
                cur_pm_low = min(cur_pm_low, low[i])

            pm_high[i] = cur_pm_high
            pm_low[i] = cur_pm_low
            if cur_pm_high > 0 and cur_pm_low > 0:
                pm_mid[i] = (cur_pm_high + cur_pm_low) / 2.0

        # ── Track bar count in day for entry window ──
        bar_count = np.zeros(n, dtype=np.int32)
        prev_day = -1
        cnt = 0
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                cnt = 0
            else:
                cnt += 1
            bar_count[i] = cnt

        # ── Consecutive bars above PM high / below PM low ──
        above_pm_count = np.zeros(n, dtype=np.int32)
        below_pm_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if pm_high[i] > 0 and close[i] > pm_high[i] and bar_count[i] >= pm_bars:
                above_pm_count[i] = above_pm_count[i-1] + 1
            else:
                above_pm_count[i] = 0
            if pm_low[i] > 0 and close[i] < pm_low[i] and bar_count[i] >= pm_bars:
                below_pm_count[i] = below_pm_count[i-1] + 1
            else:
                below_pm_count[i] = 0

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Filters ──
        vix_ok = vix < vix_max
        # Entry only within first 30 bars of session (pm_bars to pm_bars+30)
        entry_window = (bar_count >= pm_bars) & (bar_count <= pm_bars + 30)
        time_ok = (time_mins >= 570) & (time_mins <= 840)

        # ── Entries ──
        long_entry = (
            (above_pm_count == confirm) & (close > vwap) &
            vol_ok & vix_ok & entry_window & time_ok & (pm_high > 0)
        )
        short_entry = (
            (below_pm_count == confirm) & (close < vwap) &
            vol_ok & vix_ok & entry_window & time_ok & (pm_low > 0)
        )

        # ── Signal exit: close back within PM range for 3 bars ──
        inside_pm_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if pm_high[i] > 0 and close[i] >= pm_low[i] and close[i] <= pm_high[i]:
                inside_pm_count[i] = inside_pm_count[i-1] + 1
            else:
                inside_pm_count[i] = 0

        signal_exit_long = inside_pm_count >= 3
        signal_exit_short = inside_pm_count >= 3

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=tgt_pct,
            trailing_stop_pct=0.003,
            trailing_activate_pct=0.005,
            time_stop_bars=90,
        )
