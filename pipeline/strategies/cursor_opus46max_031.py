"""Gap Continuation Momentum v1 — cursor_opus46max_031

Thesis: Stocks gapping >1.5% at open that continue higher in the first
15 minutes exhibit strong continuation momentum. The 15-min confirmation
filters out dead-cat-bounce gaps. Entry after 09:30 when gap continuation
is confirmed with volume.
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


def _compute_ema(data, period):
    n = len(data)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = data[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = data[i] * k + ema[i-1] * (1 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_031"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 840     # 14:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min_pct", default=1.5, low=0.8, high=2.5),
            TunableParam("gap_max_pct", default=5.0, low=3.0, high=7.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("target_pct", default=0.008, low=0.005, high=0.012),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_min = params.get("gap_min_pct", 1.5)
        gap_max = params.get("gap_max_pct", 5.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        tgt_pct = params.get("target_pct", 0.008)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
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

        # ── EMA(20) for signal exit ──
        ema20 = _compute_ema(close, 20)

        # ── Compute gap and first-15-bar metrics per day ──
        gap_pct = np.zeros(n, dtype=np.float64)
        first_15_return = np.zeros(n, dtype=np.float64)
        first_5_high = np.zeros(n, dtype=np.float64)
        first_15_confirmed = np.zeros(n, dtype=np.bool_)
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        prev_day_close = 0.0
        prev_day = -1
        day_start_idx = 0
        day_open = 0.0

        for i in range(n):
            if day_id[i] != prev_day:
                if prev_day != -1:
                    prev_day_close = close[i-1]
                day_start_idx = i
                day_open = opn[i]
                prev_day = day_id[i]

            bars_from_open = i - day_start_idx

            if prev_day_close > 0 and day_open > 0:
                gap_pct[i] = (day_open - prev_day_close) / prev_day_close * 100.0

            # Track first 5 bars high
            if bars_from_open < 5:
                if bars_from_open == 0:
                    first_5_high[i] = high[i]
                else:
                    first_5_high[i] = max(first_5_high[i-1], high[i])
            elif i > 0:
                first_5_high[i] = first_5_high[i-1]

            # After 15 bars, check confirmation
            if bars_from_open == 15 and day_open > 0:
                first_15_return[i] = (close[i] - day_open) / day_open * 100.0
                # Compute volume surge in first 15 bars
                vol_sum = np.sum(volume[day_start_idx:i+1])
                avg_bar_vol = vol_sum / 15.0
                if avg_vol[i] > 0:
                    vol_surge = avg_bar_vol / avg_vol[i]
                else:
                    vol_surge = 0.0
                # Check all conditions
                gap_ok = (abs(gap_pct[i]) >= gap_min) and (abs(gap_pct[i]) <= gap_max)
                if gap_pct[i] > 0:
                    first_15_confirmed[i] = (
                        gap_ok and first_15_return[i] > 0 and
                        vol_surge > 2.0 and close[i] > vwap[i] and
                        close[i] > first_5_high[i]
                    )
                elif gap_pct[i] < 0:
                    first_15_confirmed[i] = (
                        gap_ok and first_15_return[i] < 0 and
                        vol_surge > 2.0 and close[i] < vwap[i]
                    )

            # Propagate gap info
            if bars_from_open > 15 and i > 0:
                first_15_return[i] = first_15_return[i-1]
                first_15_confirmed[i] = first_15_confirmed[i-1]

        # ── Entries: only on bar 15 (09:30) ──
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 570) & (time_mins <= 840)

        long_entry = first_15_confirmed & (gap_pct > 0) & vix_ok & time_ok
        short_entry = first_15_confirmed & (gap_pct < 0) & vix_ok & time_ok

        # Only fire on bar 15 of each day
        for i in range(n):
            bars_from_start = i - day_start_idx if i >= day_start_idx else 0
            if day_id[i] != day_id[day_start_idx] if i > 0 else True:
                for j in range(i, n):
                    if day_id[j] == day_id[i]:
                        day_start_idx = j
                        break
                bars_from_start = i - day_start_idx

        # ── Signal exit: close below EMA(20) for longs ──
        below_ema_count = np.zeros(n, dtype=np.int32)
        above_ema_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if close[i] < ema20[i]:
                below_ema_count[i] = below_ema_count[i-1] + 1
            else:
                below_ema_count[i] = 0
            if close[i] > ema20[i]:
                above_ema_count[i] = above_ema_count[i-1] + 1
            else:
                above_ema_count[i] = 0

        signal_exit_long = below_ema_count >= 3
        signal_exit_short = above_ema_count >= 3

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
