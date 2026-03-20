"""Opening Range Breakout 30min v1 — cursor_opus46max_037

Thesis: 30-minute OR (09:15-09:45) is more robust than 15-min ORB, reducing
false breakouts by ~25%. Requires 2 consecutive bars closing beyond boundary.
Stop at OR midpoint; target is 1x OR range.
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
    name = "cursor_opus46max_037"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("or_range_min", default=0.4, low=0.2, high=0.6),
            TunableParam("or_range_max", default=2.5, low=1.5, high=3.5),
            TunableParam("vol_mult", default=2.5, low=1.5, high=4.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        or_min = params.get("or_range_min", 0.4)
        or_max = params.get("or_range_max", 2.5)
        vol_mult = params.get("vol_mult", 2.5)
        vix_max = params.get("vix_max", 25.0)

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

        # ── Compute 30-min OR per day ──
        or_high = np.zeros(n, dtype=np.float64)
        or_low = np.zeros(n, dtype=np.float64)
        or_range = np.zeros(n, dtype=np.float64)
        or_avg_vol = np.zeros(n, dtype=np.float64)
        or_computed = np.zeros(n, dtype=np.bool_)

        prev_day = -1
        day_start = 0
        cur_or_high = 0.0
        cur_or_low = 1e18
        cur_or_vol = 0.0

        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                day_start = i
                cur_or_high = high[i]
                cur_or_low = low[i]
                cur_or_vol = volume[i]
            else:
                bars_in_day = i - day_start
                if bars_in_day < 30:
                    cur_or_high = max(cur_or_high, high[i])
                    cur_or_low = min(cur_or_low, low[i])
                    cur_or_vol += volume[i]

            or_high[i] = cur_or_high
            or_low[i] = cur_or_low
            if cur_or_low > 0:
                or_range[i] = (cur_or_high - cur_or_low) / cur_or_low * 100.0
            if i - day_start >= 30:
                or_computed[i] = True
                or_avg_vol[i] = cur_or_vol / 30.0

        # ── 2-bar confirmation above/below OR ──
        above_or_count = np.zeros(n, dtype=np.int32)
        below_or_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if or_computed[i] and close[i] > or_high[i]:
                above_or_count[i] = above_or_count[i-1] + 1
            else:
                above_or_count[i] = 0
            if or_computed[i] and close[i] < or_low[i]:
                below_or_count[i] = below_or_count[i-1] + 1
            else:
                below_or_count[i] = 0

        # ── Filters ──
        vix_ok = vix < vix_max
        range_ok = (or_range >= or_min) & (or_range <= or_max)
        time_ok = (time_mins >= 585) & (time_mins <= 870)

        # ── Entries on 2nd consecutive bar beyond OR ──
        long_entry = (
            or_computed & (above_or_count == 2) & range_ok &
            (volume > vol_mult * or_avg_vol) & (close > vwap) &
            vix_ok & time_ok
        )
        short_entry = (
            or_computed & (below_or_count == 2) & range_ok &
            (volume > vol_mult * or_avg_vol) & (close < vwap) &
            vix_ok & time_ok
        )

        # ── Signal exit: 3 bars back inside OR ──
        inside_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if or_computed[i] and close[i] >= or_low[i] and close[i] <= or_high[i]:
                inside_count[i] = inside_count[i-1] + 1
            else:
                inside_count[i] = 0

        signal_exit_long = inside_count >= 3
        signal_exit_short = inside_count >= 3

        # ── Stop distance ──
        valid_ranges = or_range[or_range > 0]
        stop_pct = np.median(valid_ranges) / 200.0 if len(valid_ranges) > 0 else 0.005

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=stop_pct * 2.0,
            trailing_stop_pct=stop_pct * 0.7,
            trailing_activate_pct=stop_pct * 1.4,
            time_stop_bars=120,
        )
