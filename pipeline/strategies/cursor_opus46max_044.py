"""Consolidation Breakout v1 — cursor_opus46max_044

Thesis: Mid-session consolidation (range < 0.2% for 20+ bars) AFTER an
initial directional move represents institutional order absorption. Breakout
in continuation direction (same as prior move) tends to follow through.
Stop at opposite side of consolidation range.
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
    name = "cursor_opus46max_044"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 840     # 14:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("consol_range_max", default=0.2, low=0.1, high=0.4),
            TunableParam("consol_min_bars", default=20.0, low=10.0, high=30.0),
            TunableParam("vol_mult", default=2.0, low=1.2, high=3.0),
            TunableParam("prior_move_min", default=0.5, low=0.3, high=0.8),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        consol_max = params.get("consol_range_max", 0.2)
        consol_min = int(params.get("consol_min_bars", 20.0))
        vol_mult = params.get("vol_mult", 2.0)
        prior_move_min = params.get("prior_move_min", 0.5)
        stop_pct = params.get("stop_loss_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Rolling consolidation detection ──
        consol_high = np.zeros(n, dtype=np.float64)
        consol_low = np.zeros(n, dtype=np.float64)
        consol_range = np.full(n, 999.0, dtype=np.float64)
        consol_duration = np.zeros(n, dtype=np.int32)

        for i in range(consol_min - 1, n):
            h_win = np.max(high[i - consol_min + 1:i + 1])
            l_win = np.min(low[i - consol_min + 1:i + 1])
            consol_high[i] = h_win
            consol_low[i] = l_win
            if l_win > 0:
                consol_range[i] = (h_win - l_win) / l_win * 100.0

        # Count consecutive bars within tight range
        for i in range(1, n):
            if consol_range[i] < consol_max:
                consol_duration[i] = consol_duration[i-1] + 1
            else:
                consol_duration[i] = 0

        # ── Prior move direction (from day open to consolidation zone) ──
        prior_move_dir = np.zeros(n, dtype=np.float64)
        prev_day = -1
        day_open = 0.0
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                day_open = opn[i]
            if day_open > 0:
                prior_move_dir[i] = (consol_high[i] - day_open) / day_open * 100.0

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Breakout detection ──
        breakout_long = np.zeros(n, dtype=np.bool_)
        breakout_short = np.zeros(n, dtype=np.bool_)
        for i in range(consol_min, n):
            if consol_duration[i-1] >= consol_min:
                ch = consol_high[i-1]
                cl = consol_low[i-1]
                if close[i] > ch and (close[i] - ch) / ch * 100.0 > 0.05:
                    breakout_long[i] = True
                if close[i] < cl and (cl - close[i]) / cl * 100.0 > 0.05:
                    breakout_short[i] = True

        # ── Filters ──
        time_ok = (time_mins >= 600) & (time_mins <= 840)

        # ── Entries (continuation only) ──
        long_entry = (
            breakout_long & (prior_move_dir > prior_move_min) &
            vol_ok & time_ok
        )
        short_entry = (
            breakout_short & (prior_move_dir < -prior_move_min) &
            vol_ok & time_ok
        )

        # ── Signal exit: close back within consolidation for 3 bars ──
        inside_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if (consol_high[i] > 0 and close[i] >= consol_low[i] and
                close[i] <= consol_high[i]):
                inside_count[i] = inside_count[i-1] + 1
            else:
                inside_count[i] = 0

        signal_exit_long = inside_count >= 3
        signal_exit_short = inside_count >= 3

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=stop_pct * 2.5,
            time_stop_bars=45,
        )
