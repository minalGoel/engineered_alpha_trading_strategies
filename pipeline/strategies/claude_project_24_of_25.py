"""Gap Continuation — claude_project_24_of_25

Thesis: Large opening gaps (>1.5%) that survive the first 30 minutes
without filling signal strong institutional conviction. When price
breaks above the first-30-min high, the gap is likely to extend.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "gap_continuation_v1"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh_pct", default=0.015, low=0.008, high=0.03),
            TunableParam("breakeven_activation_pct", default=0.5, low=0.3, high=0.7),
            TunableParam("vix_max", default=28.0, low=20.0, high=35.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_thresh = params.get("gap_thresh_pct", 0.015)
        breakeven_act = params.get("breakeven_activation_pct", 0.5)
        vix_max = params.get("vix_max", 28.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        vix = df["vix"].to_numpy().astype(np.float64)

        vix = np.nan_to_num(vix, nan=99.0)

        # ── Per-day gap and first-30-bar range ──
        gap_pct = np.zeros(n, dtype=np.float64)
        first_30_high = np.zeros(n, dtype=np.float64)
        first_30_low = np.zeros(n, dtype=np.float64)
        first_30_mid = np.zeros(n, dtype=np.float64)
        gap_filled = np.ones(n, dtype=np.bool_)  # default: gap filled (no trade)

        unique_days = np.unique(day_ids)
        prev_close_val = np.nan

        for d in unique_days:
            day_mask = day_ids == d
            day_indices = np.where(day_mask)[0]
            if len(day_indices) == 0:
                continue

            day_times = time_mins[day_indices]
            day_open = open_[day_indices[0]]

            # Compute gap
            if not np.isnan(prev_close_val) and prev_close_val > 0:
                g = (day_open - prev_close_val) / prev_close_val
                gap_pct[day_indices] = g
            prev_close_val = close[day_indices[-1]]

            # First 30 bars (~09:15 to 09:44, time_mins 555 to 584)
            first_30_mask = day_times < 585  # before 09:45
            if np.any(first_30_mask):
                f30_idx = day_indices[first_30_mask]
                f30_h = np.max(high[f30_idx])
                f30_l = np.min(low[f30_idx])
                first_30_high[day_indices] = f30_h
                first_30_low[day_indices] = f30_l
                first_30_mid[day_indices] = (f30_h + f30_l) / 2.0

                # Check if gap filled during first 30 bars
                g = gap_pct[day_indices[0]]
                if g > 0:
                    # Up gap: filled if low dipped to prev close
                    if not np.isnan(prev_close_val):
                        filled = f30_l <= (day_open / (1.0 + g))
                    else:
                        filled = True
                elif g < 0:
                    # Down gap: filled if high reached prev close
                    if not np.isnan(prev_close_val):
                        filled = f30_h >= (day_open / (1.0 + g))
                    else:
                        filled = True
                else:
                    filled = True

                gap_filled[day_indices] = filled
            else:
                # No bars before 09:45 — use first available
                first_30_high[day_indices] = high[day_indices[0]]
                first_30_low[day_indices] = low[day_indices[0]]
                first_30_mid[day_indices] = (high[day_indices[0]] + low[day_indices[0]]) / 2.0

        # ── Entry window: 09:45-11:00 (585-660) ──
        entry_time_ok = (time_mins >= 585) & (time_mins <= 660)

        # ── VIX filter ──
        vix_ok = vix < vix_max

        # ── Entry: unfilled gap + breakout ──
        long_entry = (
            (gap_pct > gap_thresh)
            & (~gap_filled)
            & (close > first_30_high)
            & entry_time_ok
            & vix_ok
        )
        short_entry = (
            (gap_pct < -gap_thresh)
            & (~gap_filled)
            & (close < first_30_low)
            & entry_time_ok
            & vix_ok
        )

        # ── Target: gap_pct as extension target ──
        target_pct_val = float(np.median(np.abs(gap_pct[np.abs(gap_pct) > gap_thresh]))) \
            if np.any(np.abs(gap_pct) > gap_thresh) else gap_thresh

        # ── Stop: first_30_mid → use as pct ──
        # Approximate stop as half the first-30 range pct
        median_close = np.nanmedian(close[close > 0]) if np.any(close > 0) else 1.0
        median_f30_range = np.nanmedian(first_30_high - first_30_low)
        stop_pct = (median_f30_range / (2.0 * median_close)) if median_close > 0 else 0.005
        stop_pct = max(stop_pct, 0.002)

        # ── Breakeven after 50% of target ──
        breakeven_pct = target_pct_val * breakeven_act

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct_val,
            stop_loss_pct=stop_pct,
            breakeven_pct=breakeven_pct,
            time_stop_bars=300,
        )
