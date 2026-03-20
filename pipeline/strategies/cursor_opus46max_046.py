"""Fibonacci Breakout v1 — cursor_opus46max_046

Thesis: After an initial intraday move (>0.5%), price retraces to a
Fibonacci level (38.2%, 50%, 61.8%) before resuming. Bounce off a Fib
retracement level + break above prior swing high (or below swing low)
confirmed by volume. Target at 161.8% Fibonacci extension.
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
    name = "cursor_opus46max_046"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("pivot_lookback", default=5.0, low=3.0, high=7.0),
            TunableParam("min_swing_pct", default=0.5, low=0.3, high=0.8),
            TunableParam("fib_tolerance", default=0.1, low=0.05, high=0.2),
            TunableParam("vol_mult", default=1.0, low=0.8, high=2.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("target_pct", default=0.006, low=0.004, high=0.010),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        pivot_lb = int(params.get("pivot_lookback", 5.0))
        min_swing = params.get("min_swing_pct", 0.5) / 100.0
        fib_tol = params.get("fib_tolerance", 0.1) / 100.0
        vol_mult = params.get("vol_mult", 1.0)
        stop_pct = params.get("stop_loss_pct", 0.004)
        tgt_pct = params.get("target_pct", 0.006)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
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

        # ── Swing high/low detection (5-bar pivot) ──
        # swing_high[i] = True if high[i] is highest in [i-pivot_lb, i+pivot_lb]
        is_swing_high = np.zeros(n, dtype=np.bool_)
        is_swing_low = np.zeros(n, dtype=np.bool_)
        for i in range(pivot_lb, n - pivot_lb):
            left_h = high[i - pivot_lb:i]
            right_h = high[i + 1:i + pivot_lb + 1]
            if high[i] > np.max(left_h) and high[i] > np.max(right_h):
                is_swing_high[i] = True
            left_l = low[i - pivot_lb:i]
            right_l = low[i + 1:i + pivot_lb + 1]
            if low[i] < np.min(left_l) and low[i] < np.min(right_l):
                is_swing_low[i] = True

        # ── Track most recent swing high and swing low per day ──
        # For long: need prior up-swing (swing_low then swing_high), retracement to fib, break above swing_high
        # For short: need prior down-swing (swing_high then swing_low), retracement to fib, break below swing_low
        last_sh_val = np.zeros(n, dtype=np.float64)  # last swing high value
        last_sl_val = np.zeros(n, dtype=np.float64)  # last swing low value
        last_sh_idx = np.full(n, -1, dtype=np.int64)
        last_sl_idx = np.full(n, -1, dtype=np.int64)

        prev_day = -1
        cur_sh_val = 0.0
        cur_sl_val = 0.0
        cur_sh_idx = -1
        cur_sl_idx = -1
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                cur_sh_val = 0.0
                cur_sl_val = 0.0
                cur_sh_idx = -1
                cur_sl_idx = -1
            # We can only confirm a swing at i - pivot_lb (need right side confirmed)
            confirm_i = i - pivot_lb
            if confirm_i >= 0 and day_id[confirm_i] == day_id[i]:
                if is_swing_high[confirm_i]:
                    cur_sh_val = high[confirm_i]
                    cur_sh_idx = confirm_i
                if is_swing_low[confirm_i]:
                    cur_sl_val = low[confirm_i]
                    cur_sl_idx = confirm_i
            last_sh_val[i] = cur_sh_val
            last_sl_val[i] = cur_sl_val
            last_sh_idx[i] = cur_sh_idx
            last_sl_idx[i] = cur_sl_idx

        # ── Fibonacci levels + entry signals ──
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        # For long setup: need swing_low BEFORE swing_high, swing range > min_swing,
        # price retraces to fib level of (swing_low, swing_high), then breaks above swing_high
        # For short setup: need swing_high BEFORE swing_low, swing range > min_swing,
        # price retraces to fib level of (swing_high, swing_low), then breaks below swing_low

        fib_levels = [0.382, 0.500, 0.618]

        # Track whether price touched a fib retracement level since last swing
        touched_fib_long = np.zeros(n, dtype=np.bool_)
        touched_fib_short = np.zeros(n, dtype=np.bool_)

        prev_day = -1
        fib_long_active = False
        fib_short_active = False
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                fib_long_active = False
                fib_short_active = False

            sh = last_sh_val[i]
            sl = last_sl_val[i]
            sh_i = last_sh_idx[i]
            sl_i = last_sl_idx[i]

            # Long setup: swing low came before swing high
            if sh > 0 and sl > 0 and sl_i < sh_i and sl_i >= 0:
                swing_range = sh - sl
                if sl > 0 and swing_range / sl > min_swing:
                    # Check if current price is near a fib retracement level
                    for fib in fib_levels:
                        fib_price = sh - fib * swing_range
                        if abs(close[i] - fib_price) / sh < fib_tol and close[i] > fib_price:
                            fib_long_active = True
                            break

            # Short setup: swing high came before swing low
            if sh > 0 and sl > 0 and sh_i < sl_i and sh_i >= 0:
                swing_range = sh - sl
                if sl > 0 and swing_range / sl > min_swing:
                    for fib in fib_levels:
                        fib_price = sl + fib * swing_range
                        if abs(close[i] - fib_price) / sl < fib_tol and close[i] < fib_price:
                            fib_short_active = True
                            break

            touched_fib_long[i] = fib_long_active
            touched_fib_short[i] = fib_short_active

            # Reset if new swing detected (confirmed swing changes)
            if i > 0 and last_sh_idx[i] != last_sh_idx[i-1]:
                fib_long_active = False
            if i > 0 and last_sl_idx[i] != last_sl_idx[i-1]:
                fib_short_active = False

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Breakout detection ──
        # Long: price breaks above last swing high after touching fib
        # Short: price breaks below last swing low after touching fib
        for i in range(1, n):
            sh = last_sh_val[i]
            sl = last_sl_val[i]
            if (touched_fib_long[i] and sh > 0 and close[i] > sh and
                    close[i] > vwap[i] and vol_ok[i] and
                    time_mins[i] >= 585 and time_mins[i] <= 870):
                long_entry[i] = True
            if (touched_fib_short[i] and sl > 0 and close[i] < sl and
                    close[i] < vwap[i] and vol_ok[i] and
                    time_mins[i] >= 585 and time_mins[i] <= 870):
                short_entry[i] = True

        # ── Signal exit: price breaks back below fib level it bounced from ──
        # Approximate: close crosses back to wrong side of the swing midpoint
        swing_mid = np.where(
            (last_sh_val > 0) & (last_sl_val > 0),
            (last_sh_val + last_sl_val) / 2.0,
            0.0
        )
        below_mid_count = np.zeros(n, dtype=np.int32)
        above_mid_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if swing_mid[i] > 0 and close[i] < swing_mid[i]:
                below_mid_count[i] = below_mid_count[i-1] + 1
            else:
                below_mid_count[i] = 0
            if swing_mid[i] > 0 and close[i] > swing_mid[i]:
                above_mid_count[i] = above_mid_count[i-1] + 1
            else:
                above_mid_count[i] = 0

        signal_exit_long = below_mid_count >= 2
        signal_exit_short = above_mid_count >= 2

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
            trailing_activate_pct=0.004,
            time_stop_bars=60,
        )
