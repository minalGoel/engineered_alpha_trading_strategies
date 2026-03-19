"""Numba-compiled position state machine.

This is the single hottest function in the pipeline. It processes bar-by-bar
through numpy arrays, tracking FLAT→LONG→SHORT state, applying exit rules
(stop, target, trailing, time, signal, EOD), and recording trades.

All inputs are numpy arrays (float64/int32/bool). No Python objects inside.
"""
from __future__ import annotations
import numpy as np
from numba import njit

# Position states
_FLAT = 0
_LONG = 1
_SHORT = 2

# Exit reasons (encoded as int for Numba)
_EXIT_NONE = 0
_EXIT_STOP = 1
_EXIT_TARGET = 2
_EXIT_TRAILING = 3
_EXIT_SIGNAL = 4
_EXIT_TIME = 5
_EXIT_EOD = 6

EXIT_REASON_MAP = {
    _EXIT_NONE: "NONE",
    _EXIT_STOP: "STOP",
    _EXIT_TARGET: "TARGET",
    _EXIT_TRAILING: "TRAILING",
    _EXIT_SIGNAL: "SIGNAL",
    _EXIT_TIME: "TIME",
    _EXIT_EOD: "EOD",
}


@njit(cache=True)
def run_state_machine(
    # Price data
    open_arr: np.ndarray,     # float64[n]
    high_arr: np.ndarray,     # float64[n]
    low_arr: np.ndarray,      # float64[n]
    close_arr: np.ndarray,    # float64[n]
    # Day/time info
    day_id: np.ndarray,       # int32[n]
    time_minutes: np.ndarray, # int32[n] — minutes since midnight
    # Entry signals (pre-computed boolean masks)
    long_entry: np.ndarray,   # bool[n]
    short_entry: np.ndarray,  # bool[n]
    signal_exit_long: np.ndarray,  # bool[n] — signal-based exit for longs
    signal_exit_short: np.ndarray, # bool[n] — signal-based exit for shorts
    # ATR for ATR-based stops (can be zeros if not used)
    atr_arr: np.ndarray,      # float64[n]
    # Indicator values for target (e.g., vwap for "vwap touch" targets)
    target_indicator: np.ndarray,  # float64[n] — target reference level (vwap, ema, etc.)
    # Exit parameters
    stop_loss_pct: float,         # as decimal (0.003 = 0.3%)
    stop_loss_atr_mult: float,    # 0 if not using ATR stops
    target_pct: float,            # as decimal (0 if no % target)
    target_atr_mult: float,       # 0 if not using ATR targets
    use_target_indicator: bool,   # True if target is vwap/ema touch
    trailing_stop_pct: float,     # as decimal (0 if no trailing)
    trailing_activate_pct: float, # activate after this profit (0 = always active)
    breakeven_pct: float,         # move SL to breakeven after this profit (0 = disabled)
    time_stop_bars: int,          # 0 if no time stop
    # Session limits
    eod_flatten_minutes: int,     # e.g., 920 for 15:20 IST
    session_start_minutes: int,   # e.g., 555 for 09:15 IST
    session_end_minutes: int,     # e.g., 920 for 15:20 IST
    # Risk limits
    max_trades_per_day: int,
    max_daily_loss: float,        # in currency units (0 = no limit)
    capital_per_trade: float,
    # Warmup
    warmup_bars: int,
) -> tuple:
    """Run the state machine and return trade arrays.

    Returns:
        entry_bar: int64[max_trades] — bar index of entry
        exit_bar: int64[max_trades] — bar index of exit
        side: int8[max_trades] — 1=LONG, 2=SHORT
        entry_price: float64[max_trades]
        exit_price: float64[max_trades]
        exit_reason: int8[max_trades]
        trade_count: int — actual number of trades
    """
    n = len(close_arr)
    max_trades = n // 2 + 1  # upper bound on number of trades

    # Output arrays
    out_entry_bar = np.empty(max_trades, dtype=np.int64)
    out_exit_bar = np.empty(max_trades, dtype=np.int64)
    out_side = np.empty(max_trades, dtype=np.int8)
    out_entry_price = np.empty(max_trades, dtype=np.float64)
    out_exit_price = np.empty(max_trades, dtype=np.float64)
    out_exit_reason = np.empty(max_trades, dtype=np.int8)

    trade_count = 0
    state = _FLAT
    entry_price = 0.0
    entry_bar_idx = 0
    stop_price = 0.0
    target_price = 0.0
    trailing_stop_price = 0.0
    trailing_active = False
    highest_since_entry = 0.0
    lowest_since_entry = 1e18
    bars_held = 0
    current_day = -1
    daily_trades = 0
    daily_pnl = 0.0
    daily_loss_hit = False

    for i in range(n):
        # ── Day boundary detection ──────────────────────────────────────
        if day_id[i] != current_day:
            current_day = day_id[i]
            daily_trades = 0
            daily_pnl = 0.0
            daily_loss_hit = False

        # Skip warmup
        if i < warmup_bars:
            continue

        # ── In position: check exits ────────────────────────────────────
        if state != _FLAT:
            bars_held += 1
            exit_triggered = False
            reason = _EXIT_NONE
            fill_price = 0.0

            if state == _LONG:
                # Update tracking
                if high_arr[i] > highest_since_entry:
                    highest_since_entry = high_arr[i]

                # 1. Stop loss (check against bar's low)
                if stop_price > 0 and low_arr[i] <= stop_price:
                    fill_price = stop_price
                    reason = _EXIT_STOP
                    exit_triggered = True

                # 2. Target (check against bar's high)
                if not exit_triggered:
                    actual_target = target_price
                    if use_target_indicator and target_indicator[i] == target_indicator[i]:
                        actual_target = target_indicator[i]
                    if actual_target > 0 and high_arr[i] >= actual_target:
                        fill_price = actual_target
                        reason = _EXIT_TARGET
                        exit_triggered = True

                # 3. Trailing stop
                if not exit_triggered and trailing_stop_pct > 0:
                    profit_pct = (close_arr[i] - entry_price) / entry_price
                    activate_threshold = trailing_activate_pct if trailing_activate_pct > 0 else 0.0

                    if profit_pct >= activate_threshold:
                        trailing_active = True

                    if trailing_active:
                        new_trail = highest_since_entry * (1.0 - trailing_stop_pct)
                        if new_trail > trailing_stop_price:
                            trailing_stop_price = new_trail
                        if low_arr[i] <= trailing_stop_price and trailing_stop_price > 0:
                            fill_price = trailing_stop_price
                            reason = _EXIT_TRAILING
                            exit_triggered = True

                    # Breakeven logic
                    if not exit_triggered and breakeven_pct > 0:
                        if profit_pct >= breakeven_pct:
                            if stop_price < entry_price:
                                stop_price = entry_price

                # 4. Signal exit
                if not exit_triggered and signal_exit_long[i]:
                    fill_price = close_arr[i]
                    reason = _EXIT_SIGNAL
                    exit_triggered = True

                # 5. Time stop
                if not exit_triggered and time_stop_bars > 0 and bars_held >= time_stop_bars:
                    fill_price = close_arr[i]
                    reason = _EXIT_TIME
                    exit_triggered = True

                # 6. EOD flatten (lowest priority — only if no other exit triggered)
                if not exit_triggered and time_minutes[i] >= eod_flatten_minutes:
                    fill_price = close_arr[i]
                    reason = _EXIT_EOD
                    exit_triggered = True

            else:  # SHORT
                # Update tracking
                if low_arr[i] < lowest_since_entry:
                    lowest_since_entry = low_arr[i]

                # 1. Stop loss (check against bar's high)
                if stop_price > 0 and high_arr[i] >= stop_price:
                    fill_price = stop_price
                    reason = _EXIT_STOP
                    exit_triggered = True

                # 2. Target (check against bar's low)
                if not exit_triggered:
                    actual_target = target_price
                    if use_target_indicator and target_indicator[i] == target_indicator[i]:
                        actual_target = target_indicator[i]
                    if actual_target > 0 and low_arr[i] <= actual_target:
                        fill_price = actual_target
                        reason = _EXIT_TARGET
                        exit_triggered = True

                # 3. Trailing stop
                if not exit_triggered and trailing_stop_pct > 0:
                    profit_pct = (entry_price - close_arr[i]) / entry_price
                    activate_threshold = trailing_activate_pct if trailing_activate_pct > 0 else 0.0

                    if profit_pct >= activate_threshold:
                        trailing_active = True

                    if trailing_active:
                        new_trail = lowest_since_entry * (1.0 + trailing_stop_pct)
                        if new_trail < trailing_stop_price or trailing_stop_price == 0:
                            trailing_stop_price = new_trail
                        if high_arr[i] >= trailing_stop_price and trailing_stop_price > 0:
                            fill_price = trailing_stop_price
                            reason = _EXIT_TRAILING
                            exit_triggered = True

                    # Breakeven
                    if not exit_triggered and breakeven_pct > 0:
                        if profit_pct >= breakeven_pct:
                            if stop_price > entry_price:
                                stop_price = entry_price

                # 4. Signal exit
                if not exit_triggered and signal_exit_short[i]:
                    fill_price = close_arr[i]
                    reason = _EXIT_SIGNAL
                    exit_triggered = True

                # 5. Time stop
                if not exit_triggered and time_stop_bars > 0 and bars_held >= time_stop_bars:
                    fill_price = close_arr[i]
                    reason = _EXIT_TIME
                    exit_triggered = True

                # 6. EOD flatten (lowest priority — only if no other exit triggered)
                if not exit_triggered and time_minutes[i] >= eod_flatten_minutes:
                    fill_price = close_arr[i]
                    reason = _EXIT_EOD
                    exit_triggered = True

            if exit_triggered:
                if trade_count >= max_trades:
                    state = _FLAT
                    continue

                if state == _LONG:
                    pnl = (fill_price - entry_price) * (capital_per_trade / entry_price)
                else:
                    pnl = (entry_price - fill_price) * (capital_per_trade / entry_price)
                daily_pnl += pnl

                out_entry_bar[trade_count] = entry_bar_idx
                out_exit_bar[trade_count] = i
                out_side[trade_count] = state
                out_entry_price[trade_count] = entry_price
                out_exit_price[trade_count] = fill_price
                out_exit_reason[trade_count] = reason
                trade_count += 1
                state = _FLAT

                # Check daily loss limit
                if max_daily_loss > 0 and daily_pnl < -max_daily_loss:
                    daily_loss_hit = True

        # ── Entry logic (only when FLAT) ────────────────────────────────
        if state == _FLAT:
            # Skip if within session but outside entry window
            if time_minutes[i] < session_start_minutes or time_minutes[i] >= session_end_minutes:
                continue

            # Skip if daily limits hit
            if daily_loss_hit:
                continue
            if max_trades_per_day > 0 and daily_trades >= max_trades_per_day:
                continue

            entered = False

            # Check long entry
            if long_entry[i]:
                state = _LONG
                entry_price = close_arr[i]
                entry_bar_idx = i
                bars_held = 0
                highest_since_entry = high_arr[i]
                lowest_since_entry = low_arr[i]
                trailing_stop_price = 0.0
                trailing_active = False
                entered = True

                # Compute stop and target prices
                if stop_loss_pct > 0:
                    stop_price = entry_price * (1.0 - stop_loss_pct)
                elif stop_loss_atr_mult > 0 and atr_arr[i] == atr_arr[i]:
                    stop_price = entry_price - stop_loss_atr_mult * atr_arr[i]
                else:
                    stop_price = 0.0

                if target_pct > 0:
                    target_price = entry_price * (1.0 + target_pct)
                elif target_atr_mult > 0 and atr_arr[i] == atr_arr[i]:
                    target_price = entry_price + target_atr_mult * atr_arr[i]
                else:
                    target_price = 0.0

            # Check short entry (only if not just entered long)
            if not entered and short_entry[i]:
                state = _SHORT
                entry_price = close_arr[i]
                entry_bar_idx = i
                bars_held = 0
                highest_since_entry = high_arr[i]
                lowest_since_entry = low_arr[i]
                trailing_stop_price = 1e18
                trailing_active = False
                entered = True

                # Compute stop and target for short
                if stop_loss_pct > 0:
                    stop_price = entry_price * (1.0 + stop_loss_pct)
                elif stop_loss_atr_mult > 0 and atr_arr[i] == atr_arr[i]:
                    stop_price = entry_price + stop_loss_atr_mult * atr_arr[i]
                else:
                    stop_price = 0.0

                if target_pct > 0:
                    target_price = entry_price * (1.0 - target_pct)
                elif target_atr_mult > 0 and atr_arr[i] == atr_arr[i]:
                    target_price = entry_price - target_atr_mult * atr_arr[i]
                else:
                    target_price = 0.0

            if entered:
                daily_trades += 1

    # ── Force-close any position still open at end of data ────────────
    if state != _FLAT and trade_count < max_trades:
        fill_price = close_arr[n - 1]
        if state == _LONG:
            pnl = (fill_price - entry_price) * (capital_per_trade / entry_price)
        else:
            pnl = (entry_price - fill_price) * (capital_per_trade / entry_price)
        out_entry_bar[trade_count] = entry_bar_idx
        out_exit_bar[trade_count] = n - 1
        out_side[trade_count] = state
        out_entry_price[trade_count] = entry_price
        out_exit_price[trade_count] = fill_price
        out_exit_reason[trade_count] = _EXIT_EOD
        trade_count += 1

    # Trim output arrays
    return (
        out_entry_bar[:trade_count],
        out_exit_bar[:trade_count],
        out_side[:trade_count],
        out_entry_price[:trade_count],
        out_exit_price[:trade_count],
        out_exit_reason[:trade_count],
        trade_count,
    )


def warmup_numba():
    """Pre-compile the Numba function with small dummy data."""
    n = 100
    dummy = np.random.randn(n).astype(np.float64)
    dummy_pos = np.abs(dummy) + 1.0
    day_ids = np.zeros(n, dtype=np.int32)
    time_mins = np.full(n, 600, dtype=np.int32)  # 10:00
    bools = np.zeros(n, dtype=np.bool_)
    target_ind = np.zeros(n, dtype=np.float64)

    run_state_machine(
        dummy_pos, dummy_pos + 0.1, dummy_pos - 0.1, dummy_pos,
        day_ids, time_mins,
        bools, bools, bools, bools,
        np.ones(n, dtype=np.float64),
        target_ind,
        0.003, 0.0, 0.005, 0.0, False,
        0.002, 0.0, 0.0, 30,
        920, 555, 920,
        10, 5000.0, 100000.0, 5,
    )
