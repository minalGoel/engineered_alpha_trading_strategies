"""Numba-compiled position state machine for option trading.

Handles FLAT → LONG_CE / LONG_PE → FLAT state transitions.
PnL is computed in option premium points × lot size.
No simultaneous CE + PE positions.
No option writing/selling to open — buy CE or buy PE only, sell to close.

CRITICAL: Entry strike is LOCKED for the duration of the trade.
Premium lookup always uses the entry strike, not the current ATM strike.
"""
from __future__ import annotations
import numpy as np
from numba import njit

# Position states
_FLAT = 0
_LONG_CE = 1  # holding a bought call
_LONG_PE = 3  # holding a bought put

# Exit reasons (encoded as int for Numba)
_EXIT_NONE = 0
_EXIT_STOP = 1
_EXIT_TARGET = 2
_EXIT_SIGNAL = 4
_EXIT_TIME = 5
_EXIT_EOD = 6

EXIT_REASON_MAP = {
    _EXIT_NONE: "NONE",
    _EXIT_STOP: "STOP",
    _EXIT_TARGET: "TARGET",
    _EXIT_SIGNAL: "SIGNAL",
    _EXIT_TIME: "TIME",
    _EXIT_EOD: "EOD",
}


@njit(cache=True)
def run_state_machine(
    # Spot price data (for signal timing — length n)
    spot_close: np.ndarray,       # float64[n] — index close
    # Option premium grid (for PnL — n × n_strikes)
    # Each cell [bar, strike_idx] = close premium of that strike at that bar.
    ce_grid: np.ndarray,          # float64[n, n_strikes]
    pe_grid: np.ndarray,          # float64[n, n_strikes]
    # ATM strike index per bar
    atm_strike_idx: np.ndarray,   # int32[n] — index into strikes dimension
    # Day/time info
    day_id: np.ndarray,           # int32[n]
    time_minutes: np.ndarray,     # int32[n] — IST minutes from midnight
    # Entry signals (pre-computed boolean masks)
    buy_ce: np.ndarray,           # bool[n] — buy call signal
    buy_pe: np.ndarray,           # bool[n] — buy put signal
    # Exit signals
    sell_ce: np.ndarray,          # bool[n] — signal exit for calls
    sell_pe: np.ndarray,          # bool[n] — signal exit for puts
    # Stop/target in option premium points (per bar, length n)
    stop_points: np.ndarray,      # float64[n] — stop loss in premium pts
    target_points: np.ndarray,    # float64[n] — target in premium pts
    # Scalar parameters
    time_stop_bars: int,          # max bars before time stop (0=disabled)
    # Session limits
    eod_flatten_minutes: int,     # e.g., 925 for 15:25 IST
    session_start_minutes: int,   # e.g., 560 for 09:20 IST
    session_end_minutes: int,     # e.g., 925 for 15:25 IST
    # Risk limits
    max_trades_per_day: int,
    # Lot size for PnL
    lot_size: int,
    # Warmup
    warmup_bars: int,
    # Is expiry day flags (for early flatten)
    is_expiry: np.ndarray,        # bool[n]
    expiry_flatten_minutes: int,  # e.g., 920 for 15:20 IST
) -> tuple:
    """Run the option trading state machine with strike-locked positions.

    Returns:
        entry_bar: int64[max_trades]
        exit_bar: int64[max_trades]
        side: int8[max_trades] — 1=CE, 3=PE
        entry_premium: float64[max_trades] — option premium at entry
        exit_premium: float64[max_trades] — option premium at exit
        exit_reason: int8[max_trades]
        pnl: float64[max_trades] — (exit-entry) × lot_size per trade
        entry_strike_indices: int32[max_trades] — locked strike index
        premium_missing: bool[max_trades] — True if premium was missing at exit
        trade_count: int
    """
    n = len(spot_close)
    max_trades = n // 2 + 1

    # Output arrays
    out_entry_bar = np.empty(max_trades, dtype=np.int64)
    out_exit_bar = np.empty(max_trades, dtype=np.int64)
    out_side = np.empty(max_trades, dtype=np.int8)
    out_entry_premium = np.empty(max_trades, dtype=np.float64)
    out_exit_premium = np.empty(max_trades, dtype=np.float64)
    out_exit_reason = np.empty(max_trades, dtype=np.int8)
    out_pnl = np.empty(max_trades, dtype=np.float64)
    out_entry_strike_idx = np.empty(max_trades, dtype=np.int32)
    out_premium_missing = np.zeros(max_trades, dtype=np.bool_)

    trade_count = 0
    state = _FLAT
    entry_premium = 0.0
    entry_bar_idx = 0
    entry_strike_idx_locked = 0   # strike index locked at entry
    stop_price = 0.0
    target_price = 0.0
    bars_held = 0
    current_day = -1
    daily_trades = 0
    last_known_premium = 0.0
    premium_was_missing = False

    for i in range(n):
        # ── Day boundary detection ──
        if day_id[i] != current_day:
            current_day = day_id[i]
            daily_trades = 0

        # Skip warmup
        if i < warmup_bars:
            continue

        # ── In position: check exits ──
        if state != _FLAT:
            bars_held += 1

            # Get current option premium at LOCKED strike
            if state == _LONG_CE:
                current_premium = ce_grid[i, entry_strike_idx_locked]
            else:
                current_premium = pe_grid[i, entry_strike_idx_locked]

            # Handle missing premium data (0 or NaN)
            if current_premium <= 0.0 or current_premium != current_premium:
                current_premium = last_known_premium
                premium_was_missing = True
            else:
                last_known_premium = current_premium

            exit_triggered = False
            reason = _EXIT_NONE
            fill_premium = 0.0

            # Determine effective flatten time
            flatten_time = eod_flatten_minutes
            if is_expiry[i] and expiry_flatten_minutes < eod_flatten_minutes:
                flatten_time = expiry_flatten_minutes

            # 1. Stop loss — premium drops below entry - stop_points
            if stop_price > 0 and current_premium <= stop_price:
                fill_premium = current_premium
                reason = _EXIT_STOP
                exit_triggered = True

            # 2. Target — premium rises above entry + target_points
            if not exit_triggered and target_price > 0 and current_premium >= target_price:
                fill_premium = current_premium
                reason = _EXIT_TARGET
                exit_triggered = True

            # 3. Signal exit
            if not exit_triggered:
                if state == _LONG_CE and sell_ce[i]:
                    fill_premium = current_premium
                    reason = _EXIT_SIGNAL
                    exit_triggered = True
                elif state == _LONG_PE and sell_pe[i]:
                    fill_premium = current_premium
                    reason = _EXIT_SIGNAL
                    exit_triggered = True

            # 4. Time stop
            if not exit_triggered and time_stop_bars > 0 and bars_held >= time_stop_bars:
                fill_premium = current_premium
                reason = _EXIT_TIME
                exit_triggered = True

            # 5. EOD / Expiry flatten
            if not exit_triggered and time_minutes[i] >= flatten_time:
                fill_premium = current_premium
                reason = _EXIT_EOD
                exit_triggered = True

            if exit_triggered:
                if trade_count >= max_trades:
                    state = _FLAT
                    continue

                pnl = (fill_premium - entry_premium) * lot_size

                out_entry_bar[trade_count] = entry_bar_idx
                out_exit_bar[trade_count] = i
                out_side[trade_count] = state
                out_entry_premium[trade_count] = entry_premium
                out_exit_premium[trade_count] = fill_premium
                out_exit_reason[trade_count] = reason
                out_pnl[trade_count] = pnl
                out_entry_strike_idx[trade_count] = entry_strike_idx_locked
                out_premium_missing[trade_count] = premium_was_missing
                trade_count += 1
                state = _FLAT

        # ── Entry logic (only when FLAT) ──
        if state == _FLAT:
            if time_minutes[i] < session_start_minutes or time_minutes[i] >= session_end_minutes:
                continue
            if max_trades_per_day > 0 and daily_trades >= max_trades_per_day:
                continue

            entered = False
            s_idx = atm_strike_idx[i]

            # Check CE entry (bullish)
            if buy_ce[i]:
                premium = ce_grid[i, s_idx]
                # Guard: premium must be valid (not NaN, not zero)
                if premium == premium and premium > 0:
                    state = _LONG_CE
                    entry_premium = premium
                    entry_bar_idx = i
                    entry_strike_idx_locked = s_idx
                    last_known_premium = premium
                    premium_was_missing = False
                    bars_held = 0
                    entered = True

                    # Set stop/target from per-bar arrays
                    sp = stop_points[i]
                    if sp == sp and sp > 0:
                        stop_price = entry_premium - sp
                        if stop_price < 0:
                            stop_price = 0.1  # floor at near-zero
                    else:
                        stop_price = 0.0

                    tp = target_points[i]
                    if tp == tp and tp > 0:
                        target_price = entry_premium + tp
                    else:
                        target_price = 0.0

            # Check PE entry (bearish) — only if not just entered CE
            if not entered and buy_pe[i]:
                premium = pe_grid[i, s_idx]
                if premium == premium and premium > 0:
                    state = _LONG_PE
                    entry_premium = premium
                    entry_bar_idx = i
                    entry_strike_idx_locked = s_idx
                    last_known_premium = premium
                    premium_was_missing = False
                    bars_held = 0
                    entered = True

                    sp = stop_points[i]
                    if sp == sp and sp > 0:
                        stop_price = entry_premium - sp
                        if stop_price < 0:
                            stop_price = 0.1
                    else:
                        stop_price = 0.0

                    tp = target_points[i]
                    if tp == tp and tp > 0:
                        target_price = entry_premium + tp
                    else:
                        target_price = 0.0

            if entered:
                daily_trades += 1

    # ── Force-close open position at end of data ──
    if state != _FLAT and trade_count < max_trades:
        if state == _LONG_CE:
            fill_premium = ce_grid[n - 1, entry_strike_idx_locked]
        else:
            fill_premium = pe_grid[n - 1, entry_strike_idx_locked]

        if fill_premium <= 0.0 or fill_premium != fill_premium:
            fill_premium = last_known_premium
            premium_was_missing = True

        pnl = (fill_premium - entry_premium) * lot_size

        out_entry_bar[trade_count] = entry_bar_idx
        out_exit_bar[trade_count] = n - 1
        out_side[trade_count] = state
        out_entry_premium[trade_count] = entry_premium
        out_exit_premium[trade_count] = fill_premium
        out_exit_reason[trade_count] = _EXIT_EOD
        out_pnl[trade_count] = pnl
        out_entry_strike_idx[trade_count] = entry_strike_idx_locked
        out_premium_missing[trade_count] = premium_was_missing
        trade_count += 1

    return (
        out_entry_bar[:trade_count],
        out_exit_bar[:trade_count],
        out_side[:trade_count],
        out_entry_premium[:trade_count],
        out_exit_premium[:trade_count],
        out_exit_reason[:trade_count],
        out_pnl[:trade_count],
        out_entry_strike_idx[:trade_count],
        out_premium_missing[:trade_count],
        trade_count,
    )


def warmup_numba():
    """Pre-compile the Numba function with small dummy data."""
    n = 100
    n_strikes = 5
    ce_grid = np.abs(np.random.randn(n, n_strikes).astype(np.float64)) + 50.0
    pe_grid = np.abs(np.random.randn(n, n_strikes).astype(np.float64)) + 50.0
    atm_idx = np.full(n, 2, dtype=np.int32)  # middle strike
    spot = np.abs(np.random.randn(n).astype(np.float64)) * 100 + 24000
    day_ids = np.zeros(n, dtype=np.int32)
    time_mins = np.full(n, 600, dtype=np.int32)
    bools = np.zeros(n, dtype=np.bool_)
    stops = np.full(n, 5.0, dtype=np.float64)
    targets = np.full(n, 10.0, dtype=np.float64)
    is_exp = np.zeros(n, dtype=np.bool_)

    run_state_machine(
        spot, ce_grid, pe_grid, atm_idx,
        day_ids, time_mins,
        bools, bools, bools, bools,
        stops, targets,
        24,
        925, 560, 925,
        10,
        75,
        5,
        is_exp, 920,
    )
