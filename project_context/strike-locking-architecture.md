# Strike Locking Architecture

## Summary
The single most damaging bug in the options pipeline: the state machine tracked "current ATM option premium" instead of "the premium of the specific strike you entered." When spot moved and ATM shifted, entry and exit premiums came from different option contracts, producing phantom profits and losses. Fixing it required replacing a 1D premium array with a 2D premium grid.

## The Bug

### Setup
- Bar 384: Spot = 24379, ATM = 24400 → buy 24400 CE at ₹316.00
- Bar 386: Spot = 24374.6, ATM shifts to 24350

### What the Bug Did
The state machine read "ATM option premium" at bar 386 → got ₹344.65 (the 24350 CE, which is deeper in-the-money and therefore more expensive).

```
Pipeline PnL: (344.65 - 316.00) × 75 = +₹2,149 (PHANTOM)
```

### What Actually Happened
The 24400 CE at bar 386 was ₹308.45 (spot went down, so the 24400 CE lost value — as it should).

```
Correct PnL: (308.45 - 316.00) × 75 = -₹101 (REAL)
```

### Impact
- vol_surface_signal showed Sharpe +7.17 → actually deeply negative
- Every trade where ATM shifted between entry and exit had wrong PnL
- ATM shifts on 3.6% of bars — enough to corrupt aggregate metrics
- Phantom profits masked by real losses → Sharpe inflated by 10-20 points

## The Wrong Architecture (1D Premium Array)

```python
# BROKEN: builds a single ATM premium per bar
atm_premiums = np.zeros(n_bars)
for i in range(n_bars):
    atm_strike = nearest_strike(spot_close[i])
    atm_premiums[i] = option_close[atm_strike][i]  # THIS changes which contract we're reading
```

The Numba state machine received this 1D array and used `atm_premiums[entry_bar]` and `atm_premiums[exit_bar]` — but these could be different contracts.

## The Correct Architecture (2D Premium Grid)

```python
# CORRECT: builds a grid of ALL nearby strikes
# premium_grid[bar_index][strike_index] = option close price
premium_grid = np.zeros((n_bars, n_strikes))  # e.g., ATM-2, ATM-1, ATM, ATM+1, ATM+2

for bar in range(n_bars):
    for s_idx, strike in enumerate(available_strikes):
        premium_grid[bar][s_idx] = option_close[strike][bar]
```

The Numba state machine:
1. On entry: records `entry_strike_idx` (which column of the grid)
2. On every bar while holding: reads `premium_grid[bar][entry_strike_idx]`
3. Stop/target comparisons use `premium_grid[bar][entry_strike_idx]`
4. On exit: PnL = `premium_grid[exit_bar][entry_strike_idx] - premium_grid[entry_bar][entry_strike_idx]`

The entry strike is LOCKED for the duration of the trade.

## Verification Protocol

After implementing the fix, verify with the nuclear check:

```python
import polars as pl

opts = pl.read_parquet("5second_data/option_candles.parquet")

# For trade #1: entry on 24400 CE at epoch X, exit at epoch Y
strike_data = opts.filter(
    (pl.col("strike") == 24400) &
    (pl.col("option_type") == "CE") &
    (pl.col("underlying") == "NSE:NIFTY50-INDEX") &
    (pl.col("session_date") == "2026-03-04")
).sort("epoch")

# Read the raw premium at entry and exit epochs
# Compare against pipeline output
# Must match within ₹0.01
```

Run this for EVERY trade: add a "raw_check" column that re-reads premiums directly from the parquet file. Zero tolerance for mismatches.

## Edge Cases

### Missing Premium Data
If a strike becomes so far OTM that it's no longer in the option chain data at exit time, the premium may be missing. Handle by: force exit at last known premium for that strike. Log the event.

### Strike Step Size
NIFTY strikes are at 50-point intervals. BANKNIFTY at 100-point intervals. ATM identification uses nearest strike, not rounding:
- NIFTY at 24025 → ATM = 24050 (nearest), not 24000
- Use `floor(spot + step/2) / step * step` for consistent rounding, not Python's banker's rounding (`round(480.5) = 480` is wrong)

### Expiry Selection
When multiple expiries exist in the data, the pipeline uses the nearest expiry. There's no explicit parameter for weekly vs monthly. This should be made explicit per strategy.

## Indicator Corruption (Related)

Any strategy that computes indicators on "ATM option data" suffers from the same bug. If the indicator tracks "ATM CE IV" across bars, and ATM shifts, the indicator jumps between strikes — it's noise, not a signal.

**Fix**: Indicators that use option data should use a FIXED reference (e.g., day-open ATM strike ±100) for the entire day, not the shifting current ATM.

Of 6 strategies audited, 5 used spot/VIX-only indicators (safe). Only vol_surface_signal used option-chain indicators (was corrupt, fixed to use day-open reference strikes).

## Decisions Made
- 2D grid over 1D array — architecturally harder but the only correct approach
- Grid covers ATM ± 2 strikes (5 columns) — enough for strategies using strike_offset 0, ±1
- Strike locked at entry, never re-evaluated during the trade
- Nuclear verification (raw parquet comparison) is mandatory after any premium-handling code change

## Corrections Log
- ~~1D ATM premium array was "adequate for short holds"~~ → Wrong. ATM shifts on 3.6% of bars, enough to corrupt Sharpe by 10-20 points
- ~~The initial "fix" patched individual trades~~ → Root cause was architectural (1D vs 2D). Patching symptoms didn't fix it.
- ~~vol_surface_signal Sharpe +7.17~~ → Entirely phantom. After correct strike locking: deeply negative.
- ~~ATM identification used Python `round()`~~ → Banker's rounding is inconsistent (`round(480.5) = 480`). Changed to `floor(x + 0.5)`.
