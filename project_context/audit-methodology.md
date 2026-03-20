# Audit Methodology

## Summary
A 3-pass audit approach where each pass uses a different methodology — code reading, data flow tracing, and real-data execution. Each pass caught bugs the prior ones missed. The critical lesson: the most destructive bugs only surface when running on real data with realistic conditions.

## The Three Passes

### Pass 1: Code Reading (Catches ~40% of bugs)
Read every file character by character. Check:
- Signal codes consistent across all modules
- State transitions valid (can't BUY from LONG state)
- Division-by-zero guards present
- NaN propagation paths identified
- Array bounds checked
- Type consistency (int8 vs int32 vs float64)

**What it catches**: Obvious logic errors, missing guards, inconsistent constants.
**What it misses**: Timezone bugs, data-dependent bugs, bugs that only manifest with specific parameter combinations.

### Pass 2: Data Flow Trace (Catches ~30% of remaining bugs)
Pick ONE trade and trace it through the ENTIRE system:
1. Load raw data → verify timestamps
2. Compute indicators → print values at specific bars
3. Check entry conditions → verify all conditions satisfied
4. Follow through state machine → check state transitions
5. Track exit logic → verify stop/target/time/EOD
6. Compute PnL → verify by hand
7. Check metrics aggregation → verify daily sums

**What it catches**: Data transformation bugs, indicator computation errors, PnL miscalculations.
**What it misses**: Bugs that only appear on specific data patterns (e.g., ATR=0 during warmup).

### Pass 3: Real-Data Execution (Catches ~30% of remaining bugs)
Run the full pipeline on actual market data. Check results for impossible values:
- CE premium going UP when spot goes DOWN
- Trades outside market hours
- Strategy with 100% win rate
- Sharpe > 10
- Zero trades when signals clearly fired

**What it catches**: Everything the other passes missed. The UTC→IST bug survived two passes and was only caught when someone noticed entries clustered in the last 45 minutes of each day.

## The Nuclear Verification

For any result you need to trust absolutely, bypass the entire pipeline:

```python
# Read raw parquet directly
opts = pl.read_parquet("5second_data/option_candles.parquet")
strike_data = opts.filter(
    (pl.col("strike") == entry_strike) &
    (pl.col("option_type") == entry_type) &
    (pl.col("session_date") == entry_date)
).sort("epoch")

# Compare pipeline's entry/exit premium against raw data
# at the exact epoch timestamps
```

This bypasses data_loader, option_utils, and the premium grid — reading directly from the source file. If the pipeline's numbers match the raw parquet, the pipeline is correct. If they don't, there's a data processing bug.

In the options pipeline, this check caught the strike-locking bug: the pipeline showed +₹2,149 profit on a trade where the raw parquet showed -₹101.

## Specific Checks by Category

### Timestamp Verification
- Print first bar's hour in IST. Must be 9.
- Print last bar's hour in IST. Must be 15.
- Count bars per day. Must be consistent (4,500 for 5-second, 375 for 1-minute).

### Signal-Condition Alignment
For every entry signal, verify the actual indicator values satisfy the conditions:
```
Condition: spot_return_6bar > 0.0003 | Actual: 0.00047 | PASS
Condition: vol_ratio > 1.5          | Actual: 1.82     | PASS
```
If any condition fails at an entry bar, the signal generation logic is buggy.

### Strike Locking (Options Only)
For each trade, verify entry and exit premiums come from the SAME strike:
- Entry: spot=24379, strike=24400, CE premium=316
- Exit: spot=24374.6 (ATM shifted to 24350), CE premium of 24400 = 308.45 (not 344.65 from 24350)

### Cost Model Verification
Compute one trade's costs by hand:
```
NIFTY 24000 CE: entry ₹300, exit ₹303, 1 lot (75 units)
Gross: (303-300) × 75 = ₹225
Spread: 1.0 × 2 × 75 = ₹150
STT: 303 × 0.000625 × 75 = ₹14.20
Brokerage: 20 × 2 = ₹40
Exchange: 5 × 2 = ₹10
Net: 225 - 150 - 14.20 - 40 - 10 = ₹10.80
```
Compare against pipeline output. Must match within ₹0.01.

### Cross-Module Consistency
grep for critical constants across the entire codebase:
- `"IST"`, `"UTC"`, `"convert_time_zone"` — timezone handling
- `"lot_size"`, `"75"`, `"15"` — NIFTY/BANKNIFTY lot sizes
- `"0.000625"`, `"0.0625"`, `"STT"` — tax rates
- Signal codes: `BUY_CE=1`, `SELL_CE=2`, `BUY_PE=3`, `SELL_PE=4`

## Decisions Made
- Three passes with different methodologies > three identical passes
- Real-data execution is mandatory — synthetic unit tests miss the worst bugs
- The nuclear verification (raw parquet comparison) is the gold standard for any disputed result
- Every audit must fix bugs AND re-verify after fixing

## Pitfalls & Anti-patterns
- **"Looks fine" without running code**: The options pipeline passed a code review but had never actually executed a backtest. It counted signals and reported them as trades.
- **Fixing one symptom without understanding the root cause**: The strike-locking "fix" initially patched one trade. The root cause was the 1D premium array architecture — fixing required a 2D premium grid.
- **Trusting audit results from the same session that wrote the code**: Sonnet auditing Sonnet's own code is self-grading. Use a fresh session (ideally Opus) for audits.

## Corrections Log
- ~~Two audit passes were considered sufficient~~ → Three passes needed, each with different methodology. The UTC bug survived two passes.
- ~~Unit tests on synthetic data validate the pipeline~~ → Necessary but insufficient. The most destructive bugs only surface on real data.
