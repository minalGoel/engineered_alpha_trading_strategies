# Backtesting Pipeline Design

## Summary
An 8-phase backtesting pipeline that takes strategy definitions through data discovery, optimization, validation, and reporting. Designed for unattended EC2 execution with crash resilience. The pipeline went through 4 audit passes and 33+ bug fixes before producing trustworthy results.

## The 8 Phases

### Phase 1: Data Discovery & Preparation
- Explore data directory (never assume column names or file structure)
- Document everything in `outputs/data_inventory.json`
- Convert CSV → Parquet once (5-10x faster reads thereafter)
- Validate date ranges, flag gaps >5% missing trading days

### Phase 2: Strategy Parsing & Indicator Coverage
- Parse all strategy JSONs
- Determine which indicators can be computed, which conditions can be parsed
- Log unparseable strategies as FAILED_PARSE_ERROR immediately
- Create strategy folder and verdict.json for failures

### Phase 3: Parameter Optimization (Training Data ONLY)

**Step 3a: Cross-validation with DEFAULT parameters**
- Run strategy with original parameters through walk-forward CV
- The thesis must work WITHOUT optimization. If it doesn't, optimization is curve-fitting.
- Must be profitable in ≥4 of 5 test folds (or ≥9 of 12 for leave-one-day-out)
- If fails → FAILED_OVERFIT

**Step 3b: Parameter optimization**
- Only for strategies that passed 3a
- Extract tunable parameters: ONLY numeric thresholds, NOT indicator periods
- Compute indicators ONCE before optimization (periods are fixed)
- Optuna: TPESampler, 100 trials, NO pruner, 30-min timeout
- Objective: after-cost Sharpe ratio (or pre-cost for the equity version)
- Reject combos with <50 trades
- If optimized Sharpe > 3× default Sharpe → flag POTENTIAL_OVERFIT

### Phase 4: Per-Instrument Backtesting (Training, Optimized Params)
- Parameters locked from Phase 3
- Run each strategy on each instrument individually
- Purpose: diagnostic — which instruments does this strategy work on?
- Save per-instrument results

### Phase 5: Instrument Filtering
A stock/instrument passes if ALL hold:
- ≥30 trades (Chan's minimum)
- Profit factor > 1.2
- Sharpe > 0.5
- Max drawdown < 50% of total PnL (only when PnL > 0)
- Do NOT filter on win rate (kills valid low-win-rate trend strategies)

Strategies with <10 passing instruments → FAILED_NARROW

### Phase 6: Out-of-Sample Validation
- NOW touch test data (2025 for equity, last 4 days for options)
- Run with optimized params, on passing instruments only
- Validates if: PF > 1.0, Sharpe > 0.0, Sharpe retention ≥ 30%, ≥50% of instruments still profitable

### Phase 7: Per-Strategy Reports
- Full markdown report for EVERY strategy (pass or fail)
- Includes: thesis, mechanics, parameters table, train/test metrics, stock coverage, last 20 signals, weaknesses, assumptions made

### Phase 8: Leaderboard
- Ranking: `Test Sharpe × sqrt(passing_instrument_count)`
- Data-snooping correction: `required_Sharpe = 0.5 × sqrt(1 + ln(N))`
- Failure breakdown by category (tells whether strategy generation produces garbage or near-misses)
- `master_signals.parquet` — all signals from winners for downstream OMS

## Folder Structure

```
strategy_results/{strategy_name}/
├── strategy.json           # copy of original definition
├── optimization.json       # default vs optimized params, CV fold results
├── stock_filter.json       # passing/failing instruments with reasons
├── verdict.json            # VALIDATED / FAILED_* with reason
├── training/
│   ├── signals.parquet
│   ├── trades.parquet
│   ├── metrics.json
│   ├── monthly_pnl.json
│   └── per_stock/{SYMBOL}.json
├── test/
│   └── (same structure)
└── report.md
```

Every file must be self-contained. Zip one folder → complete strategy audit.

## Verdict Categories
- `VALIDATED` — passed everything, edge appears real
- `FAILED_PARSE_ERROR` — couldn't parse indicators or conditions
- `FAILED_NO_TRADES` — zero trades with default params on training
- `FAILED_OVERFIT` — <4/5 CV folds profitable with defaults
- `FAILED_NARROW` — <10 passing instruments
- `FAILED_OOS` — passed training, failed out-of-sample
- `FAILED_TIMEOUT` — hit safety timeout
- `FAILED_ERROR` — uncaught exception

## Crash Resilience
- Each strategy's folder saved completely before next strategy starts
- `--phase N` flag: restart from any phase, load prior results from disk
- `--strategy NAME` flag: run one strategy end-to-end
- `--dry-run`: parse everything, run nothing

## Decisions Made
- Validate thesis BEFORE optimizing (Phase 3a before 3b) — this is Chan's core principle
- Optimize thresholds only, never indicator periods — prevents overfitting and allows indicator pre-computation
- Sharpe as primary metric everywhere, including Optuna objective
- Win rate deliberately excluded from filtering — valid strategies can have 25% win rate with 4:1 winner-to-loser
- Data-snooping correction reported alongside each Sharpe (not enforced as a hard filter, but visible)

## Pitfalls & Anti-patterns
- **Optimizing on full training then "validating" on subsets of training**: The original Phase 3 did this — CV test folds were inside the optimization data. Fixed by splitting into 3a (CV with defaults) and 3b (optimize only after thesis proven).
- **Including test data anywhere before Phase 6**: Any code that loads test data before Phase 6 is a bug.
- **Sharpe computed on gross PnL**: When costs are included, Sharpe must use net PnL. Three separate bugs found where gross PnL leaked into Sharpe calculation.

## Corrections Log
- ~~CV was run AFTER optimization on the same data~~ → CV with default params runs BEFORE optimization. Thesis must work without parameter tuning.
- ~~Win rate > 35% was a filter criterion~~ → Removed. Sharpe and profit factor already capture profitability. Win rate filter kills valid low-win-rate strategies.
- ~~12 days of 5-second data uses 5-fold expanding window CV~~ → With only 12 days, use leave-one-day-out (train 11, test 1, rotate 12 times). Must pass ≥9 of 12.
