# ACTIVE_THREADS.md — Current Project State

## Last Updated
2026-03-20 evening IST

## Repository
`~/Coding_Projects/engineered_alpha_trading_strategies/`

## Current State: Strategy Re-Conversion Pipeline BLOCKED

### What Was Happening
Running `auto_one_by_one.sh` — one Opus session per strategy to convert, implement, backtest, and verify each strategy for the 5-second NIFTY/BANKNIFTY options regime.

### The Block
The script's source file lookup is broken. Strategy names in `outputs/strategy_retriage.json` (e.g., `vwap_mean_reversion_v1`) don't match filenames in `trading_strategies/unique_strategies_all/` (e.g., `Strategy_1.json`). All 291 strategies failed with "SOURCE NOT FOUND."

### Fix Needed
The retriage JSON should have a `source_file` field per strategy. The script should use that field directly instead of searching by name. If `source_file` doesn't exist, build a mapping from JSON "name" field → filepath.

A prompt was given to Claude Code to fix this but hasn't been executed yet.

### Current File State
- `trading_strategies/unique_strategies_all/` — 358 original strategy JSONs (Strategy_1.json format). READ ONLY archive.
- `unique_strategies/` — 30 JSONs from the Opus gold-standard session (first 30 conversions). These are high quality.
- `pipeline/strategies/` — 19 Python implementations + base.py + CONVENTIONS.md
- `outputs/strategy_retriage.json` — 291 qualified, 67 discarded
- `logs/completed_strategies.txt` — 291 entries (all marked "completed" as failures — needs clearing)
- `logs/failed_strategies.txt` — 291 entries

### To Resume
1. Fix the source file lookup in `auto_one_by_one.sh` (use `source_file` field from retriage JSON or build name→file mapping)
2. Clear `logs/completed_strategies.txt` and `logs/failed_strategies.txt`
3. Rerun the script

---

## Completed Work

### Equity Pipeline (Complete, Parked)
- 358 strategies generated, deduplicated, implemented, and audited
- Pipeline built: Polars + Numba + Optuna + walk-forward CV
- 29 bugs fixed across 3 audit passes
- Data: 206 stocks × 3 years of 1-min OHLCV
- All committed to git, pushed to origin

### Options Pipeline Refactor (Complete, Working)
- Pipeline refactored for 5-second options: state machine, cost model, strike locking
- 2D premium grid architecture (replaces broken 1D ATM array)
- 14 additional bugs fixed during options audit
- Strike locking verified against raw parquet
- Upstox cost model integrated

### Strategy Triage (Complete)
- First triage (wrong frame): 6 survivors, 0 profitable
- Second triage (correct frame): 291 qualified
- Correct framing established: "spot direction prediction, options as instrument"

### Gold Standard Conversions (Partial — 30 of 291)
- 30 strategies converted with unique mechanisms, reasoned lookbacks, individual stop rationales
- CONVENTIONS.md updated with conversion principles and examples
- These are the quality reference for the batch conversion

---

## Open Decisions

### 1. Model Choice for Batch Conversion
**Options**: Opus ($$$, highest quality) vs Sonnet ($, adequate with fresh-session architecture)
**Analysis done**: Fresh sessions neutralize Sonnet's drift weakness. Sonnet will kill fewer marginal strategies (~10-15% vs Opus ~20-30%). The extra survivors cost ~2 hours of backtest compute but save ~15 hours of Opus time.
**Recommendation**: Sonnet for bulk, Opus audit on the 30-50 winners after backtesting.
**Decision**: User chose Opus for diamond standard. Script configured for Opus.

### 2. How Much Historical Data to Buy
- Current: 12 trading days (insufficient for statistical validation)
- Minimum: 6 months (~125 days) — 25 expiry days, 2+ VIX regimes, 1000+ trades per strategy
- Ideal: 12 months (~250 days) — seasonal coverage, 50 expiry days
- Cost: ₹5-15K from data vendors (TrueData, Global Datafeeds, NSE TBT data)
**Decision**: Pending. Build pipeline on 12 days, buy more data for real validation.

### 3. Execution Infrastructure for Live Trading
- Current: Upstox broker API
- Concern: Upstox/Zerodha latency (~200-500ms) may be too slow for 5-second scalping
- Options: Symphony Fintech (<50ms colo), Dhan (~200ms), direct NSE colo
- SEBI algo trading circular requires exchange approval for automated strategies
**Decision**: Not yet addressed. Comes after backtesting proves edge exists.

---

## Next Steps (In Order)

1. **Fix the source file lookup bug** in `auto_one_by_one.sh`
2. **Clear tracker files** and restart the batch conversion
3. **Let the batch run** — 261 remaining strategies × ~5 min each ≈ 22 hours
4. **Opus audit** on mechanism uniqueness, stop/target distribution, quality spot-check
5. **Run `prompt_4_backtest.md`** — train/test split, optimization, OOS validation
6. **Assess results honestly** — how many strategies show gross edge? How many survive costs?
7. **Buy 6 months of data** if any strategies show promise
8. **Real validation** on 6 months of out-of-sample data

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `outputs/strategy_retriage.json` | Which 291 strategies qualified |
| `outputs/conversion_kills.json` | Strategies killed during conversion |
| `pipeline/strategies/CONVENTIONS.md` | The quality standard for implementations |
| `pipeline/strategies/base.py` | OptionSignals interface |
| `pipeline/cost_model.py` | Upstox cost calculations |
| `pipeline/state_machine.py` | Numba state machine with 2D premium grid |
| `5second_data/SCHEMA.md` | Data schema reference |
| `scripts/auto_one_by_one.sh` | Batch automation script (needs source file fix) |
| `logs/one_by_one_runner.log` | Runner log |
| `logs/per_strategy/*.log` | Per-strategy Opus session logs |
