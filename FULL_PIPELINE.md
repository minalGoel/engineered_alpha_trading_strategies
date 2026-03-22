# Full Backtest Pipeline — Reference Guide

## Quick Reference

```bash
# First run (all 291 strategies, full nested CV pipeline)
.venv/bin/python -m pipeline.run_all --full --parallel-strategies 9

# Crashed at strategy 150? Resume from checkpoint:
.venv/bin/python -m pipeline.run_all --full --parallel-strategies 9 --resume

# Quick single-pass (default params only, ~5 min):
.venv/bin/python -m pipeline.run_all --save --parallel-strategies 9

# Run only one strategy through full pipeline:
.venv/bin/python -m pipeline.run_all --full --strategy vwap_mean_reversion_v18

# Launch the results dashboard:
.venv/bin/python -m pipeline.run_all --dashboard

# Audit what's stored:
.venv/bin/python -m pipeline.run_all --audit
```

---

## Validation Method: Nested Leave-One-Day-Out Cross-Validation

The pipeline uses **nested LOO-CV** to eliminate parameter leakage. For each of 12
trading days, parameters are optimized on the *other 11 days only*, then tested on the
held-out day. This ensures no test data is ever seen during optimization.

### What `--full` Does Per Strategy

Each of 291 strategies passes through 4 sequential stages:

### Stage 1 — Default Params Full Backtest (baseline)
- Runs `backtest_strategy()` with the strategy's hardcoded default parameters on all 12 days
- Saved to ResultStore with `notes="default_params_full"`, `split="full"`
- **Gate**: if `total_trades == 0` → stops here, verdict = `ZERO_TRADES`

### Stage 2+3 — Nested LOO-CV (optimization + out-of-sample test, merged)
For each of the 12 folds (held-out day):
1. **Split**: train = 11 days, test = 1 held-out day
2. **Train viability check**: run default backtest on train split; need `MIN_TRADES_FULL=50` trades
3. **Optimize on TRAIN only**: 50 Optuna trials (`OPTUNA_TRIALS_PER_FOLD`), 2-min timeout per fold
   - TPE sampler, seed=42
   - No individual trial persistence (`save_trials=False`) — avoids storing 7,200 train-fold trials
4. **Test on held-out day**: run full 12-day backtest with fold-optimal params (indicators warm up
   causally from all days), then **filter to only trades entering on the test day**
5. **Save fold**: each fold saved to ResultStore with `notes="nested_cv_fold_N"`, `split="test"`,
   `split_type="nested_loo"`

**CV pass condition**: profitable on >= 9 of 12 held-out days (`MIN_PROFITABLE_DAYS=9`)

Total Optuna trials per strategy: 12 folds x 50 = **600 trials** (but each only sees 11 days)

### Stage 4 — Sensitivity Analysis (Chan's +/-20% test)
- **Gate**: skipped if `total_trades < 10` or no tunable params
- Uses **consensus parameters**: element-wise median of the 12 fold-optimized param sets
- Each tunable parameter is perturbed +/-20% independently; strategy re-backtested
- Results saved to ResultStore with `is_sensitivity_run=True`
- Verdict: `ROBUST` if no single perturbation drops Sharpe by > 50%; `FRAGILE` otherwise

---

## Why Nested CV (Not Simple LOO-CV)

The original pipeline had **parameter leakage**: Optuna optimized on all 12 days, then
LOO-CV "validated" on those same 12 days. The optimizer had already seen the test data.

Nested CV fixes this:
- **Outer loop** (12 folds): splits data into train/test
- **Inner loop** (50 Optuna trials): optimizes only on train
- **Test**: fold-optimal params evaluated on unseen held-out day

This is the gold standard for small-sample validation (Chan 2013, Bailey & de Prado 2016).

---

## Storage Layout

```
result_store/
├── backtest.db                          SQLite: all metadata tables
│   ├── runs                             one row per backtest run
│   ├── parameters                       strategy param values per run
│   ├── features                         feature names per run
│   ├── splits                           fold/split definitions
│   ├── results                          metrics per run x split
│   └── daily_returns                    per-day return arrays (for DSR)
└── trades/
    └── <strategy_id>/
        └── <run_id>.parquet             all trades for this run
outputs/
├── full_pipeline_checkpoint.json        progress checkpoint (for --resume)
├── full_pipeline_results.json           per-strategy summary after run
└── pipeline_log.json                    run counts by verdict
```

### ResultStore Run Notes — What Each Means
| `notes` value | What it is |
|---|---|
| `default_params_full` | Stage 1: default params, all 12 days |
| `nested_cv_fold_N` | Stage 2+3: fold N — optimized on 11 days, tested on held-out day |
| `sensitivity:param_name:+20%` | Stage 4: perturbed run using consensus params |

---

## Checkpoint / Resume

After **every strategy** completes, its summary is atomically written to:
```
outputs/full_pipeline_checkpoint.json
```

If the run crashes or is killed, restart with `--resume` to skip already-done strategies:
```bash
.venv/bin/python -m pipeline.run_all --full --parallel-strategies 9 --resume
```

The checkpoint maps `strategy_id -> summary_dict`. It is never deleted automatically.
To start fresh (re-run everything), delete the checkpoint:
```bash
rm outputs/full_pipeline_checkpoint.json
```

---

## Parallelism Model

| Setting | Value |
|---|---|
| `--full` mode | `ProcessPoolExecutor` with `mp.get_context("spawn")` |
| `--save` mode | `ThreadPoolExecutor` (fast enough for single-pass) |
| Why processes for --full | NumPy holds the GIL during `strategy.compute()`; separate processes = true CPU parallelism |
| Per-process memory | ~50MB (data loaded from disk independently per worker) |
| 9 processes memory | ~450MB total (trivial on 16GB M4 with swap) |
| Worker init | Each process: load Parquet data + warm Numba JIT (~3s startup) |
| Recommended workers | `--parallel-strategies 9` (10-core M4, leave 1 for OS) |
| `NUMBA_NUM_THREADS=1` | Set at startup to prevent Numba from spawning per-thread pools |
| SQLite WAL mode | Enabled — concurrent reads + serialised writes without blocking |
| File descriptor limit | Raised to 4096 at startup (macOS default 256 is insufficient) |
| Checkpoint writes | Main process only (after each `as_completed` future resolves) |

---

## Key Configuration (`pipeline/config.py`)

| Constant | Value | Meaning |
|---|---|---|
| `OPTUNA_TRIALS_PER_FOLD` | 50 | Optuna trials per CV fold (12 x 50 = 600 total) |
| `OPTUNA_FOLD_TIMEOUT_SECS` | 120 | Hard timeout per fold optimization (2 min) |
| `OPTUNA_TIMEOUT_SECS` | 1800 | Safety timeout for non-nested optimization |
| `MIN_TRADES_FULL` | 50 | Min trades on train split to run optimization |
| `MIN_PROFITABLE_DAYS` | 9 | CV pass threshold (9 of 12 held-out days) |
| `TOTAL_TRADING_DAYS` | 12 | Days in the dataset |
| `OVERFIT_SHARPE_RATIO` | 3.0 | Flag if optimized/default Sharpe > 3x |
| `CAPITAL_PER_ENTRY` | 500,000 | INR per strategy entry order |
| `STT_RATE` | 0.0015 | 0.15% on sell-side premium (from 1 Apr 2025) |

---

## DSR (Deflated Sharpe Ratio)

Every stored result has `sharpe_deflated` — the probability that the strategy's Sharpe
is genuinely above the expected maximum from a random search over N strategies.

Formula: Bailey & de Prado (2016).
- Benchmark: `E[max SR | N_eff, T]` where N_eff = matrix rank of return correlations
- `sharpe_deflated` close to 1.0 = genuine edge; close to 0 = likely overfitting

---

## Average Hold Time

`avg_hold_seconds` is computed for every run in `pipeline/metrics.py` as:
```
avg_holding_bars x 5  (1 bar = 5 seconds)
```

Stored in the `results` table as `avg_hold_seconds`. Displayed in the dashboard
Trade Log (converted to minutes) and in the Overview cost breakdown. Use this to
assess slippage exposure: strategies with avg hold < 60 seconds are extremely
sensitive to spread and fill model assumptions.

---

## Expected Run Times (M4 MacBook Air, 9 processes)

| Mode | Time |
|---|---|
| `--save` (default params only) | ~6 minutes |
| `--full` (nested CV, 600 Optuna trials/strategy) | ~4 hours |
| Single strategy `--full` | ~7-8 minutes |

The dominant cost is per-fold Optuna optimization (50 trials x 12 folds = 600 per strategy).
Strategies without tunable params or with < 50 trades skip optimization and complete in
~10 seconds each.

---

## Dashboard

After a run, start the results dashboard:
```bash
.venv/bin/python -m pipeline.run_all --dashboard
# -> http://127.0.0.1:8765
```

First-time setup (downloads Chart.js for offline use):
```bash
.venv/bin/python dashboard/setup_dashboard.py
```

---

## Known Issues / Watchouts

1. **Strategies with no tunable params** skip per-fold optimization and Stage 4 sensitivity.
   CV still runs (each fold uses default params). These are flagged in the summary.

2. **CV on 1 trade** — some strategies produce only 1-2 trades per day. Fold PnL
   is binary (win/loss). Treat CV pass/fail as directional signal, not statistical proof.

3. **`Cannot load strategy class` warnings** — a handful of entries in
   `strategy_retriage.json` reference strategies that don't have a `.py` file yet
   (stubs or planned). These are safely skipped.

4. **`RuntimeWarning: divide by zero`** in several strategies — numpy divide-by-zero
   in indicator calculations (volume ratios etc.). The `np.where` guard catches these;
   the warning is cosmetic and does not affect results.

5. **Fold timeout** — if a fold's Optuna optimization is very slow, the 2-minute per-fold
   timeout (`OPTUNA_FOLD_TIMEOUT_SECS=120`) caps it. The best trial found so far is used.

6. **Consensus params for sensitivity** — Stage 4 uses element-wise median of the 12
   fold-optimized parameter sets. This avoids choosing any single fold's params and
   gives a representative "average" parameter set for robustness testing.

7. **ProcessPoolExecutor spawn** — each worker loads data independently from disk.
   If data files change between the parent process loading and workers spawning,
   results may be inconsistent. Don't modify Parquet files during a run.
