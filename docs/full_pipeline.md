# Full Pipeline: Optimize → CV → Sensitivity

## Overview

The full pipeline runs four stages per strategy, saving all intermediate results to `result_store/` (SQLite + Parquet) so the dashboard can display comprehensive analysis.

## Usage

```bash
# Full pipeline — all 291 strategies, 9 threads, ~2-4 hours on M4
.venv/bin/python -m pipeline.run_all --full --parallel-strategies 9

# Single strategy (for debugging)
.venv/bin/python -m pipeline.run_all --full --strategy vwap_mean_reversion_v18 --parallel-strategies 1

# Quick mode — default params only, save to store
.venv/bin/python -m pipeline.run_all --save --parallel-strategies 9

# Dashboard — view results after pipeline completes
.venv/bin/python -m pipeline.run_all --dashboard

# Audit — check result store integrity
.venv/bin/python -m pipeline.run_all --audit
```

## Stages Per Strategy (--full)

### Stage 1: Default Backtest
- Runs backtest with strategy's default parameters on full dataset (12 trading days)
- Saves 1 run to ResultStore with `notes="default_params_full"`, `split="full"`
- If zero trades → strategy is skipped (verdict: `ZERO_TRADES`)

### Stage 2: Optuna Optimization
- **Prerequisite**: ≥50 trades in default backtest and strategy has tunable params
- Runs 100 Optuna trials (TPE sampler, seed=42, 30min timeout per strategy)
- Each trial: suggests parameter values in [low, high] bounds → backtests → returns Sharpe
- Save callback: each completed trial is saved to ResultStore with `optuna_study_id` and `optuna_trial_number`
- After optimization: runs full backtest with best params and saves with `notes="optimized_params_full"`
- **Runs saved**: ~100 (Optuna trials) + 1 (optimized full)

### Stage 3: Leave-One-Day-Out Cross-Validation
- 12 folds (one per trading day)
- Per fold: backtest on single test day using best params (optimized or default)
- Each fold saved with `split="test"`, `fold_number=N`, `split_type="leave_one_out"`
- CV passes if ≥9 of 12 days are profitable
- **Runs saved**: up to 12

### Stage 4: Sensitivity Analysis (±20%)
- **Prerequisite**: ≥10 trades and strategy has tunable params
- For each parameter: perturbs ±20%, re-runs backtest
- Verdict: `ROBUST` if no single perturbation causes >50% Sharpe degradation
- Perturbed runs saved automatically with `is_sensitivity_run=True`
- **Runs saved**: ~2 × N_params (minus clamped)

## Total Runs Per Strategy

For a strategy with 3 tunable parameters:
- 1 (default) + 100 (Optuna) + 1 (optimized) + 12 (CV) + 6 (sensitivity) = **~120 runs**
- For 291 strategies: ~35,000 total runs in the store

## Result Store Layout

```
result_store/
├── backtest.db              # SQLite: runs, parameters, features, splits, results, daily_returns
└── trades/
    ├── vwap_mean_reversion_v18/
    │   ├── <run_id_1>.parquet
    │   ├── <run_id_2>.parquet
    │   └── ...
    ├── rsi_connors_pullback_v42/
    │   └── ...
    └── ...
```

## Key Tables in backtest.db

| Table | Description |
|-------|-------------|
| `runs` | One row per backtest run. Fields: run_id, strategy_id, strategy_family, backtest_engine, cost_model_version, optuna_study_id, optuna_trial_number, is_sensitivity_run, notes |
| `parameters` | Parameter values per run. `is_optimized=1` for Optuna-optimized params |
| `results` | Metrics per run: sharpe_raw, sharpe_deflated (DSR), cagr, max_drawdown, win_rate, net_edge_bps, kill_condition_triggered |
| `splits` | Train/test split info per fold |
| `daily_returns` | Daily return series for DSR computation |
| `features` | Feature/signal descriptions per strategy |

## How the Dashboard Uses This Data

- **Leaderboard**: queries `results` table grouped by `strategy_family`, ranks by `sharpe_deflated`
- **Strategy Detail**: shows all runs, parameter comparison, CV fold results, sensitivity verdict
- **Portfolio View**: computes correlation heatmap from `daily_returns`, DSR summary
- **Trade Log**: reads Parquet files from `result_store/trades/`

## Threading Model

- `ThreadPoolExecutor(max_workers=N)` runs N strategies in parallel
- Numba `@njit` releases the GIL → true CPU parallelism on M4's 10 cores
- Data loaded once, shared read-only across threads (Polars DataFrames are immutable)
- SQLite WAL mode handles concurrent writes (each thread creates its own connection)
- Recommended: `--parallel-strategies 9` on M4 10-core (leave 1 for OS)

## File Descriptor Limit

macOS default is 256 open files per process. The pipeline raises this to 4096+ at startup via `resource.setrlimit()`. Without this, parallel strategy loading will fail with `[Errno 24] Too many open files`.

## Config Defaults (pipeline/config.py)

| Setting | Value | Description |
|---------|-------|-------------|
| `OPTUNA_TRIALS` | 100 | Trials per strategy |
| `OPTUNA_TIMEOUT_SECS` | 1800 | 30 min timeout per strategy |
| `MIN_TRADES_FULL` | 50 | Min trades to qualify for optimization |
| `MIN_PROFITABLE_DAYS` | 9 | Min profitable days (of 12) to pass CV |
| `OVERFIT_SHARPE_RATIO` | 3.0 | Flag if optimized/default Sharpe > 3× |
| `CAPITAL_PER_ENTRY` | 500,000 | INR per strategy entry |

## Estimated Runtime

| Hardware | Workers | Estimated Time |
|----------|---------|----------------|
| M4 Air 10-core | 9 | 2-4 hours |
| M4 Air 10-core | 4 | 4-8 hours |
| Single thread | 1 | 12-20 hours |

Per strategy: ~3 min (observed for vwap_mean_reversion_v18 with 3 tunable params).
