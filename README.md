# Engineered Alpha Trading Strategies

A research and backtesting framework for intraday Indian options strategies on 5-second data, with a reproducible optimization pipeline, robustness checks, and a local dashboard for analysis.

## Why This Repository Exists

This repository is a public showcase of systematic strategy research work:
- End-to-end pipeline from raw strategy definitions to ranked outputs
- Strict validation workflow (nested leave-one-day-out CV)
- Robustness checks (sensitivity testing, deflated Sharpe tracking)
- Practical artifacts (trade logs, summaries, dashboard views)

## Core Features

- 5-second intraday strategy backtesting
- Batch strategy evaluation and filtering
- Nested CV optimization with per-fold out-of-sample testing
- Deflated Sharpe Ratio (DSR) and stability-focused diagnostics
- Result store (SQLite + Parquet) for reproducibility
- Lightweight dashboard for local exploration

## Repository Structure

- `pipeline/` - Main backtest, optimization, metrics, storage, and dashboard code
- `pipeline/strategies/` - Strategy implementations used by the pipeline
- `unique_strategies/` - Strategy definition/config artifacts
- `result_store/` - Persistent run metadata and trade outputs
- `outputs/` - Pipeline summaries/checkpoints/log artifacts
- `docs/` - Architecture and system design documents
- `tests/` - Test suite
- `dashboard/` - Dashboard assets/setup helpers

## Quick Start

```bash
# 1) Create and activate environment
python3 -m venv .venv
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run a quick single-pass pipeline
.venv/bin/python -m pipeline.run_all --save --parallel-strategies 9
```

## Common Commands

```bash
# Full nested-CV run (all strategies)
.venv/bin/python -m pipeline.run_all --full --parallel-strategies 9

# Resume from checkpoint
.venv/bin/python -m pipeline.run_all --full --parallel-strategies 9 --resume

# Run one strategy through full pipeline
.venv/bin/python -m pipeline.run_all --full --strategy vwap_mean_reversion_v18

# Open dashboard
.venv/bin/python -m pipeline.run_all --dashboard

# Audit stored outputs
.venv/bin/python -m pipeline.run_all --audit
```

For a deeper walkthrough, see `FULL_PIPELINE.md`.

## Validation Approach

The full pipeline uses nested leave-one-day-out cross-validation to reduce parameter leakage risk:
- Train/optimize on 11 days
- Test on the held-out day
- Repeat across all folds
- Then run parameter sensitivity checks

This design emphasizes out-of-sample behavior and robustness over in-sample optimization.

## Security Note (Public Repo)

- No real API keys, session cookies, or private keys are intended to be stored in this repository.
- Secrets should be supplied through environment variables or external secret management.
- If you discover sensitive data, please report it privately and do not open a public issue with the secret contents.

## Disclaimer

This project is for research and educational purposes. Backtest results are not guarantees of live trading performance. Trading involves substantial risk.
