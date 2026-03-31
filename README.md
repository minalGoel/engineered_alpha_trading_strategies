# 1-Min Spot Strategies Pipeline

This branch (`1min_spot_strategies`) contains a full research/backtest pipeline for **1-minute spot-based strategy generation and validation**.

It includes:
- Strategy parsing and validation
- Multi-phase backtesting pipeline
- Leaderboard and winning-strategy generation
- EC2-run outputs that were downloaded back into this repo under `ec2_artifacts/`

## Project Layout

- `pipeline/` - core backtest/validation pipeline
- `trading_strategies/` - strategy definitions used by parser
- `data/` - parquet market datasets (`stocks_1min.parquet`, index and VIX data)
- `strategy_results/` - canonical local per-strategy outputs used by code
- `outputs/` - canonical aggregate outputs (`leaderboard.json`, `winning_strategies.json`, etc.)
- `ec2_artifacts/` - artifacts downloaded from EC2 training/backtest runs
- `scripts/` - run and operations helpers

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Pipeline

```bash
# Full run
python pipeline/run_all.py --parallel-strategies 8 --cores-per-strategy 4

# Single strategy smoke run
./scripts/run_local.sh vwap_mean_reversion_basic_v1

# Parse-only dry run
python pipeline/run_all.py --dry-run
```

## EC2 Artifacts: Wiring Into Local Codebase

Backtests were trained/run on EC2, and downloaded into:

`ec2_artifacts/`

The pipeline code expects local canonical paths (`outputs/`, `strategy_results/`, and logs).  
Use the sync helper to wire EC2 artifacts into those locations:

```bash
# Preview only
./scripts/sync_ec2_artifacts.sh --dry-run

# Sync from default ./ec2_artifacts
./scripts/sync_ec2_artifacts.sh

# Sync from a custom artifacts folder
./scripts/sync_ec2_artifacts.sh --source /path/to/ec2_artifacts
```

### What gets mapped

- `ec2_artifacts/outputs` -> `outputs/`
- `ec2_artifacts/pipeline_artifacts/strategy_results` (preferred) or `ec2_artifacts/strategy_results` -> `strategy_results/`
- `ec2_artifacts/*.log` and related files -> `logs/ec2/`

After sync, existing pipeline/analysis code can read outputs without path changes.

## Spin Up Dashboard

```bash
# Serves outputs/winning_strategies_dashboard.html on http://127.0.0.1:8765
./scripts/start_dashboard.sh

# Optional custom port
./scripts/start_dashboard.sh 9000
```

## Key Output Files

- `outputs/leaderboard.json`
- `outputs/leaderboard.md`
- `outputs/winning_strategies.json`
- `outputs/family_summary.json`
- `outputs/master_signals.parquet`
- `outputs/pipeline_log.json`

## Notes

- Default pipeline constants are configured in `pipeline/config.py`.
- `pipeline.log` captures run logs in repo root.
- This repository is intended as a showcase and research artifact, not financial advice.
