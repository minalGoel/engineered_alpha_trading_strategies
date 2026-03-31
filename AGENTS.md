# AGENTS.md

## Purpose
This file gives human and AI contributors a fast, reliable workflow for making safe changes in this repository.

## Project Snapshot
- Domain: 1-minute spot strategy research/backtesting pipeline.
- Primary code: `pipeline/`
- Strategy inputs: `trading_strategies/`
- Tests: `tests/test_pipeline.py`
- Canonical outputs consumed by tooling: `outputs/`, `strategy_results/`
- Synced external artifacts: `ec2_artifacts/`

## Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## High-Value Commands
```bash
# Full pipeline
python pipeline/run_all.py --parallel-strategies 8 --cores-per-strategy 4

# Parse-only dry run
python pipeline/run_all.py --dry-run

# Single strategy smoke run
./scripts/run_local.sh vwap_mean_reversion_basic_v1

# Unit tests
python -m pytest tests/test_pipeline.py -v
```

## Change Guardrails
- Keep changes minimal and scoped to the requested task.
- Prefer fixing root causes over adding one-off patches.
- Do not rewrite or delete generated artifacts in `outputs/`, `strategy_results/`, or `logs/` unless explicitly asked.
- Do not move EC2 download artifacts out of `ec2_artifacts/`; use `./scripts/sync_ec2_artifacts.sh` for mapping.
- Avoid large, expensive full-pipeline runs unless required for the request.

## Validation Expectations
- For parser/indicator/state machine edits: run `python -m pytest tests/test_pipeline.py -v`.
- For script changes: run the specific script with a small/smoke input and confirm exit behavior.
- For pipeline orchestration changes: at minimum run `python pipeline/run_all.py --dry-run`.

## File/Code Style
- Follow existing Python style in the touched files.
- Keep functions deterministic where possible; avoid hidden global state.
- Add brief comments only for non-obvious logic.
- Preserve public JSON/output schema unless the task explicitly changes it.

## Operational Safety
- This repository is research-oriented; avoid language that implies financial advice.
- Call out assumptions and risk tradeoffs in summaries when behavior changes.

## Handoff Checklist
- What changed (files + behavior)
- How it was validated (commands run)
- Any assumptions or follow-up risks
