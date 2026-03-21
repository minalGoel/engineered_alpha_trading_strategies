# Backtest Audit Configuration

## Audit Checklist
The pre-backtest audit checklist is at `./backtest-audit-prompt.md` in this repo root.

## Subagent Dispatch Rules
When running a backtest audit, always use the Task tool to spawn these PARALLEL subagents:

- **data_agent**: Audit sections 1 (Data Integrity) and 2 (Look-ahead Bias). Read all data loading, preprocessing, and signal generation code. Read-only — no file edits.
- **overfitting_agent**: Audit section 3 (Data Snooping/Overfitting) and section 3.7 (ML checks). Count ALL free parameters. Read-only.
- **cost_fill_agent**: Audit sections 4 (Transaction Costs) and 5 (Fill Model). Verify against NSE cost structure: brokerage, STT, exchange fees, SEBI turnover fee, GST, stamp duty. Read-only.
- **regime_capacity_agent**: Audit sections 6 (Regime Change) and 7 (Capacity/Scaling). Check NSE structural breaks. Read-only.

## Rules
- Every checkbox in the audit must be answered explicitly: [PASS], [FAIL], or [UNKNOWN — reason]
- Never skip checkboxes or summarize sections
- HARD KILLs must include exact file name + line number
- Do not edit any source files
