# METRICS AUDIT REPORT
## Engineered Alpha Trading Strategy Dashboard

**Audited**: 2026-03-23
**Scope**: All metrics displayed across Dashboard Homepage, Strategy Overlay (Overview, CV Analysis, Features, Backtest, Trade Log tabs), and Portfolio view.
**Data**: 12 NSE trading days, 5-second NIFTY/BANKNIFTY bars, 227 strategies in DB.

---

## Executive Summary

The dashboard is **structurally sound** with correct formulas for the core trading metrics (Sharpe, Win Rate, Net Edge BPS, Max Drawdown, Profit Factor, Avg Hold Time). The cost model is correctly implemented and applied consistently. However, **three bugs and four design issues** require attention before this system can be trusted for strategy selection.

| # | Severity | Issue | Fix Complexity |
|---|----------|-------|----------------|
| B1 | HIGH | BANKNIFTY strategies show 2.17× wrong P&L, Avg Win/Loss, Days Profitable | 5 lines |
| B2 | HIGH | DSR N uses ~2,700 runs instead of ~227 strategies → all DSR values ≈ 0 | 2 lines |
| B3 | MEDIUM | CAGR extrapolation from 12 days produces meaningless 1000%+ values | Display-only |
| B4 | MEDIUM | `optimized_sharpe` in pipeline JSON is avg OOS PnL (INR), not Sharpe | Rename + guard |
| D1 | MEDIUM | Kill condition triggers on 1-day fold runs (Sharpe=0), making `any_kill` unreliable | SQL fix |
| D2 | LOW | CV profitable-day count: pipeline JSON uses PnL > 0, overlay uses net_edge_bps > 0 | Standardize |
| D3 | LOW | Drawdown understated when strategy opens with losing streak (cumsum starts at first trade, not 0) | 1 line |
| D4 | LOW | SSOT: lot sizes and cost rates defined in both config.py and cost_model.py | Consolidate |

---

## Section 1: What Is Correct ✓

### 1.1 Core P&L Chain

The P&L computation is internally consistent end-to-end:

```
entry_premium + exit_premium (from option parquet, candle close)
    → net_pnl_quick(ep, xp, lot_size, capital=500_000)
    → lots = floor(500_000 / (ep × lot_size)), min 1
    → gross = (xp - ep) × qty
    → costs = STT(0.15% × sell_to) + brokerage(₹40) + exchange_txn(0.03553% × total_to)
              + SEBI(0.0001% × total_to) + stamp(0.003% × buy_to) + IPFT(₹0.50/lakh × total_to)
              + GST(18% × (brokerage + exchange_txn + IPFT))
    → net_pnl = gross - costs
```

This is the **single authoritative cost computation** used by:
- `pipeline/metrics.py` — for Sharpe, drawdown, win rate
- `pipeline/result_store.py` — for stored daily returns and BPS
- `pipeline/dashboard_server.py` (trade log endpoint) — for per-trade display
- `scripts/backfill_optimized_full.py` — for recovered runs

All paths use `cost_model.net_pnl_quick()`. No divergence found.

### 1.2 Cost Model Rates (Upstox, post-Apr 2025)

| Component | Rate | Basis | Verified |
|-----------|------|-------|----------|
| STT | 0.15% | Sell-side turnover | ✓ |
| Brokerage | ₹20 flat × 2 | Per order (entry + exit) | ✓ |
| Exchange txn | 0.03553% | Total turnover (both sides) | ✓ |
| SEBI fee | 0.0001% | Total turnover | ✓ |
| Stamp duty | 0.003% | Buy-side turnover | ✓ |
| IPFT | ₹0.50/lakh | Total turnover | ✓ |
| GST | 18% | On brokerage + exchange + IPFT | ✓ |
| Spread | 0 | Limit order model, no bid-ask assumed | Documented |

### 1.3 Sharpe Ratio Formula

`Sharpe = mean(R_daily) / std(R_daily, ddof=1) × √252`

- Returns = net PnL per day / ₹5L capital
- Zero-filled for non-trading days within the data window
- `total_days` = actual unique day_ids in spot_df (not hardcoded 252)
- Consistent between Optuna objective (`compute_sharpe_from_trades`) and stored metric (`compute_metrics`)

### 1.4 BPS Display

Gross, Cost, Net Edge BPS all computed per-trade and averaged:
- Gross BPS = `(xp - ep) / ep × 10,000`
- Cost BPS = `total_cost / (ep × qty) × 10,000`
- Net BPS = Gross - Cost (correctly, no spread)

### 1.5 Run Selection Priority

The dashboard correctly prefers `optimized_full` runs:
- Overview tab: `runList.find(r => r.notes === 'optimized_full') || bestSharpe fallback`
- Trade Log: same
- Features & Params: `next((r for r in runs if notes == 'optimized_full'), None)` or latest
- Portfolio: `strategy_optimized_run.get(sid, latest)`
- Sensitivity baseline: `optimized_full` preferred over `default_params_full`
- Leaderboard tier metrics: from DB `optimized_full` run via SQL join

### 1.6 Nested LOO-CV Design

The CV design is correct:
- For each of 12 held-out days: optimize on other 11 → test on held-out day only
- No leakage: train data never includes test day in signal computation (full 12-day backtest run with fold params, then filtered to test-day entries only — indicators see all 12 days which is fine; the ENTRIES are filtered to the test day)
- Consensus params = element-wise median of 12 fold-optimal param dicts
- `optimized_full` run = full 12-day backtest with consensus params → this is what the dashboard shows

---

## Section 2: Bugs — Must Fix

### Bug B1 (HIGH): BANKNIFTY lot size hardcoded to 65

**File**: `pipeline/dashboard_server.py:360`
**Code**: `lot_size = NIFTY_LOT  # TODO: resolve per underlying`

**Affected metrics shown in Overview tab**:
- Total P&L
- Avg Winner / Avg Loser
- Profit Factor
- Profitable Days (sign of daily PnL could flip)

**Scale of error**: BANKNIFTY lot = 30, NIFTY lot = 65. Error factor = 65/30 = 2.17×. A BANKNIFTY strategy showing Total P&L of ₹2,17,000 actually made ₹1,00,000.

**NOT affected** (computed correctly at save time with right lot_size):
- Sharpe, DSR, Net Edge BPS, Win Rate, Max Drawdown, Trades/Day, Avg Hold, Avg Daily Return %

**Fix** (5 lines):
```python
# In get_strategy_runs():
# After loading trades_df for a run, resolve lot_size from the strategy's underlying
strategy_obj = _load_strategy_class(strategy_id)  # or look up from run record
underlying = (strategy_obj.underlying.upper() if strategy_obj else "NIFTY")
lot_size = BANKNIFTY_LOT if "BANK" in underlying else NIFTY_LOT
```

Or simpler: query the `runs` table for the strategy's data_source or look up any stored feature that encodes underlying.

---

### Bug B2 (HIGH): DSR N = total runs, not total strategies

**File**: `pipeline/result_store.py:382`
**Code**: `n_current = self.count_total_runs()`

**Impact**: With ~2,700 total runs (227 strategies × ~12 runs each), the DSR benchmark `E[max SR | N, T]` with N=2700 and T=12 returns approximately SR* ≈ 5+ annualized. Almost no strategy's Sharpe exceeds this, so `passes_dsr = False` for nearly everything and `sharpe_deflated ≈ 0`. The DSR is essentially disabled.

**Correct**: N should count independent strategy trials. Each strategy is one trial regardless of how many runs (sensitivity, CV folds) it has. N = 227.

**Fix** (2 lines):
```python
# result_store.py:382
n_strategies = len(self.get_strategy_id_list())  # SELECT COUNT(DISTINCT strategy_id) FROM runs
n_total = max(n_strategies, 1)
dsr_result = compute_dsr(daily_returns, n_total)
```

Also need to recompute stored `sharpe_deflated` for all existing runs after fixing N.

---

### Bug B3 (MEDIUM): CAGR extrapolation from 12 days is misleading

**File**: `pipeline/result_store.py:952`
**Code**: `cagr = (1 + total_return) ** (252.0 / n_days) - 1.0`

**Example**: Strategy with Sharpe 5, 30 trades, net PnL ₹50,000 over 12 days.
Total return = 50,000 / 500,000 = 10%. CAGR = (1.10)^21 - 1 = 672%.

**Fix**: Don't display CAGR when n_days < 60. Add label "extrapolated — unreliable at 12 days" if shown. Never rank strategies by CAGR on this dataset.

---

### Bug B4 (MEDIUM): `optimized_sharpe` in pipeline JSON is NOT a Sharpe ratio

**File**: `pipeline/run_all.py:601-602`
**Code**:
```python
fold_pnls = [d["pnl"] for d in cv_fold_results if d["trades"] > 0]
if fold_pnls:
    summary["optimized_sharpe"] = round(sum(fold_pnls) / max(len(fold_pnls), 1), 2)
```

**Actual value**: Average NET PnL in INR across held-out test days. Could be ₹2,500 per day, displayed as "2500.00" — not a Sharpe ratio.

**Impact**: The `/api/nested-cv` endpoint returns this as `optimized_sharpe`. If any display reads from that endpoint and shows it as "Sharpe", it would display ₹ amounts as dimensionless Sharpe values. The `/api/leaderboard` endpoint correctly overrides this with the DB value. Currently safe in leaderboard/tier cards.

**Fix**: Rename to `avg_oos_pnl_inr` in the JSON and the nested-cv endpoint. Add a note: "This is average daily OOS PnL in INR, not Sharpe."

---

## Section 3: Design Issues — Should Fix

### D1 (MEDIUM): Kill condition fires on 1-day fold runs

**File**: `pipeline/result_store.py:404-407`
**Code**: `kill = sharpe < KILL_SHARPE_THRESHOLD` applied to ALL runs including 1-day CV folds

With T=1 day, Sharpe computation produces `std=0` → Sharpe=0.0 → kill triggered.
Result: `any_kill = 1` for nearly every strategy in the leaderboard SQL, making the kill flag useless.

**Fix**: Only trigger kill for `split='full'` runs with `total_trading_days >= 10`. Better: only for `notes = 'optimized_full'` runs.

---

### D2 (LOW): CV profitable-day count inconsistency

**Pipeline JSON** (`run_all.py:587`): `profitable = day_pnl > 0` (total net PnL for the day > 0)
**Overlay CV tab** (`dashboard_server.py:763`): `n_profitable = sum(1 for f in folds if net_edge_bps > 0)` (positive edge per fold)

These can differ. Example: a fold with 3 trades, all with positive edge (net_edge_bps > 0) but one large loser could make total_pnl < 0.

**Fix**: Use `total_pnl > 0` in both places (the pipeline JSON source is authoritative).

---

### D3 (LOW): Max drawdown understated at sequence start

**Formula**: Drawdown is measured from the running maximum of the cumulative net PnL series starting at the first trade. If trades start with losses, the first cumulative value is negative but the running maximum is also negative (same value), so drawdown at that point is 0.

**Correct**: The sequence should start from 0 (before any trades). The drawdown should be measured from the peak of `[0] + cumsum(net_pnls)`.

**Current behavior example**: `net_pnls = [-100, -200, +500]`
- Current: `cum = [-100, -300, 200]`, `running_max = [-100, -100, 200]`, `drawdowns = [0, 200, 0]`, max DD = 200
- Correct: `cum = [0, -100, -300, 200]`, `running_max = [0, 0, 0, 200]`, `drawdowns = [0, 100, 300, 0]`, max DD = 300

**Fix**: Prepend 0 to sorted net PnL array before cumsum.

---

### D4 (LOW): SSOT violation — constants duplicated

`NIFTY_LOT`, `STT_RATE`, `BROKERAGE_PER_ORDER`, `EXCHANGE_TXN_RATE`, `CAPITAL_PER_ENTRY` defined in both `config.py` and `cost_model.py`. Currently consistent, but any change to one without the other silently diverges.

**Fix**: `config.py` should import from `cost_model.py` for the shared constants, not re-define them.

---

## Section 4: Cross-Consistency Checks

| Identity | Result |
|----------|--------|
| `net_edge_bps > 0` ↔ `total_pnl > 0` (for single-trade days) | PASS — both derived from same `net_pnl_quick()` |
| `sharpe > 0` ↔ `avg_daily_return > 0` | PASS — same formula, same daily returns |
| `win_rate × n_trades` = winners count | PASS — `win_rate = len(winners) / n_trades` |
| `avg_winner × winners - avg_loser × losers = total_pnl` | PASS — algebraically consistent |
| `profit_factor` = `sum(winners) / abs(sum(losers))` | PASS |
| `max_drawdown >= max(-net_pnl)` (single trade can cause max DD) | PASS — cumsum captures this |
| `trades_per_day × days_with_trades ≈ total_trades` | PASS — `trades_per_day = total_trades / total_trading_days`, not days_with_trades. So `trades_per_day × 12 = total_trades`. Different from `total_trades / days_with_trades`. |

**Note on last check**: `trades_per_day` uses `total_trading_days = 12` in the denominator (all days in dataset), not just days when trades occurred. So for a strategy trading 30 times on 10 days, `trades_per_day = 30/12 = 2.5`, not 30/10 = 3. This is the correct interpretation (average over all available days) for comparing across strategies.

---

## Section 5: What Cannot Be Verified Without Running the System

| Check | Why Not Verifiable Statically |
|-------|-------------------------------|
| IST timestamp correctness | Need to inspect actual parquet file timestamps |
| Zero-premium entry guard (state machine) | `state_machine.py` not audited (Numba compiled) |
| Strategy-level look-ahead in `compute()` | Per-strategy signal computation |
| Actual fill at candle close (vs open) | Depends on state machine implementation |
| Option premium staleness (missing/zero premiums) | Requires data quality check on parquet |
| Autocorrelation adjustment in DSR | Paper's Eq. 3 includes this term; code omits it |

---

## Section 6: Priority Fix Plan

### P0 — Fix before using for any strategy selection decision:

1. **Bug B2**: Fix DSR N. Change `count_total_runs()` → `count_distinct_strategies()`. Recompute all stored `sharpe_deflated`.
2. **Bug B1**: Fix BANKNIFTY lot size in runs endpoint. Add `lot_size = BANKNIFTY_LOT if "BANK" in strategy_underlying else NIFTY_LOT`.

### P1 — Fix before sharing with anyone externally:

3. **Bug B3**: Suppress CAGR display or add "NOT MEANINGFUL at 12 days" label.
4. **Bug B4**: Rename `optimized_sharpe` in pipeline JSON to `avg_oos_pnl_inr`.
5. **D3**: Fix drawdown to start from 0.

### P2 — Fix when time permits:

6. **D1**: Kill condition only for `split='full'` and `notes='optimized_full'` runs.
7. **D2**: Standardize CV profitable-day count to use `total_pnl > 0`.
8. **D4**: Remove duplicate constants from config.py.

---

## Section 7: Statistical Context (Non-Bug)

These are not bugs but fundamental limitations to be aware of:

1. **12 trading days is statistically insufficient.** Standard error of Sharpe estimate ≈ 1 Sharpe unit with T=12. Any Sharpe between roughly [-2, +2] is indistinguishable from zero at 95% confidence. Only extreme Sharpe values (>5) are credibly positive.

2. **The "Promising" tier (6 strategies passing LOO-CV) is meaningful** but only proves the strategy wasn't random over 12 specific days. Out-of-sample performance requires a new unseen data period.

3. **All 12 trading days are from 2026-03-04 → 2026-03-19.** This is a single market regime. Strategies may be fitting to a specific trending/volatile pattern of that 2-week window.

4. **Sensitivity analysis with ±20% perturbation on 12 days** provides weak robustness evidence. It shows parameter stability within a 40% range, but market-regime sensitivity is not tested.

---

## Appendix: File Locations

| File | Purpose |
|------|---------|
| `pipeline/cost_model.py` | Authoritative cost computation + lot sizes |
| `pipeline/metrics.py` | Sharpe, drawdown, win rate, hold time |
| `pipeline/dsr.py` | DSR / PSR computation |
| `pipeline/result_store.py` | DB schema, save/load, BPS, CAGR, kill condition |
| `pipeline/run_all.py` | Backtest orchestration, CV pipeline, consensus params |
| `pipeline/dashboard_server.py` | API endpoints, trade-level stat recomputation |
| `dashboard/static/overlay.js` | Strategy detail display logic |
| `dashboard/static/leaderboard.js` | Tier classification, card rendering |
| `outputs/full_pipeline_results.json` | CV/sensitivity verdicts, default_sharpe, avg_oos_pnl (mislabeled optimized_sharpe) |
| `result_store/backtest.db` | SQLite: all stored runs, results, daily_returns |
