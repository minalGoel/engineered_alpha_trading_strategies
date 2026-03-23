# Audit Document 1: Dependency Map & SSOT Violations

Generated: 2026-03-23
Codebase: engineered_alpha_trading_strategies
Data: 12 trading days, NIFTY/BANKNIFTY 5-second bars, 227 strategies

---

## Part A: Metric Dependency Map

Each displayed metric is traced from display layer → API → computation → raw source.

---

### 1. Sharpe Ratio (Annualized)

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js:109` | `res.sharpe_raw` |
| API | `dashboard_server.py:333` | `row.get("sharpe_raw")` from `compare_runs()` |
| DB read | `result_store.py:496-506` | `SELECT results.sharpe_raw` |
| DB write | `result_store.py:402-403` | `r["sharpe_raw"] = dsr_result["sharpe_raw"]` (DSR overrides if None) |
| Stored in | `result_store.py:957` | `build_result_record()` → `"sharpe_raw": metrics.get("sharpe_annualized")` |
| Computed | `metrics.py:148-154` | `mean_daily / std_daily * sqrt(252)` |
| Input | `metrics.py:127-143` | daily NET PnL aggregated by exit_time date / CAPITAL_PER_ENTRY, zero-filled |
| Net PnL | `metrics.py:55-57`, `cost_model.py:142-171` | `net_pnl_quick(ep, xp, lot_size, capital)` |
| Capital | `cost_model.py:45` | `CAPITAL_PER_ENTRY = 500_000` |

**Formula**: `Sharpe = mean(R_daily) / std(R_daily, ddof=1) * sqrt(252)`
where `R_daily = sum(net_pnl_per_trade_on_day) / 500_000`, zero-filled to 12 days.

---

### 2. DSR (Deflated Sharpe Ratio / sharpe_deflated)

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js:110` | `res.sharpe_deflated` |
| API | `dashboard_server.py:334` | `row.get("sharpe_deflated")` |
| DB write | `result_store.py:401` | `r["sharpe_deflated"] = dsr_result["sharpe_deflated"]` |
| Computed | `result_store.py:384` | `compute_dsr(daily_returns, n_total)` |
| N input | `result_store.py:382-383` | `n_current = self.count_total_runs(); n_total = max(n_current+1, 1)` |
| Count query | `result_store.py:508-512` | `SELECT COUNT(*) FROM runs` |
| Formula | `dsr.py:107-187` | Bailey & de Prado 2016, PSR at E[max SR | N, T] benchmark |

**⚠ SSOT / CORRECTNESS ISSUE — see Part C, Issue #1**

---

### 3. Net Edge BPS

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js:111` | `res.net_edge_bps` |
| API | `dashboard_server.py:335` | `row.get("net_edge_bps")` |
| DB write | `result_store.py:395-397` | from `bps` dict computed in `save_run()` |
| Computed | `result_store.py:80-139` | `_compute_bps_breakdown(trades, lot_size)` |
| Formula | `result_store.py:111-115,138` | `avg(gross_bps_per_trade) - avg(fees_bps_per_trade)` |
| Per-trade gross | `result_store.py:111` | `(gross_pnl / (ep * qty)) * 10000` |
| Per-trade fees | `result_store.py:112` | `(total_cost / (ep * qty)) * 10000` |

**Formula**: Unweighted average across trades of `(exit-entry)/entry * 10_000` minus `costs/(entry*qty) * 10_000`
**Note**: `spread_cost_bps = slippage_cost_bps = impact_cost_bps = 0` (limit order model, no spread assumed).

---

### 4. Win Rate

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js` (row 2) | `res.win_rate` formatted as % |
| API | `dashboard_server.py:337` | `row.get("win_rate")` |
| DB | `results.win_rate` | stored in `build_result_record()` → `metrics.get("win_rate")` |
| Computed | `metrics.py:63-65` | `len(net_pnls[net_pnls > 0]) / n_trades` |

**Formula**: `count(trades where net_pnl > 0) / total_trades`
**Note**: Uses NET PnL for classification. A gross-winning trade that loses after costs is counted as a LOSS. ✓ economically correct.

---

### 5. Max Drawdown

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js:158-159` | `res.max_drawdown` (₹ + % of ₹5L) |
| API | `dashboard_server.py:340` | `row.get("max_drawdown")` |
| DB | `results.max_drawdown` | `metrics.get("max_drawdown")` |
| Computed | `metrics.py:88-105` | sorted by exit_time, cumsum(net_pnl), running_max - cumulative |

**Formula**: `max(running_max(cum_net_pnl) - cum_net_pnl)` over trades sorted by exit_time
**Scope**: Trade-sequence drawdown only. Intraday drawdown within open positions not captured.

---

### 6. Total P&L

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js:154-157` | `res.total_pnl` |
| API | `dashboard_server.py:381` | `summary["total_pnl"] = round(float(np.sum(net_pnls)), 0)` |
| Computed at API time | `dashboard_server.py:360-381` | `net_pnl_quick(ep, xp, lot_size, capital)` per trade from parquet |
| **lot_size used** | `dashboard_server.py:360` | `lot_size = NIFTY_LOT  # TODO: resolve per underlying` |

**⚠ HIGH BUG — BANKNIFTY lot size hardcoded. See Part C, Issue #2**

---

### 7. Profit Factor

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js:160,row3` | `res.profit_factor` |
| API | `dashboard_server.py:384` | `gross_profit / gross_loss` |
| Computed at API time | `dashboard_server.py:365-368` | from net_pnls array |

**Formula**: `sum(net_pnl where net_pnl > 0) / abs(sum(net_pnl where net_pnl < 0))`
Uses net PnL (after costs). The internal variable names `gross_profit`/`gross_loss` are misleading (they contain NET values).

---

### 8. Avg Winner / Avg Loser

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js` (row 3) | `res.avg_winner`, `res.avg_loser` |
| API | `dashboard_server.py:382-383` | `np.mean(winners)`, `np.mean(losers)` |
| Computed at API time | same scope, uses net_pnls |
| **lot_size** | `dashboard_server.py:360` | `NIFTY_LOT` (hardcoded) |

**⚠ HIGH BUG — same BANKNIFTY lot size issue as Total P&L**

---

### 9. Profitable Days

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js:166-168` | `res.days_profitable / res.days_total` |
| API | `dashboard_server.py:376` | `int((daily["day_pnl"] > 0).sum())` |
| Computed at API time | `dashboard_server.py:371-378` | daily group_by on exit_time date |
| **lot_size** | `dashboard_server.py:360` | `NIFTY_LOT` (hardcoded) |

**⚠ HIGH BUG — BANKNIFTY lot size affects daily PnL sign → could flip profitable/unprofitable days**

---

### 10. Avg Daily Return %

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js:151-153` | `res.avg_daily_return_pct` |
| API | `dashboard_server.py:331` | `float(daily_rets.mean() * 100)` |
| Source | `dashboard_server.py:330` | `daily_rets = all_return_series.get(rid)` from `get_all_return_series()` |
| DB | `daily_returns` table | stored by `result_store.py:419` |
| Computed | `result_store.py:142-170` | `_compute_daily_returns()` → net_pnl_quick / CAPITAL_PER_ENTRY |

**Note**: Uses stored daily_returns from DB (computed at save time with correct lot_size). ✓ Not affected by BANKNIFTY bug.

---

### 11. Avg Hold Time

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js:163-164` | `res.avg_hold_seconds` |
| API | `dashboard_server.py:346` | `row.get("avg_hold_seconds")` from DB |
| DB | `results.avg_hold_seconds` | `metrics.get("avg_holding_seconds")` |
| Computed | `metrics.py:80-86` | `mean(holding_bars) * 5` (5 seconds per bar) |

---

### 12. Trades Per Day

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js` (row 4) | `res.trades_per_day` |
| API | `dashboard_server.py:345` | `row.get("trades_per_day")` |
| DB | `results.trades_per_day` | `build_result_record()` line 964 |
| Computed | `result_store.py:946` | `total_trades / total_trading_days` |

**Note**: `total_trading_days` = `spot_df["day_id"].n_unique()` at save time. ✓

---

### 13. CAGR

| Layer | Location | Code |
|-------|----------|------|
| Display | `overlay.js` (Backtest tab dropdown) | `res.cagr` |
| DB | `results.cagr` | `build_result_record()` |
| Computed | `result_store.py:948-952` | `(1 + total_pnl/500000)^(252/n_days) - 1` |

**⚠ MISLEADING — with n_days=12, exponent = 252/12 = 21. A 20% return becomes (1.2)^21 = 5700% CAGR. See Part C, Issue #4**

---

### 14. Leaderboard Tier Classification

| Input | Source | Notes |
|-------|--------|-------|
| `cv_passed` | `full_pipeline_results.json` via `/api/leaderboard` | set in `run_all.py:593` |
| `sensitivity_verdict` | `full_pipeline_results.json` | `run_sensitivity()` result |
| `optimized_sharpe` (tier card) | `dashboard_server.py:184` | from DB `optimized_full` run ✓ |
| `optimized_net_edge` | `dashboard_server.py:185` | from DB `optimized_full` run ✓ |
| `optimized_win_rate` | `dashboard_server.py:187` | from DB `optimized_full` run ✓ |
| `optimized_max_dd` | `dashboard_server.py:188` | from DB `optimized_full` run ✓ |
| `optimized_trades_per_day` | `dashboard_server.py:189` | from DB `optimized_full` run ✓ |

**Note**: The `/api/leaderboard` endpoint CORRECTLY overrides `optimized_sharpe` from the pipeline JSON with the DB value. The pipeline JSON value is the average OOS PnL (INR), not a Sharpe ratio — see Part C Issue #3.

---

### 15. CV Analysis Tab — Per-Fold Results

| Metric | Source | API |
|--------|--------|-----|
| Fold Sharpe | `results.sharpe_raw` for that fold's run | `/api/strategy/{id}/cv-folds:739` |
| Fold Net Edge | `results.net_edge_bps` | same |
| Fold Win Rate | `results.win_rate` | same |
| Fold Trades | `results.total_trades` | same |
| Baseline Sharpe | `optimized_full` run's `results.sharpe_raw` | `/api/strategy/{id}/sensitivity-details:826` |

---

## Part B: SSOT Inventory

### Constants with Multiple Definitions

| Constant | `cost_model.py` | `config.py` | Value |
|----------|----------------|-------------|-------|
| `NIFTY_LOT` | line 25 | line 45 | 65 ✓ same |
| `BANKNIFTY_LOT` | line 26 | — | 30 (only in cost_model) |
| `STT_RATE` | line 37 | line 50 | 0.0015 ✓ same |
| `BROKERAGE_PER_ORDER` | line 36 | line 51 | 20.0 ✓ same |
| `EXCHANGE_TXN_RATE` | line 39 | line 52 | 0.0003553 ✓ same |
| `SEBI_FEE_RATE` | line 40 | — | 0.000001 (only in cost_model) |
| `STAMP_DUTY_RATE` | line 41 | — | 0.00003 (only in cost_model) |
| `IPFT_RATE` | line 42 | — | 0.000005 (only in cost_model) |
| `GST_RATE` | line 43 | — | 0.18 (only in cost_model) |
| `CAPITAL_PER_ENTRY` | line 45 | line 53 | 500_000 ✓ same |

**Import chains:**
- `pipeline/run_all.py` imports `NIFTY_LOT, BANKNIFTY_LOT` from `pipeline.config`
- `pipeline/dashboard_server.py` imports `NIFTY_LOT, CAPITAL_PER_ENTRY` from `pipeline.cost_model`
- `pipeline/metrics.py` imports `CAPITAL_PER_ENTRY` from `pipeline.cost_model`
- `pipeline/result_store.py` imports `NIFTY_LOT, CAPITAL_PER_ENTRY` from `pipeline.config`

**Risk**: If `NIFTY_LOT` or `CAPITAL_PER_ENTRY` is ever changed, only updating one file would cause silent divergence. Currently both files have consistent values.

---

### Exchange Transaction Rate Comment Mismatch

- `cost_model.py` header (line 12): *"Exchange Txn (NSE): 0.03553% on premium turnover (both sides)"* — references "1 Oct 2024"
- `config.py` comment (line 52): *"0.03553% on premium turnover (both sides, NSE Mar 2026)"*
- Same numeric value, different date references. Minor documentation inconsistency.

---

## Part C: Issues Found

### Issue #1 — CRITICAL: DSR N uses total runs, not total strategies

**Location**: `result_store.py:382`, `dashboard_server.py` (all DSR calls use `store.count_total_runs()`)
**Current**: `n_total = store.count_total_runs()` → counts ALL runs (~2,700+ for 227 strategies × ~12 runs each)
**Correct**: N should be the number of INDEPENDENT STRATEGIES tested (~227), not total runs
**Impact**: N is overstated by ~12x. The expected max SR benchmark `E[max SR | N, T]` increases with N, making DSR systematically near-zero for all strategies.
**Reference**: Bailey & de Prado 2016 use N = number of TRIALS (strategies), not number of individual backtests within one strategy (sensitivity runs, CV folds, etc.)

**Fix**: Replace `store.count_total_runs()` with `len(store.get_strategy_list())` or count distinct strategy_ids.

---

### Issue #2 — HIGH: BANKNIFTY strategies use NIFTY lot size (65) for trade-level stats

**Location**: `dashboard_server.py:360`
**Code**: `lot_size = NIFTY_LOT  # TODO: resolve per underlying`
**Affected metrics** (recomputed in the runs endpoint, not from DB):
- `total_pnl` — 2.17x too high (65/30 = 2.17)
- `avg_winner`, `avg_loser` — 2.17x too high
- `profit_factor` — Not affected if all trades are scaled consistently, but if a net-loser trade becomes net-winner due to 2.17x scaling, profit_factor changes
- `days_profitable` — Potentially flipped if a just-barely-profitable day becomes clearly profitable (or vice versa)

**Not affected** (from DB, computed at save time with correct lot_size):
- `sharpe_raw`, `sharpe_deflated`, `net_edge_bps`, `trades_per_day`, `avg_hold_seconds`, `avg_daily_return_pct`

**Fix**: Load strategy's underlying from run record, resolve lot_size accordingly.

---

### Issue #3 — HIGH: `optimized_sharpe` in full_pipeline_results.json is NOT a Sharpe ratio

**Location**: `run_all.py:602`
**Code**: `summary["optimized_sharpe"] = round(sum(fold_pnls) / max(len(fold_pnls), 1), 2)`
**Actual value**: Average out-of-sample PnL in INR per held-out fold
**Named as**: "optimized_sharpe" in the JSON and in `/api/nested-cv` response
**Impact**:
- `/api/leaderboard` CORRECTLY overrides this with the DB value (`optimized_metrics[k]["sharpe"]` from the `optimized_full` run). Leaderboard and tier cards show correct Sharpe. ✓
- `/api/nested-cv` endpoint (line 687) returns the WRONG value: `s.get("optimized_sharpe")` from the JSON
- CV Analysis overlay tab that reads from `/api/nested-cv` would display this mislabeled value if it calls that endpoint

**Fix**: Rename `optimized_sharpe` in the JSON to `avg_oos_pnl_inr` and/or always use the DB value.

---

### Issue #4 — MEDIUM: CAGR formula produces astronomical values for 12-day backtest

**Location**: `result_store.py:952`
**Code**: `cagr = (1 + total_return) ** (252.0 / n_days) - 1.0`
**With n_days=12**: exponent = 252/12 = 21
**Example**: Strategy with +20% return → CAGR = (1.20)^21 - 1 = **5,580%**
**Impact**: CAGR shown in the Backtest tab dropdown is meaningless at this sample size. Could mislead.
**Fix**: Either don't display CAGR for < 60 days, or display it with a prominent "extrapolated from 12 days" warning. Do not use CAGR for strategy ranking.

---

### Issue #5 — MEDIUM: SSOT violation — constants duplicated in config.py and cost_model.py

**Affected constants**: `NIFTY_LOT`, `STT_RATE`, `BROKERAGE_PER_ORDER`, `EXCHANGE_TXN_RATE`, `CAPITAL_PER_ENTRY`
**Risk**: Silent divergence if one file is updated without the other
**Fix**: `config.py` should import from `cost_model.py`, not re-define. Or `cost_model.py` should import from `config.py`.

---

### Issue #6 — LOW: `pnl` in trades parquet is 1-lot gross, not capital-deployed

**Location**: `run_all.py:271`, state machine output
**Value**: `(exit_premium - entry_premium) × lot_size` (1 lot)
**Dashboard comment**: `dashboard_server.py:446-447` documents this correctly
**Impact**: The `pnl` column is not used for any displayed metrics (metrics.py recomputes from premiums). The column is misleading but harmless.
**Fix**: Consider removing the `pnl` column from parquet or renaming to `pnl_1lot_gross`.

---

### Issue #7 — LOW: Internal variable names in metrics.py mislead readers

**Location**: `metrics.py:69-70`
**Code**:
```python
gross_profit = float(np.sum(winners))  # winners = net_pnls > 0 (NET, not GROSS)
gross_loss = float(np.abs(np.sum(losers)))
```
**Impact**: `gross_profit` actually contains the sum of NET-positive trades. Anyone auditing this code would be confused. The final `profit_factor` computed is correct (net/net ratio). Only naming is misleading.

---

*Next: [02_pipeline_and_edge_cases.md](02_pipeline_and_edge_cases.md)*
