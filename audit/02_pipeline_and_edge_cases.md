# Audit Document 2: Pipeline Integrity & Edge Cases

Generated: 2026-03-23

---

## Part A: Data Pipeline Integrity

### A1. Timestamp Handling

| Check | Status | Evidence |
|-------|--------|----------|
| Timestamps stored as IST | UNKNOWN | `config.py` uses 09:15–15:30 market hours, implying IST interpretation. No explicit timezone conversion code found in data_loader. If parquet files are UTC, all time checks (session start/end, EOD flatten) would be wrong. |
| `entry_time`/`exit_time` in trades parquet | PASS | Derived from `spot_df["datetime"]` at bar index. If spot data is IST, trades are IST. |
| EOD flatten at 15:25 | PASS | `config.py:29`: `EOD_FLATTEN_H, EOD_FLATTEN_M = 15, 25`. State machine enforces this. |
| `exit_time` used for daily grouping | PASS | `metrics.py:124-129`, `result_store.py:163-168`: `exit_time.cast(pl.Date)` for daily aggregation. |

**Action Required**: Verify that 5-second parquet files store timestamps as IST (or that the data loader converts UTC→IST). If timestamps are UTC, market session filters would pass trades outside 09:15–15:30 IST.

---

### A2. Fill Price Provenance

| Check | Status | Notes |
|-------|--------|-------|
| Entry premium = candle close of entry bar | PASS | `run_all.py:backtest_strategy()` builds `ce_grid[bar][strike]` from option close prices. State machine uses this grid. |
| Exit premium = candle close of exit bar | PASS | Same grid, exit bar index. |
| Look-ahead: entry bar vs signal bar | PASS (design) | Signals computed from `spot_df`, entry executed at `entry_bar` close. If signal generated at bar `t`, entry at bar `t+1`? Must verify state machine. The signal arrays are boolean buy/sell arrays over all bars — state machine consumes signal at bar `t` and enters at bar `t` close. |
| Potential look-ahead at signal bar | NEEDS VERIFICATION | If `strategy.compute(spot_df, ...)` produces signals using bar `t` close prices, and entry executes at bar `t` close, that is a same-bar fill — not look-ahead, but it relies on the EXACT close price being known at signal time. For 5-second bars this is reasonable (the candle closes, signal is computed, entry at that close). |
| Option premium grid: forward-fill | PASS | `run_all.py:build_option_premium_grid()` fills day-by-day with latest available premium at/before each bar. Nearest expiry only. |
| `premium_missing_at_exit` flag | PASS | Stored in trades parquet. If True, the exit premium may be stale. |

---

### A3. Deduplication

| Check | Status | Evidence |
|-------|--------|----------|
| Duplicate trades from state machine | PASS | Numba state machine processes bars sequentially. At most 1 open position at any time (STATE_FLAT / STATE_LONG_CE / STATE_LONG_PE). No simultaneous positions. |
| Duplicate runs in DB | PASS | `run_id` = UUID4, primary key. Atomic write with ROLLBACK on error. |
| `optimized_full` run created twice (pipeline + backfill) | RISK | If `--full` pipeline run followed by `scripts/backfill_optimized_full.py`, the strategy would be skipped by backfill (it checks `notes = 'optimized_full'` exists). But if backfill ran BEFORE `--full`, a second `optimized_full` run would be inserted. Both would then exist. The API `next((r for r in runs if r.notes == 'optimized_full'))` would return the first encountered (by `get_all_runs()` which orders by `created_at DESC`). The first (most recent) would win. This is safe but creates orphaned older runs. |

---

### A4. Missing Data Handling

| Scenario | Behavior | Assessment |
|----------|----------|------------|
| Option premium = 0 at entry bar | `compute_lots()` guards `entry_premium <= 0 → return 1`. But state machine may still enter with ep=0. | RISK: If ep=0, `net_pnl_quick(0, xp, ...)` would compute lots=1 (guard), `buy_turnover=0`, `sell_turnover=xp*qty`. Net PnL = xp*qty - costs. This would record a free entry and a profit equal to exit premium. |
| Gaps in option data (no premium for a strike on a day) | `build_option_premium_grid()` forward-fills within the day but leaves zero if no data exists before that bar. | RISK: Zero premium treated as valid price. `premium_missing_at_exit` flag only marks missing exit premiums. |
| VIX data missing | `backtest_strategy()` passes `vix_df` to `strategy.compute()`. If strategy doesn't use VIX, no issue. If it does, NaN handling per-strategy. | UNKNOWN per strategy |
| Strategies with 0 trades on a fold's test day | `fold_trades = None`, `fold_metrics = _empty_metrics()`. `day_pnl = 0`, `profitable = False`. | PASS: correctly contributes a negative day to CV pass count. |

---

### A5. Ranking Logic Audit

**Leaderboard Strategy List** (`get_strategy_list()`, `result_store.py:556-573`):

```sql
SELECT r.strategy_id,
       MAX(res.sharpe_raw) as best_sharpe_raw,
       MAX(res.sharpe_deflated) as best_sharpe_deflated,
       MAX(res.net_edge_bps) as best_net_edge_bps,
       MAX(CASE WHEN res.split='full' THEN res.kill_condition_triggered ELSE 0 END) as any_kill
FROM runs r
LEFT JOIN results res ON r.run_id = res.run_id AND res.split IN ('test', 'full')
GROUP BY r.strategy_id
ORDER BY best_sharpe_deflated DESC
```

**Issues with this query:**

1. **`MAX(sharpe_deflated)` across ALL run types** — includes sensitivity runs, CV fold runs (test split), and default/optimized full runs. A sensitivity run with artificially high DSR would rank above the strategy's true optimized performance.

2. **`any_kill`** uses `MAX(kill_condition_triggered)` for `split='full'` runs. If any full-split run triggered a kill (Sharpe < 0.5), `any_kill=1`. But this includes the `default_params_full` run. A strategy whose default run had Sharpe < 0.5 but whose `optimized_full` run has Sharpe 10+ would still show `any_kill=1`.

3. **Family leaderboard** (`/api/leaderboard`): sorts families by `best_dsr` = `max(best_sharpe_deflated per member)`. Members are already aggregated via the above SQL. Same issues apply.

**Tier Classification** (`leaderboard.js:_classifyStrategies()`):

Tier rules:
- `deploy`: `cv_passed AND sensitivity_verdict === 'ROBUST'` → currently 0 strategies
- `promising`: `cv_passed` (regardless of sensitivity) → ~6 strategies
- `watchlist`: `!cv_passed AND sensitivity_verdict === 'ROBUST' AND optSharpe > 0`
- `needsWork`: `!cv_passed AND !robust AND optSharpe > 0`
- `noEdge`: `optSharpe <= 0`

**Key observation**: `optSharpe` for tier classification comes from `cv_data[k].optimized_sharpe` in the `/api/leaderboard` response, which is correctly sourced from the DB `optimized_full` run (`dashboard_server.py:184`). ✓

**Issue**: Strategies in `full_pipeline_results.json` (291 entries) but without an `optimized_full` run in DB would have `optimized_sharpe = null` from the API (line 184: `(optimized_metrics.get(k) or {}).get("sharpe")` → None). In `_classifyStrategies`, `optSharpe = v.optimized_sharpe || v.default_sharpe || 0`. So they'd fall back to `default_sharpe` from the JSON — which IS a real Sharpe (computed at default params full run). ✓

**Issue**: 291 strategies in pipeline JSON but only 227 in optimized_metrics. The 64 difference are strategies without `optimized_full` runs (either failed, had 0 trades, or had no tunable params and sensitivity was skipped). For these, the tier card displays `default_sharpe` as the Sharpe value but labels it as if it's the optimized Sharpe. The card itself doesn't label the source, so it's not explicitly wrong — but it's comparing apples (optimized Sharpe) and oranges (default Sharpe) within the same tier.

---

## Part B: Formula Correctness

### B1. Sharpe Ratio

```python
# metrics.py:148-154
mean_daily = float(np.mean(daily_returns_full))
std_daily = float(np.std(daily_returns_full, ddof=1))
sharpe = mean_daily / std_daily * math.sqrt(252)
```

**Checks:**

| Aspect | Status |
|--------|--------|
| Zero-fill non-trading days | PASS — `zero_fill = max(0, total_days - days_with_trades)` |
| `total_days` = days in spot_df, not hardcoded | PASS — `total_days = spot_df["day_id"].n_unique()` in `backtest_strategy()` |
| ddof=1 (unbiased) | PASS |
| Annualization factor sqrt(252) | PASS — daily returns |
| Denominator zero check | PASS — returns 0.0 if std=0 |
| Negative Sharpe possible | PASS — correctly shows negative |
| Same formula for Optuna objective | PASS — `compute_sharpe_from_trades()` in metrics.py uses identical formula |
| Sorting before groupby | PASS — `daily_pnl = ...sort("trade_date")` |
| Capital denominator | PASS — `CAPITAL_PER_ENTRY = 500_000` (not `ep * qty`) |

**Statistical note**: With T=12 observations (zero-filled), the standard error of the Sharpe estimator is `~sqrt((1 + SR²/2) / T)`. For SR=5, SE ≈ sqrt((1+12.5)/12) ≈ 1.06. The 95% CI is roughly ±2 Sharpe points. Any Sharpe estimate is highly uncertain.

---

### B2. DSR (Probabilistic Sharpe Ratio)

```python
# dsr.py:168
z = (sr_hat - sr_benchmark_raw) * sqrt(T-1) / sqrt(var_component)
psr = Φ(z)
```

**Checks:**

| Aspect | Status |
|--------|--------|
| N-trials benchmark `E[max SR | N, T]` | FAIL — N uses total runs (~2700) not strategies (~227). See Issue #1. |
| Variance estimator accounts for skewness/kurtosis | PASS — Bailey & de Prado eq. 4 implemented |
| Returns are non-annualized in DSR computation | PASS — `sr_hat = mu/sigma` in per-obs units |
| T = total zero-filled return series | PASS — `daily_returns` stored in DB include zero-fill |
| Skewness estimator | PASS — `_skewness()` with ddof=0, matches paper |
| Kurtosis estimator | PASS — excess kurtosis, ddof=0 |
| Edge case T < 5 | PASS — returns `_empty_dsr()` |
| Autocorrelation computed but NOT used in variance | PARTIAL — `autocorr_lag1` is returned but the variance formula in dsr.py line 157 does not include the autocorrelation adjustment. Bailey & de Prado 2016 Eq. 3 does include an autocorrelation correction term. |

---

### B3. BPS Computation

**Formula** (`result_store.py:111-115`):
```
gross_bps_per_trade = gross_pnl / (ep * qty) * 10_000
fees_bps_per_trade  = total_cost / (ep * qty) * 10_000
net_edge_bps        = mean(gross_bps_per_trade) - mean(fees_bps_per_trade)
```

**Checks:**

| Aspect | Status |
|--------|--------|
| BPS denominator = entry turnover `ep * qty` | PASS — measures return relative to deployed capital per trade |
| Weighting | ISSUE — unweighted average. Trades with low entry premiums (high lots) have high deployed capital but equal weight as high-premium low-lot trades. Should use total-deployed-capital-weighted average for portfolio-level edge. |
| Zero `ep` guard | PASS — `if ep <= 0: continue` |
| No spread/slippage | CORRECT — limit order model, documented |

---

### B4. CAGR

```python
# result_store.py:950-952
total_return = total_pnl / CAPITAL_PER_ENTRY
cagr = (1 + total_return) ** (252.0 / n_days) - 1.0
```

**Issue**: With 12 trading days and `252/12 = 21` as exponent, a 10% return over 12 days annualizes to `(1.10)^21 - 1 = 6.7x = 670%`. This is mathematically correct but **practically meaningless** as an extrapolation. It should not be used for decision-making or ranking.

---

### B5. Profit Factor

```python
# metrics.py:71-73
profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
```

Where `gross_profit = sum(net_pnls[net_pnls > 0])` (confusingly named).

**Edge cases:**
- All winners: `gross_loss = 0`, returns `inf` → capped at 999.0 at line 171 ✓
- All losers: `gross_profit = 0`, `gross_loss > 0`, returns `0.0` ✓
- Zero trades: returns `0.0` via `_empty_metrics()` ✓

---

### B6. Max Drawdown

```python
# metrics.py:102-105
cumulative_pnl = np.cumsum(sorted_net)
running_max = np.maximum.accumulate(cumulative_pnl)
drawdowns = running_max - cumulative_pnl
max_drawdown = float(np.max(drawdowns))
```

**Checks:**

| Aspect | Status |
|--------|--------|
| Sorted by exit_time before cumsum | PASS — `sorted_df = trades_df.sort("exit_time")` |
| Uses NET PnL (after costs) | PASS |
| Intraday drawdown within open trade | NOT CAPTURED — only trade-sequence drawdown |
| Edge: single trade | Returns `max(running_max - cumulative_pnl)` = `max(0, -net_pnl)`. If trade loses, DD = |net_pnl|. If trade wins, DD = 0. ✓ |
| Edge: all winners | Cumulative PnL monotonically increases, drawdowns always 0. Max DD = 0. ✓ |
| Edge: initial drawdown then recovery | `running_max` starts at cumulative value of first trade. If first trade loses, running_max = negative but drawdowns = 0 (running_max = cumulative_pnl at first step). Wait — running_max starts at `cumulative_pnl[0]`. If `sorted_net[0] = -1000`, then `cumulative_pnl[0] = -1000`, `running_max[0] = -1000`, `drawdowns[0] = 0`. Max DD might be understated if sequence starts with losses then recovers. Example: [-100, -100, +300] → cum = [-100, -200, 100] → running_max = [-100, -100, 100] → drawdowns = [0, 100, 0] → max DD = 100. Correct: peak is 0 (before any trades), so max DD should be 200. The formula starts from first trade, not from 0. **Drawdown is understated when the strategy opens with a losing streak.** |

---

### B7. `_median_params()` for consensus params

```python
# run_all.py:645-659
mid = len(vals) // 2
result[k] = vals[mid]
```

With 12 folds, `len(vals)=12`, `mid=6`, takes `vals[6]` (7th element, 0-indexed). True median for even-length list = average of `vals[5]` and `vals[6]`. This implementation is biased toward the higher value by half a position. For 12 folds, the bias is `(vals[6] - vals[5]) / 2`. For continuous parameters with small variation across folds, this is negligible. For integer params, it could matter.

---

## Part C: Edge Cases

| Scenario | Behavior | Assessment |
|----------|----------|------------|
| 1 trade only | `std_daily = 0` (single non-zero return), Sharpe = 0.0. Win rate = 1 or 0. Drawdown = 0 or |net_pnl|. | PASS — correct behavior |
| All trades are winners | Profit factor = inf, capped at 999. Max DD = 0. Win rate = 1.0. | PASS |
| All trades are losers | Profit factor = 0.0. Win rate = 0.0. Max DD = total_loss. | PASS |
| Net PnL > 0 but all trades individually lose (impossible) | Cannot happen: if net_pnl per trade is negative for all, total is negative. | N/A |
| Total costs > gross PnL (net loss) | Net PnL negative, all metrics handle negatives correctly. | PASS |
| entry_premium = 0 | `compute_lots()` returns 1. `net_pnl_quick(0, xp, 65)`: `buy_turnover=0`, `qty=65`, `gross=xp*65`, `stamp_duty=0`, `stt` on exit. Returns a profit even on "free" entry. | RISK — see A4 |
| Extremely high Sharpe (> 20) | All formulas handle. Kill condition checks Sharpe < 0.5 only (no upper bound kill). | PASS |
| Negative equity (cumulative loss > capital) | Drawdown can exceed ₹5L. No hard cap modeled. In reality, account would be stopped out. | MODEL LIMITATION — acceptable for simulation |
| Same strategy, 2 optimized_full runs | API returns first by `created_at DESC`. Second would be orphaned in DB. | PASS (safe, just duplicative) |
| Zero trading days in spot_df | `total_days = max(1, 0) = 1` in some paths. Division by zero guarded. | PASS |
| Fold with 0 trades on test day | `fold_metrics = _empty_metrics()`. `day_pnl = 0`. `profitable = False`. Contributes 0 to `days_profitable`. | PASS — correctly penalizes inactive days |

---

## Part D: Ranking Logic Cross-Checks

### D1. Leaderboard sort order

`/api/leaderboard` families sorted by `best_dsr` = max DSR across members.
Given DSR is near-zero for all strategies (due to Issue #1 with N), families are effectively unsorted (all DSR ≈ 0).

Within-tier sort in `leaderboard.js` uses `optimized_sharpe` (from DB) descending. ✓

### D2. CV Pass definition

**Count method** (`run_all.py:591-592`):
```python
days_profitable = sum(1 for d in cv_fold_results if d["profitable"])
cv_passed = days_profitable >= MIN_PROFITABLE_DAYS  # MIN_PROFITABLE_DAYS = 9
```

**`profitable` definition** (`run_all.py:587`):
```python
"profitable": day_pnl > 0,  # day_pnl = fold_metrics["total_pnl"] (NET)
```

So CV pass = net profitable on ≥ 9 of 12 held-out days. ✓ Economically correct.

**Consistency with overlay CV tab**: `get_cv_folds()` endpoint counts `n_profitable = sum(1 for f in folds if (f.get("net_edge_bps") or 0) > 0)`. This uses `net_edge_bps > 0` (positive edge), NOT `total_pnl > 0`. These can differ: a day with 1 trade and small positive premium movement but high costs could have `net_edge_bps > 0` but `total_pnl < 0` (or vice versa for multiple trades). **Minor inconsistency between pipeline JSON's `cv_profitable_days` and what the overlay CV tab independently counts.**

### D3. Sensitivity verdict

Computed by `pipeline/sensitivity.py` → `run_sensitivity()`. The verdict "ROBUST" vs "FRAGILE" is stored in `full_pipeline_results.json` and loaded into leaderboard via `cv_data`. Not recomputed from DB. If sensitivity runs in DB differ from what the JSON says, the verdict is stale.

### D4. Kill condition in leaderboard vs overlay

Leaderboard `any_kill` = `MAX(kill_condition_triggered)` across split IN ('test','full'). Includes test (fold) kills.
Kill condition = `sharpe_raw < 0.5`. Many fold runs with 1-day test data have Sharpe = 0 (not enough days to compute meaningful Sharpe), triggering kill. So `any_kill = 1` for most strategies even if optimized Sharpe is high. This makes `any_kill` nearly useless as a kill signal.

**Better definition**: Kill only for `split='full'` AND `notes='optimized_full'` runs.

---

*Next: [METRICS_AUDIT_REPORT.md](METRICS_AUDIT_REPORT.md)*
