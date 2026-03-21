# Fix Audit Findings — Post-Audit Remediation

You are fixing issues found in `outputs/pre_training_audit_report.md`. Detailed data in `outputs/phase12_results.json`, `outputs/phase3_results.json`, `outputs/phase4_results.json`, `outputs/phase6_results.json`.

**Rules:**
- Edit files in-place. JSON in `unique_strategies/`, Python in `pipeline/strategies/`.
- Python is authoritative for logic; JSON is authoritative for calibration numbers (stop/target/edge).
- After ALL fixes, re-run the audit prompt (`prompts/audit_pre_training.md`) to verify 0 critical failures.
- Save a fix log to `outputs/audit_fix_log.md` with every change made (strategy name, field, old→new).
- Use Polars not pandas. Activate venv: `source .venv/bin/activate && export PYTHONPATH="$PWD"`.

---

## TIER 1: Scripted Bulk Fixes (do these first — write and run Python scripts)

### 1A. Profit Factor Recalculation (109 strategies)

Read `outputs/phase12_results.json` → `math_failures.check8_pf` for the full list.

For each affected strategy:
1. Load its JSON from `unique_strategies/`
2. Read `win_rate_estimate` and `avg_winner_to_loser` from `edge` block
3. Compute: `correct_pf = (win_rate * avg_wl) / (1 - win_rate)`
4. Overwrite `profit_factor_estimate` with `round(correct_pf, 2)`
5. Save JSON back (preserve formatting: `json.dump(indent=2)`)

**Important:** The win_rate and avg_winner_to_loser are the source-of-truth fields (set during strategy design). The profit_factor was derived but computed incorrectly. Do NOT change win_rate or avg_wl.

Save script as `scripts/fix_profit_factor.py`. Run it. Report count fixed.

### 1B. Fix bad_computed_on (6 indicators, 3 strategies)

| Strategy | Indicators | Change |
|---|---|---|
| `implied_vol_dislocation_v1` | iv_rv_ratio, ratio_zscore | `"computed"` → `"derived"` |
| `implied_vs_realized_vol_v1` | vrp, vrp_z | `"spot + vix"` → `"derived"` |
| `multi_asset_signal_v1` | divergence_bull, divergence_bear | `"spot+vix"` → `"derived"` |

Edit each JSON directly — only 3 files.

### 1C. Fix missing exit keys (2 strategies)

| Strategy | Fix |
|---|---|
| `autocorrelation_regime_v1` | Read the Python file. Extract stop_points, target_points, time_stop_bars from the `OptionSignals` return. Add to JSON `exit` block with rationale strings. |
| `momentum_reversion_switch_v1` | Same approach. |

If Python has no explicit stop/target (uses defaults from base.py), set `stop_points: 5, target_points: 8, time_stop_bars: 24` (pipeline defaults from config.py).

### 1D. Tag normalization (27 strategies)

Write a script to load all JSONs, replace `"mean-reversion"` with `"mean_reversion"` in the `tags` array, save back. Script name: `scripts/fix_tags.py`.

### 1E. Lookback type normalization (90 indicators, 42 strategies)

Write a script:
- Load all JSONs
- For each indicator where `lookback` is a string:
  - `"session"` or `"session_cumulative"` → `0` (session-scoped, no fixed lookback)
  - `"prev_day"` → `0` (structural level)
  - `"recursive"` / `"derived"` / `"dynamic"` → `0` (unbounded)
- Save back

Script name: `scripts/fix_lookbacks.py`. This converts all lookbacks to int/0 which the pipeline expects.

### 1F. Trades/day inconsistency (4 strategies)

For each: set `max_trades_per_day` = max of (current max_trades_per_day, typical_range_high). Edit JSONs directly.

| Strategy | Current max | typical_high | New max |
|---|---|---|---|
| `ema_crossover_v1` | 6 | 8 | 8 |
| `inside_bar_breakout_v177` | 8 | 10 | 10 |
| `spectral_analysis_v1` | 6 | 8 | 8 |
| `vol_surface_signal_v1` | 5 | 6 | 6 |

Also update the corresponding Python files: `Strategy.max_trades_per_day` class attribute must match.

---

## TIER 2: Per-Strategy Fixes (require reading JSON + Python for each)

### 2A. Session_end mismatches (10 strategies) — CRITICAL

Read `outputs/phase4_results.json` → `check20_eod.py_mismatch` for the full list.

For each mismatch, determine the correct session_end:
- **Gap strategies** (gap_fade, gap_continuation, gap_momentum): these intentionally trade a narrow morning window. The Python `session_end_minutes` is likely correct (shorter window). Update JSON `eod_rule` to match Python.
- **All-day strategies**: JSON `eod_rule: "flatten at 15:25"` is likely correct. Update Python `session_end_minutes = 925`.

Read BOTH files for each strategy. Decide based on the `mechanism` field — does the thesis describe a time-limited window or an all-day pattern?

Fix list (from phase4 data):
1. `european_open_momentum_continuation` — JSON=855(14:15), Py=925(15:25)
2. `gap_continuation_v1` — JSON=925(15:25), Py=660(11:00)
3. `gap_fade_v1` — JSON=565(09:25), Py=925(15:25)
4. `gap_momentum_continuation` — JSON=925(15:25), Py=690(11:30)
5. `index_correlated_breakout_v1` — JSON=925(15:25), Py=870(14:30)
6. Plus 5 more from the phase4 data — read the full list.

### 2B. Codegen bugs (2 strategies) — CRITICAL

**`parabolic_sar_momentum_v1`**: In Python, find `flip_count < 3` and change to `flip_count < 4` to match JSON spec.

**`market_profile_poc_v1`**:
1. Read the JSON entry conditions for `prev_close` comparison
2. Add the missing condition to the Python `compute()` method
3. Find the look-ahead bias in market profile sizing (references full-day range). Fix to use only data up to current bar.

### 2C. Undocumented Python parameters (4+ strategies)

These strategies have conditions in Python not specified in JSON. For each, update the JSON to document what's actually coded:

| Strategy | Undocumented Parameter | Action |
|---|---|---|
| `sma_crossover_banknifty_v1` | `sma_separation` guard + TunableParam | Add to JSON `indicators` and `entry.conditions` |
| `elder_impulse_v1` | `ema_slope_min` threshold | Add to JSON `indicators` |
| `gap_and_go_with_volume_v1` | VWAP touch detection more permissive in py | Tighten Python to match JSON spec, OR update JSON to match Python |
| `multi_factor_composite_v1` | momentum threshold discrepancy | Align JSON and Python |
| `vwap_mean_reversion_v19` | VWAP guard in JSON not coded in py | Add the guard to Python |

For each: read both files, determine which version is more correct (Python's actual behavior vs JSON's intended design), align to the better version.

### 2D. Entry asymmetry — `ema200_dip_reversal_v1`

Read JSON and Python. The buy_ce block has 6 conditions, buy_pe has 1. Either:
- Add the missing 5 symmetric PE conditions to both JSON and Python, OR
- If the strategy is intentionally long-biased, add `"directional_bias": "bullish"` to the JSON `mechanism` block and document why

### 2E. VWAP reset bug — `previous_day_high_low_breakout_v1.py`

Read the Python file. Find the VWAP computation. Replace the manual `session_dates` loop with the standard `day_id` groupby pattern used in other strategy files. Test that the compute() method still runs without error.

---

## TIER 3: Portfolio-Level Fixes (structural decisions needed)

### 3A. Stop/Target Diversification (113 strategies at 4/7)

This is the hardest fix. 44% of strategies use stop=4, target=7. The numbers aren't necessarily wrong — they're within the CONVENTIONS.md calibration range for 15-120s holds. But the uniformity is a codegen artifact (the LLM defaulted to 4/7 when uncertain).

**Approach:** Write a script (`scripts/diversify_stop_target.py`) that:

1. Loads all JSONs
2. For each strategy with stop=4, target=7:
   a. Read its `hold_time_seconds` range and `mechanism` type
   b. Assign a new (stop, target) based on mechanism:
      - **Fast mean reversion** (hold 15-30s): stop=3, target=5
      - **Standard mean reversion** (hold 30-60s): stop=4, target=6 or 3/6
      - **Momentum/breakout** (hold 30-120s): stop=5, target=8 or 5/9
      - **Scalp** (hold 5-15s): stop=2.5, target=4
   c. Recalculate `avg_winner_to_loser = target / stop`
   d. Recalculate `profit_factor_estimate = (win_rate * new_avg_wl) / (1 - win_rate)`
   e. Update JSON
   f. Update corresponding Python file: find `stop_points` and `target_points` in the OptionSignals return, update values

3. After script runs, verify: at least 20 unique (stop, target) pairs, no single pair >25% of corpus.

**Do NOT change stop/target for strategies that already have non-4/7 pairs — those were deliberately calibrated.**

Also update each affected Python file's `OptionSignals(stop_points=..., target_points=...)` to match.

### 3B. Session Overlap — Document Only (no code fix needed)

The 236 concurrent strategies at 10:15 IST is a training data concern, not a per-strategy bug. No code changes needed now. This is a batch scheduling/portfolio construction issue to address when building the live allocator.

Add a note to `outputs/audit_fix_log.md`: "CHECK29: 236 concurrent strategies at peak. Accepted for training phase — will be addressed during portfolio allocation design."

### 3C. No VIX Filter (116 strategies) — Document Only

For each of the 116 strategies in `outputs/phase4_results.json` → `check19_no_vix`:
- Add `"vix_agnostic": true` to the JSON top level
- This is metadata, not a code change — it flags the strategy as not using VIX filtering

Write a script: `scripts/tag_vix_agnostic.py`.

---

## TIER 4: Verification

After all fixes are applied:

1. Run `scripts/fix_profit_factor.py` — verify 0 PF mismatches remain
2. Run `scripts/fix_tags.py` — verify no `mean-reversion` tags remain
3. Run `scripts/fix_lookbacks.py` — verify all lookbacks are int or 0
4. Run `scripts/diversify_stop_target.py` — verify ≥20 unique pairs, max <25%
5. Run `scripts/tag_vix_agnostic.py` — verify 116 strategies tagged
6. Spot-check 5 of the 10 session_end fixes manually
7. Spot-check the 2 codegen bug fixes
8. Import-test all 258 Python files:
   ```bash
   for f in pipeline/strategies/*.py; do
     [ "$(basename $f)" = "__init__.py" ] && continue
     [ "$(basename $f)" = "base.py" ] && continue
     python3 -c "import importlib.util; spec=importlib.util.spec_from_file_location('m','$f'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(f'OK: {m.Strategy.name}')" 2>&1 || echo "FAIL: $f"
   done
   ```
9. Re-run the full audit (`prompts/audit_pre_training.md`) to confirm 0 critical failures
10. Save fix log and commit: `git add -A && git commit -m "fix: remediate pre-training audit findings (109 PF, 10 session, 2 bugs, stop/target diversification)"`

---

## Execution Order

```
1. Tier 1 scripts (1A→1B→1C→1D→1E→1F) — automated, ~10 min
2. Tier 2 per-strategy fixes (2A→2B→2C→2D→2E) — ~30 min
3. Tier 3A stop/target diversification — ~20 min
4. Tier 3B+3C documentation — ~5 min
5. Tier 4 verification — ~15 min
Total: ~80 min
```

Do Tier 1 all at once (write all scripts, run them). Then Tier 2 one strategy at a time. Then Tier 3. Then verify everything.
