# Pre-Training Audit — 5s NIFTY Options Pipeline

## Context

Auditing ~258 strategy JSONs (`unique_strategies/`) and matching Python (`pipeline/strategies/`) before training. 5-second bar NIFTY/BANKNIFTY index option strategies adapted from 1-min equity originals.

**Files:** JSONs in `unique_strategies/`, Python in `pipeline/strategies/` (+ `base.py`, `__init__.py`), qualified list in `outputs/strategy_retriage.json` (291 entries, 21 killed, 1 failed), kills in `outputs/conversion_kills.json`, conventions in `pipeline/strategies/CONVENTIONS.md`, cost model in `pipeline/cost_model.py`, config in `pipeline/config.py`.

**Data:** `5second_data/spot_candles.parquet` (71 days Dec25–Mar26), `5second_data/option_candles.parquet` (12 days Mar26 only). Pipeline intersects to 12 days. Timestamps UTC→IST in loader. Columns: spot_df has datetime/session_date/OHLCV/volume/day_id/time_minutes/atm_strike; option_df adds symbol/underlying/expiry/strike/option_type/open_interest.

**Execution:** Limit orders at candle close. **Spread = 0.** Hold 5–120s (1–24 bars). Capital ₹5L/trade.

**Costs (pipeline/cost_model.py):** STT 0.15% sell-side | Brokerage ₹20/order (₹40 RT flat) | Exchange 0.03553% both sides | SEBI ₹10/Cr | Stamp 0.003% buy-side | IPFT ₹0.50/lakh both sides | GST 18% on (brokerage+exchange+IPFT). **No spread.**

**Lots:** NIFTY=65, BANKNIFTY=30. At ₹5L/₹200 premium: 38 lots, breakeven ~0.5 pts. At ₹300: 25 lots, ~0.7 pts.

**Conventions ref:** Typical moves — 15s: 1–3 option pts, 30s: 2–5, 1min: 3–7, 2min: 5–10. Stops/targets in option premium points (1 opt pt ≈ 2 NIFTY spot pts via delta=0.5).

**BaseStrategy:** `name`, `underlying` (NIFTY|BANKNIFTY|BOTH), `session_start_minutes` (560=09:20), `session_end_minutes` (925=15:25), `max_trades_per_day` (10), `max_lookback` (120). Method `compute(spot_df, option_df, vix_df, params) → OptionSignals`.

**OptionSignals:** `buy_ce`/`buy_pe`/`sell_ce`/`sell_pe` (bool arrays), `stop_points`/`target_points` (float64 arrays), `strike_offset` (int32), `time_stop_bars` (int=24), `max_trades_per_day` (int=10).

---

## PHASE 1: Schema & Structure (batch all JSONs)

1. **Count JSONs** in `unique_strategies/`. Cross-check vs `strategy_retriage.json` and `conversion_kills.json`.

2. **Schema** — required top-level keys: `name, version, author, tags, original_strategy, original_name, mechanism, universe, timeframe, session, hold_time_seconds, max_hold_bars, trades_per_day, max_trades_per_day, data, indicators, entry, exit, filters, edge, adaptation_notes, weaknesses, _source`. Flag missing.

3. **Exit block** — required: `stop_points, stop_rationale, target_points, target_rationale, time_stop_bars, eod_rule`.

4. **Edge block** — required: `expected_bps, win_rate_estimate, avg_winner_to_loser, profit_factor_estimate, edge_decay_risk`.

5. **Indicator objects** — required: `name, formula, lookback, computed_on, purpose`. `computed_on` ∈ {spot, option, vix, derived}. `lookback` must be int, `"real_time"`, or `0`.

6. **Timeframe** must be `"5s"`. Flag others.

7. **Universe** ∈ {NIFTY, BANKNIFTY, BOTH}. Flag others.

## PHASE 2: Math Checks

8. **Profit factor:** `expected_pf = (win_rate × avg_winner_to_loser) / (1 - win_rate)`. Flag if |expected_pf - stated_pf| > 0.15. List all failures.

9. **R:R ratio:** `rr = target_points / stop_points`. Flag if rr < 1.0 or |rr - avg_winner_to_loser| > 0.5.

10. **Edge vs cost:** Flag `expected_bps < 3` (below viable) or `> 60` (suspiciously high).

11. **Hold time:** `max_hold_seconds = max_hold_bars × 5`. Flag if >2× stated max hold or <0.5× stated min hold.

12. **Stop/target calibration:** Flag stop < 2 (noise), stop > 8 (too wide), target > 15 (unrealistic), target < stop (negative R:R).

## PHASE 3: Indicator & Logic

13. **Lookback sanity:** `wall_clock_min = lookback × 5 / 60`. Flag if >30 min UNLESS VWAP or structural level (PDH/PDL/ORB).

14. **Warmup:** `max_warmup = max(lookbacks)`, `warmup_min = max_warmup × 5 / 60`. Cross-check vs Python `max_lookback` (flag >20% diff). Flag if session start < 09:15 + warmup_min.

15. **Duplicates:** Group by >3 shared tags. Within groups, flag pairs with >80% indicator name overlap AND identical stop/target. Cross-check vs `conversion_kills.json`.

16. **computed_on correctness:** Formula mentions OI/premium/greeks → must be `option`. VIX → `vix`. Spot OHLCV → `spot`. Flag mismatches.

17. **Entry completeness:** Must have `buy_ce` and `buy_pe` blocks, each with `conditions` (list) + `confirmation` (string). Flag asymmetric condition counts.

## PHASE 4: Red Flags

18. **Unused option_chain:** If `data.option_chain: true` but no indicator with `computed_on: option` and no entry/filter using option data → flag. (Note: most strategies correctly compute on SPOT, not options.)

19. **No VIX filter:** List strategies with no vix_filter AND no VIX indicator. Not necessarily wrong (ATR self-adapts), but flag for review.

20. **EOD rule:** Must be 15:10–15:25 IST. Flag >15:25 or <15:00. Cross-check vs Python `session_end_minutes`.

21. **Trades/day:** Flag `max_trades_per_day > 20`. Flag if typical range HIGH > max_trades_per_day.

22. **Win rate sanity:** Mean reversion: 0.52–0.65. Momentum: 0.42–0.55. Flag >0.65 or <0.40. Flag mean_reversion tag with win_rate <0.50.

## PHASE 5: Codegen (pipeline/strategies/*.py)

23. **Match JSON↔Python:** Report: JSONs without .py (should be ~22), .py without JSON (orphans), matched count.

24. **Sample 15 strategies**, read JSON + .py, verify: `underlying` matches `universe`, `name` matches, `max_lookback` consistent with max indicator lookback, indicator formulas match, entry conditions match, stop/target match, `time_stop_bars` match, session times match, filters implemented not just documented.

25. **Bug scan (grep ALL .py files):**
- Look-ahead bias: signal uses current bar instead of `[i-1]`
- VWAP: missing `day_id` groupby reset
- Rolling without `min_periods`
- Division by zero on volume=0
- Hardcoded NIFTY levels (e.g. `24000`)
- `np.nan` in bool arrays (NaN is truthy → spurious entries)
- Missing session time filter
- Cross-strategy imports (must be self-contained)
- Class not named `Strategy`

## PHASE 6: Portfolio-Level

26. **Tag frequency:** Top 10 tags. Flag if >50 strategies share primary mechanism tag.

27. **Stop/target clustering:** Histogram of (stop, target) pairs. Flag if >30% use same combo.

28. **Edge distribution:** mean/median/std of `expected_bps`. Flag mean >30 (overconfidence) or std <5 (suspicious uniformity).

29. **Session overlap:** Count eligible strategies per 30-min window (09:15–15:30). Flag any window >200.

30. **Mechanism uniqueness:** Sample 20 mechanisms. Flag pairs with >50% word overlap.

---

## Output

Save to `outputs/pre_training_audit_report.md`:

```
## 1. Summary Statistics
Total JSONs / Python files / matched pairs / schema pass rate / math pass rate / red flags

## 2. Critical Failures (must fix before training)
[strategy name, failure type, details]

## 3. Warnings (review, may be acceptable)
[strategy name, warning type, details]

## 4. Duplicate/Near-Duplicate Pairs
[pairs with similarity details]

## 5. Distribution Analysis
Tag frequency | Stop/target clusters | Edge histogram | Session overlap

## 6. Codegen Audit
JSON↔Python match summary | 15-strategy sample | Bug scan results

## 7. Recommendations
[Prioritized fixes before training]
```

## Execution

- Batch all JSONs first, then run checks. No one-by-one commentary.
- Use **Polars** (not pandas). `source .venv/bin/activate && export PYTHONPATH="$PWD"`.
- If directory missing, note and continue. If JSON malformed, log and continue.
- Be ruthless on math. These trade real capital.
- No preamble. Just report.
