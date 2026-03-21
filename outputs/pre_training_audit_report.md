# Pre-Training Audit Report — NIFTY Options Strategy Pipeline

Generated: 2026-03-21 | Strategies audited: 258 | Method: 9 parallel agents across 6 phases

---

## 1. Summary Statistics

| Metric | Value |
|---|---|
| Total JSONs parsed | 258 |
| Parse errors | 0 |
| Python files found | 258 |
| Matched JSON↔Python pairs | 258 (100%) |
| strategy_retriage.json total | 358 |
| Qualified | 291 |
| Discarded | 67 |
| conversion_kills.json entries | 33 |
| Expected converted | 258 |
| Actual converted | 258 |
| Delta | +1 (unexplained extra) |

| Check | Category | Flags | Severity |
|---|---|---|---|
| Schema: missing_top_level | Schema | 0 | — |
| Schema: missing_exit_keys | Schema | 2 | Critical |
| Schema: missing_edge_keys | Schema | 0 | — |
| Schema: bad_computed_on | Schema | 6 indicators / 3 strategies | Critical |
| Schema: bad_lookback_type | Schema | 90 indicators / 42 strategies | Warning |
| CHECK8: profit factor mismatch | Math | 109 strategies | Critical |
| CHECK9: R:R mismatch | Math | 1 strategy | Warning |
| CHECK13: lookback >30 min wall-clock | Indicator | 14 strategies | Warning |
| CHECK14: warmup divergence | Indicator | 204 flags | Warning |
| CHECK15: near-duplicates | Indicator | 0 | — |
| CHECK16: computed_on correctness | Indicator | 3 flags (2 likely false positives) | Warning |
| CHECK17: entry asymmetry | Indicator | 1 strategy | Warning |
| CHECK18: unused option_chain=true | Red Flag | 253 (structural) | Info |
| CHECK19: no VIX filter | Red Flag | 116 strategies | Warning |
| CHECK20: EOD rule (non-intentional early) | Red Flag | 19 strategies | Warning |
| CHECK20: JSON vs Python session_end mismatch | Red Flag | 10 strategies | Critical |
| CHECK21: trades/day inconsistency | Red Flag | 4 strategies | Warning |
| CHECK22: win rate sanity | Red Flag | 16 strategies | Warning |
| CHECK23: JSON↔Python file match | Codegen | 0 orphans | — |
| CHECK24: deep sample (15 strategies) | Codegen | 12/15 flagged, 2 confirmed bugs | Critical |
| CHECK25: bug scan (258 files) | Codegen | 1 finding | Warning |
| CHECK26: tag frequency | Portfolio | 0 flags (hyphenation issue noted) | Warning |
| CHECK27: stop/target clustering | Portfolio | CRITICAL — 43.8% at single pair | Critical |
| CHECK28: edge distribution | Portfolio | 0 flags (std borderline) | — |
| CHECK29: session overlap | Portfolio | CRITICAL — 236 concurrent peak | Critical |
| CHECK30: mechanism uniqueness | Portfolio | 0 flags | — |

---

## 2. Critical Failures (must fix before training)

### 2a. Schema — Missing Exit Keys

2 strategies are missing `stop_points` and/or `target_points` at the top level. These strategies cannot be trained without valid exit definitions.

| Strategy | Missing Keys |
|---|---|
| `autocorrelation_regime_v1` | stop_points, target_points |
| `momentum_reversion_switch_v1` | stop_points, target_points |

### 2b. Schema — bad_computed_on (non-enum values)

6 indicators across 3 strategies use invalid `computed_on` values. Valid enum values are: `spot`, `vix`, `option`, `derived`, `real_time`. Strings `'computed'`, `'spot + vix'`, and `'spot+vix'` are not valid.

| Strategy | Indicators | Invalid computed_on | Correct Value |
|---|---|---|---|
| `implied_vol_dislocation_v1` | iv_rv_ratio, ratio_zscore | `'computed'` | `"derived"` |
| `implied_vs_realized_vol_v1` | vrp, vrp_z | `'spot + vix'` | `"derived"` |
| `multi_asset_signal_v1` | divergence_bull, divergence_bear | `'spot+vix'` | `"derived"` |

### 2c. Math — Profit Factor Mismatch (CHECK8)

109 of 258 strategies (42.2%) have a computed profit factor that exceeds the stated value by more than 0.15. Computed PF = (win_rate × avg_wl) / (1 − win_rate). Computed values consistently fall in the range 1.7–2.4 vs stated 1.35–1.65. This indicates systematic inconsistency between `win_rate`, `avg_winner_to_loser`, and stated `profit_factor` fields — at least one field is wrong in each affected strategy.

**Representative example:**

| Strategy | Stated PF | win_rate | avg_wl | Computed PF | Delta |
|---|---|---|---|---|---|
| `bayesian_update_v1` | 1.34 | 0.57 | 1.75 | 2.32 | +0.98 |

Full list of 109 strategies recorded in `/tmp/phase12_results.json`. All 109 must be corrected before training.

### 2d. Codegen — Confirmed Logic Bugs (CHECK24 deep sample)

Two strategies have confirmed correctness bugs identified in the 15-strategy deep sample.

| Strategy | Bug Type | Detail |
|---|---|---|
| `parabolic_sar_momentum_v1` | Off-by-one | JSON specifies `flip_count < 4` (allows 3 flips); Python checks `flip_count < 3` (excludes 3 flips). Materially reduces entry frequency vs spec. |
| `market_profile_poc_v1` | Missing condition + look-ahead bias | `prev_close` comparison present in JSON is entirely absent from Python. Additionally, market profile sizing references full-day range (look-ahead bias). |

Both strategies must be excluded from training or corrected before inclusion.

### 2e. CHECK20 — JSON vs Python session_end Mismatch (>5 min delta)

10 strategies have material discrepancies between JSON-declared session_end and the Python-coded session cutoff. At these deltas, JSON and Python describe fundamentally different strategies.

| Strategy | JSON session_end | Python session_end | Delta (min) |
|---|---|---|---|
| `gap_fade_v1` | 09:25 | 925 min (likely coding error) | ~360 |
| `gap_continuation_v1` | 15:25 | 11:00 | 265 |
| `partial_gap_fill_scalp_v1` | 15:25 | 09:45 | 340 |
| (7 additional) | — | — | >5 min each |

Full list in `/tmp/phase3_results.json`.

### 2f. CHECK27 — Stop/Target Clustering (CRITICAL)

Only 11 unique (stop_points, target_points) pairs exist across 258 strategies. The dominant pair covers 43.8% of all strategies. A model trained on this distribution will overfit to the (4,7) trade management regime.

| (stop_pts, target_pts) | Count | % of total |
|---|---|---|
| (4, 7) | 113 | 43.8% |
| (4, 6) | 47 | 18.2% |
| (5, 8) | 35 | 13.6% |
| (3, 5) | 30 | 11.6% |
| (3, 6) | 11 | 4.3% |
| (5, 9) | 8 | 3.1% |
| (remaining 5 pairs) | 14 | 5.4% |

### 2g. CHECK29 — Session Overlap (CRITICAL)

Peak concurrency is 236 strategies simultaneously eligible at 10:15 IST. 10 of 13 intraday slots exceed 200 concurrent strategies. Training on undifferentiated concurrent signals will produce noisy labels and obscure strategy-level signal attribution.

| IST Slot | Concurrent Eligible Strategies |
|---|---|
| 09:15 | 5 |
| 09:45 | 230 |
| 10:15 | **236 (peak)** |
| 10:45 | 229 |
| 11:15 | 222 |
| 11:45 | 220 |
| 12:15 | 218 |
| 12:45 | 218 |
| 13:15 | 218 |
| 13:45 | 219 |
| 14:15 | 213 |

---

## 3. Warnings (review, may be acceptable)

### 3a. Schema — bad_lookback_type (90 indicators, 42 strategies)

90 indicators across 42 strategies use descriptive strings for `lookback` instead of valid types (`int`, `"real_time"`, or `0`).

| Invalid lookback string | Primary source |
|---|---|
| `"session"` | VWAP-based indicators (60+ entries) |
| `"session_cumulative"` | VWAP variants |
| `"prev_day"` | Prior-day reference indicators |
| `"recursive"` / `"derived"` / `"dynamic"` | Kalman/recursive filters (5 entries) |

VWAP lookbacks are structurally session-scoped and cannot be expressed as a fixed integer. Kalman/recursive lookbacks are unbounded by design. Acceptable if the schema is updated to enumerate `"session"` and `"recursive"` as valid values, or if a sentinel (e.g., `0`) is standardized.

### 3b. Math — R:R Mismatch (CHECK9)

| Strategy | Declared R:R | Computed R:R (pts) | avg_winner_to_loser | Gap |
|---|---|---|---|---|
| `rsi_divergence_vwap_v1` | 1.75 | 1.75 (7pt / 4pt) | 1.2 | 0.55 |

Declared R:R matches the stop/target ratio but `avg_winner_to_loser` is materially lower, implying partial take-profits or premature exits in practice. Review exit logic.

### 3c. CHECK13 — Long Lookbacks (>30 min wall-clock, 14 strategies)

| Strategy | Indicator | Lookback (bars) | Wall-clock (min) |
|---|---|---|---|
| `orb_closing_range_v1` | close_vs_day_range | 4500 | 375.0 |
| `lunch_hour_range_compression_v1` | lunch_low / range | 1080 | 90.0 |
| `bb_squeeze_breakout_v1` | bb_width_pctile_720 | 720 | 60.0 |
| `ema200_dip_reversal_v1` | ema_720, dist_pct | 720 | 60.0 |
| `fibonacci_intraday_retracement` | morning_high / morning_low | 720 | 60.0 |
| `first_hour_range_breakout_v1` | fhr_high / fhr_low | 720 | 60.0 |
| `sma_crossover_banknifty_v1` | sma_720 | 720 | 60.0 |
| `vix_regime_switch_v1` | vix_zscore | 720 | 60.0 |
| (6 additional) | — | — | >30 min |

Full list in `/tmp/phase3_results.json`. Several (ORB, first-hour, fibonacci) are structurally justified by strategy design. Non-obvious cases require documented justification.

### 3d. CHECK14 — Warmup Divergence (204 flags)

| Sub-type | Count | Root cause |
|---|---|---|
| Divergence-only (JSON vs Python max_lookback >20%) | 132 | Python uses 2× warmup buffer convention, undocumented in JSON |
| Session-start-only (session start < 09:15 + warmup_min) | 52 | warmup_min not enforced against session_start |
| Both issues | 20 | Combined |

The 2× Python warmup convention is project-wide and not a per-strategy bug. It is not documented in JSON metadata. Recommend adding a `warmup_multiplier: 2` field to the schema or documenting the convention explicitly.

### 3e. CHECK16 — computed_on Correctness (3 flags, 2 likely false positives)

| Strategy | Indicator | Flagged computed_on | Assessment |
|---|---|---|---|
| `mean_reversion_speed_v1` | ou_theta | spot | Likely false positive — ou_theta is an Ornstein-Uhlenbeck parameter, not an options Greek |
| `vol_surface_signal_v1` | atm_ce_close | option | Likely correct — ATM option closing prices should be computed_on=option |
| `vol_surface_signal_v1` | atm_pe_close | option | Likely correct — same rationale |

Manual review of `mean_reversion_speed_v1` only is recommended.

### 3f. CHECK17 — Entry Asymmetry

| Strategy | buy_ce conditions | buy_pe conditions | Difference |
|---|---|---|---|
| `ema200_dip_reversal_v1` | 6 | 1 | 5 |

A 5-condition asymmetry suggests the PE entry logic is underspecified. Verify this is intentional (structural directional bias) or a drafting omission.

### 3g. CHECK19 — No VIX Filter (116 strategies)

| Metric | Value |
|---|---|
| Strategies with VIX filter | 142 |
| Strategies without VIX filter | 116 |
| Percent without filter | 45.0% |

116 strategies will execute in high-volatility regimes without any volatility guard. Momentum and trend-following strategies are most exposed. Absence should be explicitly documented per strategy.

### 3h. CHECK20 — Early EOD Non-Intentional (19 strategies)

19 strategies close positions before 15:00 without documented justification. Distinct from the 24 intentional early-exit strategies (gap fades, ORB variants, Monday reversals).

| Strategy | session_end |
|---|---|
| `consolidation_breakout_v1` | 14:45 |
| `vwap_mean_reversion_v10` | 14:30 |
| `vix_mean_reversion_v1` | 14:00 |
| `orb_momentum_v22` | 14:30 |
| `orb_momentum_v29` | 14:30 |
| `orb_momentum_v32` | 14:30 |
| (13 additional) | <15:00 |

### 3i. CHECK21 — Trades/Day Inconsistency

| Strategy | typical_range_high | max_trades_per_day |
|---|---|---|
| `ema_crossover_v1` | 8 | 6 |
| `inside_bar_breakout_v177` | 10 | 8 |
| `spectral_analysis_v1` | 8 | 6 |
| `vol_surface_signal_v1` | 6 | 5 |

`typical_range_high` exceeds `max_trades_per_day` in all 4 cases. The stated maximum is architecturally unreachable given the declared typical range.

### 3j. CHECK22 — Win Rate Sanity (16 flags)

| Sub-type | Count | Detail |
|---|---|---|
| Momentum win_rate 0.56–0.60 (above 0.55 upper bound) | 14 | Slightly above expected momentum range |
| Too low: `donchian_channel_breakout_v1` win_rate=0.400 | 1 | Momentum strategies expected 0.42–0.55 |
| Momentum at floor of range | 1 | Edge case |

### 3k. CHECK25 — Bug Scan Finding (1 strategy)

| Strategy | File | Bug Type | Detail |
|---|---|---|---|
| `previous_day_high_low_breakout_v1` | `previous_day_high_low_breakout_v1.py` | vwap_no_dayid | VWAP computed via manual prev_date/session_dates loop instead of standard day_id groupby pattern. Will produce incorrect results if `session_date` column is absent. |

---

## 4. Duplicate / Near-Duplicate Pairs

**CHECK15 result: 0 near-duplicate pairs detected.**

`conversion_kills.json` (33 entries) already handled known duplicates prior to codegen. No residual duplicates remain in the 258-strategy corpus. No action required.

---

## 5. Distribution Analysis

### 5a. Tag Frequency (CHECK26)

No individual tag exceeds the 50-strategy threshold. However, `mean_reversion` (85) and `mean-reversion` (27) are inconsistent hyphenation variants of the same concept. Combined count is 112 (43.4%), which exceeds the threshold and represents a meaningful concentration risk.

| Tag | Count |
|---|---|
| momentum | 97 |
| mean_reversion | 85 |
| vwap | 63 |
| breakout | 54 |
| intraday | 51 |
| microstructure | 37 |
| volatility | 31 |
| mean-reversion | 27 |
| gap | 23 |
| trend_following | 20 |

### 5b. Stop/Target Clustering (CHECK27)

See Section 2f for full table. 11 unique pairs across 258 strategies. (4,7) dominates at 43.8%. Critical training data quality issue requiring diversification.

### 5c. Edge Distribution (CHECK28)

| Statistic | Value |
|---|---|
| Mean | 21.50 |
| Median | 20.0 |
| Std dev | 5.31 |
| Min | 12.0 |
| Max | 45.0 |
| p10 | 18.0 |
| p90 | 30.0 |

No flags. Distribution is acceptable. Std dev of 5.31 is marginally above the 5.0 threshold — monitor if strategies are added.

### 5d. Session Overlap (CHECK29)

See Section 2g for full slot table. 236 concurrent strategies at peak (10:15 IST). 10 of 13 intraday slots exceed 200 concurrent strategies.

### 5e. Mechanism Uniqueness (CHECK30)

20-strategy random sample: all mechanisms are sufficiently distinct (max pairwise Jaccard similarity < 0.50). No flags.

---

## 6. Codegen Audit

### 6a. JSON↔Python File Match (CHECK23)

| Metric | Result |
|---|---|
| JSONs without .py | 0 |
| .py without JSON (orphans) | 0 |
| Matched pairs | 258 / 258 (100%) |

### 6b. 15-Strategy Deep Sample (CHECK24)

| Strategy | max_lookback_match | entry_conditions_match | Notes | Result |
|---|---|---|---|---|
| `bollinger_mean_reversion_v1` | FAIL (py=120 vs JSON=60, 2×) | PASS | 2× warmup convention | FAIL |
| `elder_impulse_v1` | FAIL (py=120 vs JSON=45, 2.67×) | PASS | Undocumented ema_slope_min threshold in py | FAIL |
| `gap_and_go_with_volume_v1` | FAIL (py=120 vs JSON=60, 2×) | FAIL | VWAP touch detection more permissive in py than spec | FAIL |
| `harmonic_pattern_v1` | FAIL (py=720 vs JSON=6, 120×) | PASS | Justified by 30-min harmonic pattern window | FAIL |
| `intraday_trend_following_v1` | PASS | PASS | — | **PASS** |
| `market_profile_poc_v1` | FAIL (py=180 vs JSON=6) | FAIL | prev_close missing from py; look-ahead bias in sizing | FAIL |
| `multi_factor_composite_v1` | FAIL (py=240 vs JSON=120, 2×) | PASS | Undocumented momentum threshold discrepancy | FAIL |
| `orb_closing_range_v1` | FAIL (JSON=4500 full session, py=400 expanding) | PASS | — | FAIL |
| `parabolic_sar_momentum_v1` | FAIL (py=300 vs JSON=60) | FAIL | **OFF-BY-ONE BUG**: JSON flip_count<4, py flip_count<3 | FAIL |
| `range_day_band_fade_003` | PASS | PASS | — | **PASS** |
| `sma_crossover_banknifty_v1` | PASS | FAIL | Undocumented sma_separation guard with TunableParam in py, absent from JSON | FAIL |
| `vix_filtered_index_trend_v126` | FAIL (py=120 vs JSON=108, 11% over) | PASS | Borderline | FAIL |
| `volume_profile_breakout_v1` | PASS | PASS | — | **PASS** |
| `vwap_mean_reversion_v19` | PASS | FAIL | Explicit VWAP guard in JSON not coded explicitly in py (only implicit via zscore) | FAIL |
| `wavelet_decomposition_v1` | FAIL (py=120 vs JSON=16, 7.5×) | PASS | Undocumented | FAIL |

**Sample pass rate: 3/15 fully clean (20%). 12/15 have at least one flag. 2 have confirmed logic bugs.**

### 6c. Bug Scan — All 258 Files (CHECK25)

| Check | Findings |
|---|---|
| vwap_no_dayid | 1 (`previous_day_high_low_breakout_v1.py`) |
| look_ahead_bias | 0 |
| rolling_no_minperiods | 0 |
| division_by_zero | 0 |
| hardcoded_levels | 0 |
| nan_in_bool_array | 0 |
| missing_session_filter | 0 |
| cross_strategy_imports | 0 |
| class_not_strategy | 0 |

Low automated bug density does not override the 12/15 deep-sample failure rate. Undocumented parameter divergences are not detectable by pattern matching alone.

---

## 7. Recommendations

### P1 — Block training until resolved

| # | Issue | Action | Affected |
|---|---|---|---|
| 1 | Missing stop_points / target_points | Add exit keys to JSON and Python | `autocorrelation_regime_v1`, `momentum_reversion_switch_v1` |
| 2 | bad_computed_on (invalid enum values) | Replace `'computed'` / `'spot + vix'` / `'spot+vix'` with `"derived"` | `implied_vol_dislocation_v1`, `implied_vs_realized_vol_v1`, `multi_asset_signal_v1` |
| 3 | Profit factor mismatch (109 strategies) | Recompute PF from win_rate and avg_wl; overwrite stated PF in JSON, or audit which field is wrong per strategy using /tmp/phase12_results.json | 109 strategies |
| 4 | Off-by-one flip_count bug | Fix Python: `flip_count < 3` → `flip_count < 4` to match JSON spec | `parabolic_sar_momentum_v1` |
| 5 | Missing condition + look-ahead bias | Add `prev_close` comparison to Python; remove full-day range reference from market profile sizing | `market_profile_poc_v1` |
| 6 | session_end mismatch >5 min | Audit all 10 strategies; determine authoritative version; align JSON and Python | `gap_fade_v1`, `gap_continuation_v1`, `partial_gap_fill_scalp_v1`, and 7 others |
| 7 | Stop/target clustering — 43.8% at (4,7) | Diversify (stop, target) pairs to at least 20–30 unique combinations distributed across the corpus before training | 113 strategies at (4,7); 258 total |
| 8 | Session overlap — 236 concurrent at peak | Introduce session-slot diversity constraints; tag strategies with primary active window; use slot-aware sampling in training batches | All 258 strategies |

### P2 — Fix before final training run

| # | Issue | Action | Affected |
|---|---|---|---|
| 1 | bad_lookback_type (90 indicators, 42 strategies) | Update schema to allow `"session"` and `"recursive"` as valid enum values, or assign sentinel integers; apply consistently | 42 strategies |
| 2 | Warmup divergence (204 flags) | Document 2× warmup convention; add `warmup_multiplier` field to schema or codegen spec | Project-wide |
| 3 | Undocumented Python parameters (12 deep-sample strategies) | Audit all 12 failing strategies; add TunableParams and threshold values to JSON metadata | `elder_impulse_v1`, `gap_and_go_with_volume_v1`, `multi_factor_composite_v1`, `sma_crossover_banknifty_v1`, `vwap_mean_reversion_v19`, and 7 others |
| 4 | Tag normalization | Normalize all `mean-reversion` → `mean_reversion`; recheck combined count (112) against diversity threshold | 27 strategies |
| 5 | Entry asymmetry — ema200_dip_reversal_v1 | Add the 5 missing buy_pe conditions or document that PE entry is intentionally minimal | `ema200_dip_reversal_v1` |
| 6 | vwap_no_dayid bug | Replace manual session loop with standard day_id groupby pattern | `previous_day_high_low_breakout_v1.py` |
| 7 | Retriage delta +1 | Identify which extra strategy was converted (258 actual vs expected with delta=+1); verify no ghost entry | Pipeline-level |

### P3 — Review and document (acceptable if justified)

| # | Issue | Action | Affected |
|---|---|---|---|
| 1 | No VIX filter (116 strategies) | Tag explicitly as `vix_agnostic: true` in JSON; verify by strategy type that absence is intentional | 116 strategies |
| 2 | Early EOD non-intentional (19 strategies) | Document session_end rationale in JSON `notes` field or adjust to standard 15:25 | 19 strategies |
| 3 | Trades/day inconsistency (4 strategies) | Align typical_range_high ≤ max_trades_per_day, or correct max_trades_per_day upward | `ema_crossover_v1`, `inside_bar_breakout_v177`, `spectral_analysis_v1`, `vol_surface_signal_v1` |
| 4 | Win rate sanity (16 strategies) | Review 14 momentum strategies at 0.56–0.60; verify `donchian_channel_breakout_v1` win_rate=0.40 is correct | 16 strategies |
| 5 | R:R mismatch — rsi_divergence_vwap_v1 | Reconcile avg_winner_to_loser=1.2 with declared R:R=1.75; document partial exit logic if applicable | `rsi_divergence_vwap_v1` |
| 6 | Long lookbacks >30 min (14 strategies) | Verify structural justification per strategy; add `lookback_justification` note in JSON for non-obvious cases | 14 strategies |
| 7 | option_chain=true semantic overload (253 strategies) | Add a separate `trades_options: true` boolean field to distinguish signal source from traded instrument; retain option_chain=true | 253 strategies |
| 8 | ou_theta computed_on flag | Confirm ou_theta is OU parameter, not options Greek; no field change needed if confirmed | `mean_reversion_speed_v1` |
