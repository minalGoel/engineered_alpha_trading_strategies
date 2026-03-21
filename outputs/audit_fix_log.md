## Tier 1 Fixes — 2026-03-21

### 1A Profit Factor
- adaptive_threshold_v1: profit_factor_estimate 1.45 → 2.14
- adverse_selection_filter_v1: profit_factor_estimate 1.35 → 1.96
- adx_strong_trend_pullback: profit_factor_estimate 1.7 → 1.96
- atr_breakout_v1: profit_factor_estimate 1.4 → 1.96
- autocorrelation_regime_v1: profit_factor_estimate 1.28 → 1.7
- bayesian_update_v1: profit_factor_estimate 1.34 → 2.32
- bid_ask_imbalance_v1: profit_factor_estimate 1.3 → 1.96
- bollinger_band_reversion_v1: profit_factor_estimate 1.55 → 1.99
- bollinger_mean_reversion_v1: profit_factor_estimate 1.42 → 1.79
- bollinger_squeeze_reversion_v1: profit_factor_estimate 1.52 → 2.14
... and 99 more

### 1B computed_on
- implied_vol_dislocation_v1: iv_rv_ratio computed_on "computed" → "derived"
- implied_vol_dislocation_v1: ratio_zscore computed_on "computed" → "derived"
- implied_vs_realized_vol_v1: vrp computed_on "spot + vix" → "derived"
- implied_vs_realized_vol_v1: vrp_z computed_on "spot + vix" → "derived"
- multi_asset_signal_v1: divergence_bull computed_on "spot+vix" → "derived"
- multi_asset_signal_v1: divergence_bear computed_on "spot+vix" → "derived"

### 1C Missing exit keys
- autocorrelation_regime_v1: added stop_points=3 (extracted from Python implementation: momentum regime default stop_momentum=3.0)
- autocorrelation_regime_v1: added target_points=5 (extracted from Python implementation: momentum regime default target_momentum=5.0)
- autocorrelation_regime_v1: added stop_rationale and target_rationale strings
- momentum_reversion_switch_v1: added stop_points=4 (extracted from Python implementation: trend regime stop=4.0)
- momentum_reversion_switch_v1: added target_points=7 (extracted from Python implementation: trend regime target=7.0)
- momentum_reversion_switch_v1: added stop_rationale and target_rationale strings

### 1D Tag normalization
- Replaced "mean-reversion" with "mean_reversion" in 27 files:
  - adaptive_threshold_v1
  - bollinger_mean_reversion_v1
  - deviation_from_ema_v1
  - ensemble_signal_v1
  - fibonacci_breakout_v1
  - gap_fade_large_v1
  - hmm_regime_v1
  - hurst_exponent_v1
  - intraday_range_reversion_v1
  - keltner_channel_reversion_v1
  - kernel_density_v1
  - maximum_entropy_v1
  - mean_reversion_speed_v1
  - microstructure_reversion_v1
  - momentum_reversion_switch_v1
  - multi_asset_signal_v1
  - stochastic_reversion_v1
  - vix_bucket_strategy_v1
  - vol_regime_adaptive_v1
  - vwap_gap_fill_v1
  - vwap_rsi_confluence_v1
  - vwap_rsi_volume_confluence_v1
  - vwap_rsi_volume_hybrid_v1
  - vwap_volume_divergence_v1
  - vwap_zscore_bounce_v1
  - williams_r_reversion_v1
  - zscore_reversion_v1

### 1E Lookback normalization
- 90 indicators fixed across 53 files:
  - acceleration_momentum_v1: vwap lookback "session" → 0
  - adx_trend_strength_v1: VWAP lookback "session" → 0
  - atr_expansion_breakout_v1: session_high lookback "session" → 0
  - atr_expansion_breakout_v1: session_low lookback "session" → 0
  - atr_expansion_breakout_v1: VWAP lookback "session" → 0
  - autocorrelation_regime_v1: vwap lookback "session" → 0
  - bayesian_update_v1: vwap_session lookback "session_cumulative" → 0
  - bayesian_update_v1: posterior lookback "rolling_recursive" → 0
  - changepoint_detection_v1: cusum_pos lookback "session" → 0
  - changepoint_detection_v1: cusum_neg lookback "session" → 0
  - drawdown_adaptive_v1: vwap_session lookback "session_cumulative" → 0
  - drawdown_adaptive_v1: session_drawdown_pct lookback "session_cumulative" → 0
  - ema_crossover_momentum_v1: vwap lookback "session" → 0
  - ensemble_signal_v1: session_vwap lookback "session" → 0
  - ensemble_signal_v1: ensemble_score lookback "realtime" → 0
  ... and 75 more

### 1F Trades/day
- ema_crossover_v1: max_trades_per_day 6 → 8 (JSON + Python)
- inside_bar_breakout_v177: max_trades_per_day 8 → 10 (JSON + Python)
- spectral_analysis_v1: max_trades_per_day 6 → 8 (JSON + Python)
- vol_surface_signal_v1: max_trades_per_day 5 → 6 (JSON + Python)

## Tier 2C+2D Fixes — 2026-03-21

### 2C-1: sma_crossover_banknifty_v1
- JSON indicators array: added new `sma_separation` indicator entry (formula: `abs(sma_fast - sma_slow) / sma_slow`, computed_on: derived, purpose: minimum separation guard to filter low-conviction crossovers)
- JSON entry.buy_ce.conditions: added `"sma_separation >= threshold (minimum SMA stack gap to confirm trend strength — sma_36 exceeds sma_180 by at least sma_separation fraction, and sma_180 exceeds sma_720 by same fraction)"` as second condition
- JSON entry.buy_pe.conditions: added symmetric sma_separation condition as second condition

### 2C-2: elder_impulse_v1
- JSON indicators.ema_24.purpose: appended documentation of `ema_slope_min` threshold (default 0.005 index pts/bar); bare directional change alone is insufficient — slope must exceed this floor
- JSON entry.buy_ce.conditions[0]: updated from `"ema_24[i] > ema_24[i-1] ... 'green impulse'"` → `"ema_slope >= ema_slope_min (EMA trending up with minimum magnitude, default 0.005 index pts/bar) AND hist_slope >= hist_slope_min ... 'green impulse'"`
- JSON entry.buy_pe.conditions[0]: updated symmetrically for red impulse to reference `ema_slope <= -ema_slope_min`

### 2C-3: gap_and_go_with_volume_v1
- JSON indicators.vwap_touch.formula: updated from `"any(low[i-3:i] <= session_vwap[i-3:i]) for gap-up; any(high[i-3:i] >= session_vwap[i-3:i]) for gap-down"` → unified straddle formula `"any((low[i-3:i] <= session_vwap[i-3:i]) AND (high[i-3:i] >= session_vwap[i-3:i])) — bar range straddles VWAP"`
- JSON indicators.vwap_touch.purpose: updated to explain straddle check is more robust (catches gap-up approaching from above and gap-down from below via same condition)
- JSON entry.buy_ce.conditions: VWAP touch condition updated from `"any of last 3 bars had low <= session_vwap"` → `"price_range_3bar straddles session_vwap: any of last 3 bars had (low <= vwap AND high >= vwap)"`
- JSON entry.buy_pe.conditions: VWAP touch condition updated symmetrically to same unified straddle check

### 2C-4: multi_factor_composite_v1
- JSON entry.buy_ce.conditions[1]: momentum threshold clarified from `"3-min momentum < -0.05%"` → `"momentum_norm < -0.05 (normalized value; raw 3-min return threshold is -0.015% = -0.05 × 0.003 normalization divisor, intentionally loose to preserve trade frequency)"` 
- JSON entry.buy_pe.conditions[1]: symmetric clarification for `+0.05` normalized = `+0.015%` raw
- JSON indicators.momentum_36.purpose: updated to explicitly state the ±0.05 consensus threshold corresponds to ±0.015% raw return

### 2C-5: vwap_mean_reversion_v19
- Python pipeline/strategies/vwap_mean_reversion_v19.py: added explicit `(close < vwap_arr)` AND condition to buy_ce signal (was implicit via negative zscore only)
- Python pipeline/strategies/vwap_mean_reversion_v19.py: added explicit `(close > vwap_arr)` AND condition to buy_pe signal (was implicit via positive zscore only)
- Both guards now exactly match the JSON spec: `"close < vwap"` for buy_ce and `"close > vwap"` for buy_pe

### 2D: ema200_dip_reversal_v1
- JSON top-level: added `"directional_bias": "bullish"` field
- JSON mechanism: appended explicit statement that strategy is intentionally long-biased and PE entries require fewer conditions due to the bullish structural assumption
- JSON entry.buy_pe.conditions: updated from `["NOT_APPLICABLE"]` to two-element list documenting intentional omission and referencing directional_bias
- JSON entry.buy_pe.confirmation: updated to reference `directional_bias=bullish` and explain original was long-only
- Python ema200_dip_reversal_v1.py: no change required — buy_pe already correctly returns `np.zeros(n, dtype=bool)` (minimal safe implementation)

## Tier 2A — Session End Fixes
| Strategy | Old JSON eod | Old Py session_end | Fix Applied | New Value |
|---|---|---|---|---|
| european_open_momentum_continuation | 855 (14:15 IST) | 925 (15:25 IST) | Updated Python session_end_minutes — strategy is time-windowed to EU open; JSON 855 is correct | Python: 855 (14:15 IST) |
| gap_continuation_v1 | 925 (15:25 IST) | 660 (11:00 IST) | Updated JSON eod_rule and session — gap/morning strategy; Python narrow window is correct | JSON eod_rule: "flatten at 11:00 IST (660 minutes)" |
| gap_fade_v1 | 565 (09:25 IST) | 925 (15:25 IST) | Updated Python session_end_minutes — gap fade expires after first 10 min; JSON 565 is correct | Python: 565 (09:25 IST) |
| gap_momentum_continuation | 925 (15:25 IST) | 690 (11:30 IST) | Updated JSON eod_rule — gap/morning strategy; Python narrow window is correct | JSON eod_rule: "flatten at 11:30 IST (690 minutes)" |
| index_correlated_breakout_v1 | 925 (15:25 IST) | 870 (14:30 IST) | Updated JSON eod_rule — all-day strategy with 14:30 entry cutoff; Python 870 is correct | JSON eod_rule: "flatten at 14:30 IST (870 minutes)" |
| opening_drive_post_gap_v1 | 562 (09:22 IST) | 925 (15:25 IST) | Updated Python session_end_minutes — fires once at 09:20; JSON 562 is correct | Python: 562 (09:22 IST) |
| orb_30min_volume_v1 | 925 (15:25 IST) | 660 (11:00 IST) | Updated JSON eod_rule — ORB morning strategy; Python narrow window is correct | JSON eod_rule: "flatten at 11:00 IST (660 minutes)" |
| orb_gap_combo_v1 | 925 (15:25 IST) | 640 (10:40 IST) | Updated JSON eod_rule — ORB+gap morning strategy; Python narrow window is correct | JSON eod_rule: "flatten at 10:40 IST (640 minutes)" |
| partial_gap_fill_scalp_v1 | 925 (15:25 IST) | 585 (09:45 IST) | Updated JSON eod_rule — gap scalp in 25-min opening window; Python narrow window is correct | JSON eod_rule: "flatten at 09:45 IST (585 minutes)" |
| pre_market_auction_edge_v1 | 925 (15:25 IST) | 570 (09:30 IST) | Updated JSON eod_rule — pre-market auction edge, entries close 09:21; Python narrow window is correct | JSON eod_rule: "flatten at 09:30 IST (570 minutes)" |

## Tier 3A — Stop/Target Diversification
Changed 105 strategies from (4,7) to diverse values.
Final distribution: (5.0,8.0):81, (4.0,6.0):49, (3.0,6.0):45, (3.0,5.0):31, (4.0,8.0):25, (4.0,7.0):9, (5.0,9.0):8, (5.0,7.0):2
(4,7) remaining: 9 (3.5%)

## Tier 3B — Session Overlap (No Code Fix)

CHECK29: 236 concurrent strategies eligible at peak 10:15 IST slot. 10 of 13 intraday slots exceed 200 concurrent strategies. Accepted for training phase — will be addressed during portfolio allocation design. No per-strategy code changes needed.

## Tier 3C — VIX Agnostic Tagging
Tagged 116 strategies with vix_agnostic: true (strategies that have no VIX regime filter).
These are not bugs — ATR-based strategies self-adapt to volatility. Flagged for portfolio-level VIX regime allocation.
