# Ernest Chan's Principles — Applied

## Summary
Nine non-negotiable principles from Ernest P. Chan's *Quantitative Trading* and *Algorithmic Trading*, operationalized for this project. These aren't theoretical — every principle has a specific implementation and every violation was caught during the project.

## The 9 Principles

### 1. Sharpe Ratio Is the Only Metric That Matters
Not total PnL. Not win rate. Sharpe.

**Calculation**:
- For each trading day, sum PnL of all closed trades
- Divide by `capital_per_trade` to get daily return percentage
- Days with ZERO trades get return = 0 (include them — they represent opportunity cost)
- Annualized: `Sharpe = mean(daily_returns) / std(daily_returns, ddof=1) × sqrt(252)`
- If std = 0: Sharpe = 0

**Why**: ₹5L with Sharpe 0.3 is worse than ₹50K with Sharpe 2.0. Sharpe penalizes variance, which means a strategy that makes consistent small profits ranks higher than one with occasional big wins and frequent losses.

### 2. Validate the Thesis BEFORE Optimizing
Run every strategy with DEFAULT parameters through walk-forward CV first. If the raw idea doesn't work across multiple time periods, optimization is just curve-fitting.

**Implementation**: Phase 3a (CV with defaults) must complete before Phase 3b (Optuna optimization). This is the most important architectural decision in the pipeline.

### 3. Parameter Optimization Tunes Thresholds Only
- The "30" in "RSI < 30" is tunable
- The "14" in "RSI(14)" is NOT — that's the strategy's identity
- Indicator structure (which indicators, which lookback periods) is fixed

**Why**: Recomputing indicators for every Optuna trial is expensive. More importantly, optimizing indicator periods is a fast track to overfitting — with enough freedom, you can fit any noise.

**Performance benefit**: Indicators computed ONCE before optimization starts. Each Optuna trial only changes boolean threshold comparisons.

### 4. Walk-Forward / Expanding Window Validation
Never optimize on the full training set and declare victory.

**For 3 years of data (equity)**:
- Fold 1: train Jan-Jun 2022, test Jul-Dec 2022
- Fold 2: train Jan 2022-Dec 2022, test Jan-Jun 2023
- Fold 3: train Jan 2022-Jun 2023, test Jul-Dec 2023
- Fold 4: train Jan 2022-Dec 2023, test Jan-Jun 2024
- Fold 5: train Jan 2022-Jun 2024, test Jul-Dec 2024

Must be profitable (net PnL > 0) in ≥4 of 5 test folds.

**For 12 days of data (options)**: Leave-one-day-out. Train on 11, test on 1, rotate. Must be profitable on ≥9 of 12 test days.

### 5. Minimum Number of Trades
- 50 trades minimum for the full training period
- 30 trades minimum per filtered segment (per-stock, per-fold)
- Parameter combinations producing fewer trades → rejected

Fewer trades = noise, not signal. This is Chan's most practical rule.

### 6. If It's Too Good to Be True, It Is
- Sharpe > 3.0 on training data → almost certainly overfit
- Optimization improving Sharpe > 3× over defaults → flag as POTENTIAL_OVERFIT
- Real edges in liquid markets: Sharpe 1.0-2.0 is excellent

### 7. Out-of-Sample Is Sacred
Test data is NEVER seen during optimization, parameter tuning, stock selection, or model training. The train/test wall is inviolable.

Any code that loads test data before Phase 6 is a bug.

### 8. Every Strategy Needs a Microstructure Mechanism
"RSI oversold on 5-second bars" is not a mechanism. "Option market makers reprice with a 1-3 bar lag after underlying futures move" IS a mechanism.

If you can't explain WHY the edge exists at the structural level, the strategy is noise.

### 9. Data-Snooping Correction
With N strategies tested, some will look good by pure chance.

**Formula**: `required_Sharpe = 0.5 × sqrt(1 + ln(N))` where ln is natural log.

Report this threshold in the leaderboard alongside each strategy's actual Sharpe. Strategies below the threshold are in the noise zone.

## Decisions Made
- Sharpe denominator uses sample std (ddof=1), not population std
- Zero-trade days ARE included in Sharpe calculation (common mistake to exclude them, inflating Sharpe)
- The data-snooping correction is reported but not enforced as a hard filter — it's a visibility tool

## Pitfalls & Anti-patterns
- **Optimizing first, validating second**: This is backwards. Validate thesis with defaults → then optimize only if the thesis works.
- **Reporting gross Sharpe when costs are active**: When the cost model is on, every Sharpe must be after-cost. Three separate bugs found where gross PnL leaked into Sharpe.
- **Excluding zero-trade days from Sharpe**: Inflates Sharpe by hiding inactive days. A strategy that trades 3 days out of 20 but is profitable on those 3 days looks amazing if you exclude the 17 zero-return days.

## Corrections Log
- ~~`log(N)` in Bonferroni formula~~ → `ln(N)` (natural log, not log10)
- ~~Sharpe denominator described as "capital" without specifying which capital~~ → Clarified: use `capital_per_trade` from strategy config
