# Pre-Backtest Strategy Audit Prompt
## Distilled from Ernest P. Chan: Quantitative Trading, Algorithmic Trading, Machine Trading
## Calibrated for NSE/BSE intraday microstructure regime

---

**INSTRUCTION:** Before backtesting or deploying any strategy, run it through every check below. Answer each question explicitly. If any HARD KILL fires, stop immediately — do not optimize, do not proceed. Flag SOFT WARNINGS for manual review before continuing.

---

## 1. DATA INTEGRITY AUDIT

### 1.1 Survivorship Bias [QT Ch3, AT Ch1]
- [ ] Does the dataset include delisted/suspended instruments for the backtest period?
- [ ] If using index constituents (NIFTY50, NIFTY200), does the universe reflect point-in-time membership, not current membership?
- [ ] **HARD KILL:** Long-only mean-reversion strategy on stocks using survivorship-biased data → results are fiction. Chan demonstrates returns flip from -42% (real) to +388% (biased) on identical strategy.

### 1.2 Split/Dividend/Corporate Action Adjustment [QT Ch3, AT Ch1]
- [ ] Are all prices adjusted for splits, dividends, and bonuses with correct ex-date alignment?
- [ ] For NSE: Are F&O lot size changes reflected? (NIFTY changed from 75→65; BANKNIFTY 25→30→discontinued weeklies)
- [ ] Does the adjustment use multiplicative factors (preserving returns), not subtractive (preserving price changes)?

### 1.3 Price Source Fidelity [AT Ch1, MT Ch6, Box 6.2]
- [ ] **For limit order strategies:** Are you using BBO (best bid/offer) data, not last-traded price? Last-traded at a different tick can be stale by seconds on illiquid NSE names.
- [ ] **For market-on-close strategies:** Are you using auction closing price from the primary exchange, not the consolidated last trade? Consolidated close inflates mean-reversion backtests.
- [ ] **For intraday strategies:** Minimum data = BBO at your strategy frequency. 1-min OHLC alone is insufficient — it hides spread, depth, and fill dynamics.
- [ ] **For spread/pairs strategies:** Are both legs' prices contemporaneous? Asynchronous last-traded prices on two instruments create phantom spreads that inflate returns.
- [ ] **SOFT WARNING:** High/Low prices are the noisiest field in daily data. Any strategy using intraday H/L for signal generation will have inflated backtest returns because fills at those prices are unreliable.

### 1.4 Tick Data Quality [MT Ch6]
- [ ] Have you verified there are no missing ticks, especially at open/close auctions?
- [ ] Are timestamps monotonically increasing? Are duplicate timestamps handled (take last)?
- [ ] For NSE TBT data: Is the source Dhan/Fyers/TrueData? Is claimed latency verified by measurement?
- [ ] **Run a sanity check:** Compute daily returns from your data. Flag any day with |return| > 4σ that has no corresponding news event or market-wide move → suspect data error.

### 1.5 Futures Continuous Contract [AT Ch1]
- [ ] If using continuous contract, which back-adjustment method? Price-adjusted (correct for P&L) or return-adjusted (correct for returns/ratios)?
- [ ] Does your strategy use price differences → must use price-adjusted. Price ratios → must use return-adjusted. You cannot have both correct simultaneously.
- [ ] Are rollover dates defined consistently (OI crossover, N days before expiry)? Different rollover rules produce different backtests.
- [ ] Do back-adjusted prices go negative in the distant past? If so, add a constant floor.

---

## 2. LOOK-AHEAD BIAS AUDIT [QT Ch3, AT Ch1]

### 2.1 Signal Generation
- [ ] Every signal at time t uses only data available at or before time t-1 (for bar-close strategies) or t (for within-bar strategies). No exceptions.
- [ ] If the strategy uses regression/ML model parameters fitted on the full dataset → **HARD KILL.** Parameters must be fitted on rolling or expanding windows using only past data.
- [ ] Moving averages, z-scores, Bollinger bands: Is the lookback window computed from lagged data only?
- [ ] Any use of daily high/low within the same day for entry signals → **HARD KILL.** You cannot know the day's high/low before market close.

### 2.2 The Truncation Test [QT Ch3]
Run this mechanical check:
1. Run backtest on full data, save position file A.
2. Truncate the last N days (N=20-100), run again, save position file B.
3. Truncate last N rows from A.
4. Compare A and B. If they differ → you have look-ahead bias. Find and fix it.

### 2.3 Unified Backtest/Live Code [AT Ch1, MT Ch1]
- [ ] Can the same code run on historical data (backtest) and live data (execution) by switching the data source only? If not, divergence between backtest and live logic is inevitable.
- [ ] **Chan's strongest recommendation across all three books:** Same code for backtest and execution. This eliminates the most common class of look-ahead and logic bugs.

---

## 3. DATA-SNOOPING / OVERFITTING AUDIT [QT Ch3, AT Ch1, MT Ch4]

### 3.1 Parameter Count Gate
- [ ] Count ALL free parameters: entry thresholds, exit thresholds, lookback windows, holding periods, filter conditions, indicator periods, stop/target levels.
- [ ] **Chan's rule of thumb:** Minimum data points needed = 252 × number_of_free_parameters (for daily). For 5-second bars: scale by bars_per_day.
- [ ] **HARD KILL:** >5 free parameters on <3 years of daily data. For intraday at 5s: >5 parameters on <3 months of data.
- [ ] **Qualitative decisions count as parameters.** Whether to enter at open vs close, whether to trade large-cap vs mid-cap, whether to hold overnight — each choice optimized on the same data is an implicit parameter.

### 3.2 Out-of-Sample Validation
- [ ] Is there a clean train/test split (minimum 50/50, acceptable 70/30)?
- [ ] Was the test set EVER used to adjust parameters, filter conditions, or qualitative decisions? If yes → your test set is now in-sample. You need new OOS data.
- [ ] **Chan's warning:** "True out-of-sample testing cannot really begin until a strategy is published and cast in stone." If you keep tweaking after seeing OOS results, you've contaminated the test set.

### 3.3 Walk-Forward / Rolling Validation
- [ ] Is the model trained on a rolling window and tested on the subsequent period, repeatedly? This is strictly superior to a single train/test split.
- [ ] **Parameterless models preferred [QT Ch3]:** All parameters dynamically optimized in a rolling window → eliminates fixed-parameter overfitting. If computationally feasible, use this.

### 3.4 Cross-Validation [MT Ch4]
- [ ] For ML/AI strategies: Is K-fold cross-validation used (K=5 or K=10)?
- [ ] Is the model selected by best OOS performance across folds, not best in-sample?
- [ ] **HARD KILL:** In-sample CAGR >> out-of-sample CAGR (e.g., 48% vs negative) → severe overfitting. Chan demonstrates this explicitly with 112-factor model in MT Ch2.

### 3.5 Sensitivity Analysis [QT Ch3]
- [ ] Vary each parameter ±20% from optimal. Does performance degrade gracefully or collapse?
- [ ] **If any single parameter set other than the optimal is unacceptable → the model is data-snooped.**
- [ ] Systematically remove entry/exit conditions one by one. Does OOS performance track IS performance? If removing a condition hurts IS but doesn't hurt OOS → that condition is noise-fitted.
- [ ] **After reducing parameters:** Consider averaging across multiple parameter sets rather than picking one optimal set. This further insulates against snooping.

### 3.6 Linearity Preference [AT Ch1]
- [ ] Can the strategy be expressed as a linear model? If yes, use the linear version. Nonlinear models with the same number of parameters are strictly more susceptible to overfitting.
- [ ] **Chan's principle:** "If a linear approximation exists, a good reason must be given for why it cannot be used."
- [ ] For ranking/stock-selection: Use equal-weight factor combination (Kahneman's insight in AT Ch1) before optimizing weights. Equal weights are "often superior because they are not affected by accidents of sampling."

### 3.7 ML-Specific Overfitting Checks [MT Ch4]
- [ ] For decision trees: Is MinLeafSize large enough (≥100)? Using all leaves for prediction instead of just extreme leaves → overfitting.
- [ ] For neural networks: Minimum architecture first (1 hidden layer, 1 neuron). More layers/neurons = higher IS, lower OOS.
- [ ] For ensemble methods (random forest, bagging): These REDUCE overfitting. Use them. Boosting does NOT reliably reduce overfitting (Chan: MT Ch4).
- [ ] For SVM: Linear kernel first. Nonlinear kernels (polynomial, RBF) add flexibility and overfitting risk.
- [ ] **Apply the Deflated Sharpe Ratio [de Prado]:** Adjust Sharpe for number of strategies/parameters tested. If DSR < 0.5 → don't proceed.

---

## 4. TRANSACTION COST AUDIT [QT Ch2-3, AT Ch1, MT Ch6]

### 4.1 Cost Model Completeness
- [ ] Are ALL of the following included in the backtest cost model?

| Component | Included? | NSE Reference Rate |
|---|---|---|
| Brokerage | | ₹20/order flat (Zerodha) |
| STT (sell-side) | | 0.05% futures, 0.15% options premium |
| Exchange transaction charges | | 0.00173% equity/futures, 0.053% options |
| SEBI turnover fee | | 0.0001% |
| Stamp duty (buy-side) | | 0.003% F&O |
| GST on (brok+exch+SEBI) | | 18% |
| Bid-ask spread | | ₹0.30/side ATM options, ₹0.05/side futures |
| Slippage (1 tick minimum) | | 1 tick additional per side |
| Market impact (for order > 20% of top-of-book) | | Model separately |
| Adverse selection cost | | 1-2 bps for intraday strategies |

- [ ] **HARD KILL:** Strategy with Sharpe ratio of 4.47 before costs can become unprofitable after costs. Chan demonstrates this with the Khandani-Lo mean-reversion model [QT Ch3 Example 3.7]. If you haven't modeled costs, your backtest is meaningless.

### 4.2 Cost-Aware Kill Condition
- [ ] Compute: `expected_edge_bps = gross_return_bps - total_RT_cost_bps`
- [ ] **HARD KILL:** `expected_edge_bps < 0` → strategy is dead. Do not optimize. Find a different strategy.
- [ ] **SOFT WARNING:** `expected_edge_bps < 2 * total_RT_cost_bps` → edge/cost ratio < 2:1. Fragile. Any cost increase, spread widening, or slippage degradation kills it.

### 4.3 Execution Realism for NSE
- [ ] At your order size, does it exceed top-of-book depth? If yes, you'll walk the book. Model multi-level impact.
- [ ] For options: ATM weekly options have wider spreads near expiry. Is this time-varying spread modeled?
- [ ] For illiquid names: Does the strategy check minimum daily volume before trading?
- [ ] **Chan [MT Ch6]:** "Even AAPL has an average top-of-book quote size of only 189 shares at Nasdaq." NSE depth is thinner. Budget accordingly.

---

## 5. FILL SIMULATION AUDIT [MT Ch6]

### 5.1 Order Type Specification
- [ ] Does every entry/exit specify: order type (limit/market/MOC), expected fill price assumption, and fill probability?
- [ ] **For market orders:** Fill at ask (buy) or bid (sell), not midprice. Midprice fills are a fantasy for market orders.
- [ ] **For limit orders:** Fill is NOT guaranteed. You must model:
  - Queue position (time priority on NSE)
  - Fill probability (function of distance from touch, order size, and market volatility)
  - Partial fills and their impact on position sizing
  - **Adverse selection:** Limit orders that DO get filled are disproportionately losers [MT Ch6]. The profitable limit orders never get touched because the market moves away. If your backtest assumes 100% fill rate on limit orders, it's inflated.

### 5.2 Backtest Engine Type
- [ ] **Is the backtest event-driven (tick-by-tick or bar-by-bar with sequential processing)?** If yes → proceed.
- [ ] **Is the backtest vectorized (all signals computed simultaneously across time)?** If yes → **SOFT WARNING.** Vectorized backtests cannot model: queue position, partial fills, market impact feedback, order book state changes, or realistic execution timing. They systematically inflate performance for HFT and intraday strategies.
- [ ] **For limit order strategies specifically:** Static order book snapshots are NOT sufficient. You need the stream of historical order book messages (add/cancel/fill events) to simulate queue position and fill probability [MT Ch6].

### 5.3 IOC and Adverse Selection
- [ ] If using limit orders: Have you measured the difference between P&L of filled orders vs P&L of unfilled orders? If filled orders systematically lose more → you have adverse selection [MT Ch6].
- [ ] Consider: Limit IOC (immediate-or-cancel) orders eliminate adverse selection but sacrifice fill rate. Is this tradeoff modeled?

---

## 6. REGIME CHANGE / STRUCTURAL BREAK AUDIT [AT Ch1, MT Ch1]

### 6.1 Known Regime Shifts (NSE-Specific)
Has the backtest period been checked against these known structural breaks?

- [ ] **April 2025:** STT hike on options (0.1% → 0.15%). Any pre-April 2025 options backtest uses wrong cost rates.
- [ ] **November 2024:** SEBI weekly expiry restriction to one per exchange. BANKNIFTY weeklies discontinued. Changes option microstructure.
- [ ] **February 2024:** SEBI peak margin norms tightened. Affects intraday leverage.
- [ ] **COVID March 2020:** VIX 87, circuit breakers, spread blowouts. Tail risk test.
- [ ] **Decimalization / tick size changes:** Any historical tick size changes in your universe?
- [ ] **Market hours changes, auction mechanism changes, lot size changes.**

### 6.2 Regime Robustness
- [ ] Run backtest on 2+ distinct market regimes (trending, mean-reverting, high-vol, low-vol). Does the strategy perform in ALL regimes or only one?
- [ ] **Chan [AT Ch1]:** "Strategies that performed superbly prior to each of these regime shifts may stop performing and vice versa. Backtests done using data prior to such regime shifts may be quite worthless."
- [ ] **SOFT WARNING:** If >80% of P&L comes from a single regime period → strategy is regime-dependent, not robust.

---

## 7. CAPACITY AND SCALING AUDIT [MT Ch1, MT Ch6]

### 7.1 Capacity Estimation
- [ ] What is the average daily volume of the instrument(s) traded?
- [ ] What % of that volume does your strategy consume at target AUM?
- [ ] **Rule of thumb:** If your order is >1% of average daily volume, market impact will materially erode returns.
- [ ] At ₹50L capital: Is each order < 20% of top-of-book depth? If not, model multi-level book impact.
- [ ] At projected ₹5Cr capital: Redo the above. Does the strategy still work?

### 7.2 Alpha Decay [MT Ch2]
- [ ] "Every time someone finds alpha, they have to keep it a secret. And still, more and more people will discover it eventually, and the alpha diminishes" [MT Ch2].
- [ ] Is the edge structural (e.g., VRP exists because of risk transfer) or statistical (e.g., pattern in returns)? Statistical edges decay faster.
- [ ] Plot rolling Sharpe ratio across the backtest period. Is it stable, declining, or regime-dependent?

---

## 8. PERFORMANCE MEASUREMENT AUDIT [QT Ch3, MT Ch1]

### 8.1 Sharpe Ratio Calculation
- [ ] For dollar-neutral / self-financing strategies: Do NOT subtract risk-free rate [QT Ch3].
- [ ] For day-trading (no overnight): Do NOT subtract risk-free rate.
- [ ] Annualization: `Sharpe_annual = sqrt(N_T) × Sharpe_period`, where N_T = number of trading periods per year. For hourly on NSE: N_T = 252 × 6.25 = 1575. NOT 252 × 24.
- [ ] **Chan's live trading rule of thumb [AT Ch1]:** "Most traders would be happy to find that live trading generates a Sharpe ratio better than half of its backtest value." If your backtest Sharpe is 1.0, expect 0.5 live.

### 8.2 Beyond Sharpe
- [ ] **Calmar ratio [MT Ch1]:** CAGR / |max drawdown over 3 years|. Target ≥ 1.0. Better than Sharpe for tail risk because it doesn't assume Gaussian returns.
- [ ] Maximum drawdown and maximum drawdown duration computed and acceptable?
- [ ] **Kelly leverage [MT Ch1]:** Compute it, then USE LESS. "Kelly's leverage is typically too high to be of practical use." A 5x Kelly on a strategy with a -20% daily tail risk = account wipeout. Lower leverage until max DD in backtest (including crises) is tolerable.

### 8.3 Statistical Significance
- [ ] How many independent trades in the backtest? If < 100 → results are not statistically significant regardless of Sharpe.
- [ ] **Compute p-value of the Sharpe ratio.** A Sharpe of 1.5 on 30 trades is not significant. A Sharpe of 0.8 on 3000 trades is.
- [ ] Apply Deflated Sharpe Ratio to adjust for multiple testing.

---

## 9. FINAL GATE CHECKLIST

Before proceeding to paper trading, every strategy must pass ALL of these:

| Gate | Criterion | Pass? |
|---|---|---|
| Data clean | No survivorship bias, no look-ahead, correct adjustments | |
| Cost-inclusive Sharpe | After-cost Sharpe ≥ 0.5 on backtest | |
| OOS validation | Walk-forward or OOS Sharpe ≥ 50% of IS Sharpe | |
| Parameter parsimony | ≤ 5 free parameters, sensitivity analysis passed | |
| Fill realism | Fill model matches strategy order type; adverse selection budgeted | |
| Regime robustness | Profitable (or at least not catastrophic) across 2+ regimes | |
| Capacity | Order size < 20% of top-of-book at target AUM | |
| Kill condition defined | Specific metric + threshold + timeframe for killing the strategy live | |
| Edge > 2× cost | Gross edge in bps > 2× round-trip cost in bps | |

**If ANY gate fails → do not paper trade. Fix the issue or kill the strategy.**

---

## 10. ANTI-PATTERNS — INSTANT RED FLAGS

If you see any of these in a strategy specification, it's broken until proven otherwise:

1. **"Backtest assumes mid-price fills"** on a strategy using market orders → inflated by half the spread on every trade.
2. **"Transaction costs excluded / to be added later"** → the most common way to waste months on a dead strategy.
3. **"Optimized on 2 years of data with 12 parameters"** → pure noise-fitting. Chan's rule: need 252×12 = 3024 daily data points minimum.
4. **"Works great on NIFTY, haven't tested on other instruments"** → likely data-snooped to this one series.
5. **"Sharpe ratio of 5+ in backtest"** → either a bug, look-ahead bias, or the cost model is missing. Almost no legitimate strategy has Sharpe > 3 after costs at retail infra.
6. **"100% fill rate on limit orders"** → ignores queue position, adverse selection, and opportunity cost. Real fill rates on passive orders at the touch are 20-60% depending on conditions.
7. **"Vectorized backtest of a microstructure strategy"** → vectorized backtests are fine for daily rebalancing; they are INVALID for tick/second-level strategies where fill dynamics matter.
8. **"ML model trained on full dataset, tested on same dataset"** → not a backtest, it's a curve fit.
9. **"Spread = 0 assumption"** → reverses the sign of returns for any strategy that crosses the spread on every trade.
10. **"Works because of [single indicator] > [threshold]"** → if the edge is this simple and this obvious, it's been arbed away. What's the microstructural or behavioral reason it persists?

---

*This audit prompt synthesizes Chan's three books with NSE-specific calibration from the master cost model. Use it as a pre-flight checklist before committing compute or capital to any strategy.*
