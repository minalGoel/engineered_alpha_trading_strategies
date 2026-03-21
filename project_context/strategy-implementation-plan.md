# Strategy Implementation Plan

## Last Updated: 2026-03-21

## Portfolio Architecture

7 strategies, ERC allocation, vol targeting 15% annualized, 0.25× Kelly → 0.5× Kelly as track record builds. Portfolio Sharpe target: 1.5-2.0.

---

## S1: NIFTY 30-Minute ORB (Option Selling)

**Thesis:** First 30 minutes establish a range from overnight information absorption. Breakout = institutional continuation. Sell the opposite side (breakout up → sell PE, breakout down → sell CE) to capture both direction and theta.

**Specs:**
- Signal: 30-min high/low of NIFTY spot (09:15-09:45)
- Entry: Sell ATM/slightly-OTM option on opposite side when spot breaks range with volume confirmation
- Instrument: NIFTY monthly options (not weekly — lower STT drag on premium)
- Hold: 45-180 min. Exit on range re-entry, time stop at 13:00, or 50% premium decay
- Sizing: 2-4 lots per trade at ₹50L; scale with capital
- Stop: Buy back at 2× premium collected
- VIX filter: Skip when VIX < 12 (range too narrow, breakouts fail)
- Max trades/day: 3 (1 initial + 2 re-entries if range extends)

**Evidence:** Zerodha Substack study (Jan 2022-Feb 2026) — all years profitable, max DD ~6%. Wang & Gangwar (2025, SSRN) — block-based breakout optimization on NIFTY. Multiple independent backtests agree.

**Expected:** Sharpe 0.8-1.2, win rate 55-71%, profit factor 1.4-1.8.
**Kill condition:** After-cost Sharpe < 0.5 over 60 trading days.
**Confidence:** MEDIUM-HIGH.

---

## S2: Overnight Gap Capture (NIFTY Futures)

**Thesis:** 92% of NIFTY up-moves since 2011 occur overnight (Zerodha analysis, 2000-2025). Pre-open call auction (introduced Oct 2010) structurally shifted information incorporation to the overnight session. Victor Haghani called this "the grandmother of all market anomalies." India-specific — US shows the opposite.

**Specs:**
- Entry: Buy NIFTY futures at 15:15-15:25 (before close)
- Exit: Sell at 09:20-09:25 (after open, post-auction)
- Instrument: NIFTY near-month futures (tightest spread, deepest book)
- Hold: ~18 hours (overnight)
- Sizing: 1-2 lots at ₹50L (margin ~₹2.5L/lot); scale with capital
- Stop: No intraday stop (overnight, can't manage). Position-level stop: -3% of capital → pause 5 days
- Regime filter: Skip when VIX > 25 (overnight gap risk too high)
- Skip: Day before budget, RBI policy, election results

**Evidence:** Zerodha data 2000-2025. Globally documented (Cliff et al., "Night Moves"). India uniquely strong.
**Expected:** Sharpe 1.0-1.5, win rate ~52-55% but positive skew (big up-gaps > big down-gaps).
**Kill condition:** Negative PnL over 40 consecutive trading days.
**Confidence:** HIGH — 25 years of data, structural explanation.

---

## S3: VWAP Mean Reversion

**Thesis:** On range-bound days (~40% of sessions), price deviating >1-2% from intraday VWAP with exhaustion signal reverts to VWAP. Complements ORB — works when ORB fails.

**Specs:**
- Signal: Price deviation >1.5% from session VWAP + RSI divergence or volume exhaustion
- Entry: Limit order at deviation extreme
- Exit: VWAP touch or 50% of deviation distance
- Instrument: NIFTY futures (15-90 min hold, futures cheaper than options here) or ATM options
- Hold: 15-90 min (avg ~30-45 min)
- Sizing: 1-2 lots at ₹50L
- Regime filter: Only trade when VIX < 15 AND opening range is narrow (< 50% of 20-day average range)
- Time filter: Best in first and last trading hours (OneTradeJournal finding)

**Evidence:** IJCRT 2025 — VWAP + OI on BankNifty, 65% accuracy, 1:2 R:R, 53 trades. OneTradeJournal (2024-25) — consistent profitability across NIFTY, BankNifty, Reliance, TCS.
**Expected:** Sharpe 0.6-1.0, valid signals on ~22-24% of trading days.
**Kill condition:** After-cost Sharpe < 0.3 over 60 days.
**Confidence:** MEDIUM — globally proven concept, India-specific evidence thin.

---

## S4: Cross-Sectional Equity Momentum

**Thesis:** Momentum is India's strongest documented factor. Nifty200 Momentum 30 delivers ~19% CAGR (backtested). Capitalmind PMS does 30.2% with adaptive momentum. Quality factor (QMJ) adds 10%+ alpha per IIM-A.

**Specs:**
- Universe: NIFTY 200 constituents
- Signal: 6-month + 12-month returns, volatility-adjusted (same methodology as Nifty200 Momentum 30 index)
- Rebalance: Monthly (semi-annual misses momentum turns)
- Portfolio: Top 25-30 stocks by momentum score
- Execution: Equity delivery via Dhan/Upstox
- Sizing: ₹10L allocation (20% of ₹50L), ~₹30-40K per stock
- Tax optimization: 60% via UTI Nifty200 Momentum 30 Index Fund (tax-efficient core, no rebalance tax) + 40% self-managed overlay
- Turnover: ~20-40% monthly → cost ~3-5% annually from STT + STCG 20%

**Evidence:** IIM-A Four-Factor Model (Agarwalla, Jacob & Varma, 1994-2025) — WML 21.9% annualized. S&P DJI Factor Study India (2005-2022) — Momentum #1 factor. Capitalmind PMS 30.2% CAGR 5yr live. -31.25% drawdown Sep 2024-Mar 2025 (momentum crash risk is real).
**Expected:** CAGR 18-25%, Sharpe 0.8-1.2, max DD 25-35%.
**Kill condition:** Underperform Nifty 50 by >15% over 12 months.
**Confidence:** HIGH — strongest evidence base of any strategy.

---

## S5: Expiry Day Structures

**Thesis:** NIFTY weekly expiry (Tuesday) shows structural vol crush + gamma effects as OI concentrates at max-pain strikes. Sell straddles/strangles in the morning, buy back as premium decays.

**Specs:**
- Entry: Sell ATM straddle or 100-pt-wide strangle at 09:30-10:00 on Tuesday
- Exit: Buy back at 60-70% premium decay, or at 14:30 (time stop)
- Instrument: NIFTY weekly options (Tuesday expiry)
- Hold: 4-6 hours
- Sizing: 1-2 lots each side at ₹50L (margin ~₹4-5L for strangle)
- Stop: Buy back at 1.5× premium collected
- Hedge: Consider buying far-OTM wings (iron condor) to cap tail risk
- Skip: Expiry weeks with RBI/budget/major event overlap

**Evidence:** Post-Oct-2024 regime (single weekly expiry) has concentrated all expiry effects on Tuesday. Max-pain convergence documented. Zerodha data shows expiry-day vol crush.
**Expected:** Sharpe 0.5-1.0, high win rate (70-80%) but occasional large losses.
**Kill condition:** 3 consecutive expiry-day losses exceeding stop.
**Confidence:** MEDIUM — pattern is real but post-Oct-2024 sample is small.

---

## S6: Volatility Premium (Iron Condors)

**Thesis:** India VIX persistently exceeds realized vol by 3-5 percentage points. Sell monthly NIFTY iron condors to harvest the variance risk premium.

**Specs:**
- Structure: Sell 200-pt-wide iron condor on NIFTY monthly options
  - Sell CE at +300 pts from spot, buy CE at +500 pts
  - Sell PE at -300 pts from spot, buy PE at -500 pts
- Entry: 15-20 DTE (2-3 weeks before monthly expiry)
- Exit: Buy back at 50% of credit received, or at 5 DTE (roll to next month)
- Sizing: 2-4 condors at ₹50L (margin ~₹2-3L per condor)
- Adjustment: If short strike breached, roll tested side out by 100 pts
- Stop: Max loss = width of spread - credit received per lot

**Evidence:** India VIX > realized vol is structural. Post-Oct-2024 regime: larger lots, higher STT, single weekly expiry reduces income frequency but monthly condors still viable. Returns compress to 1-2% monthly on margin (down from 2-4% pre-reforms). March 2020 (VIX 87), June 2024 (election VIX 25+) show tail risk.
**Expected:** Sharpe 0.8-1.5, monthly return 1-2% on margin, max DD 15-25% with wings.
**Kill condition:** -15% of allocated capital in a single month.
**Confidence:** HIGH for VRP existence, MEDIUM for post-reform returns.

---

## S7: Pairs / Statistical Arbitrage

**Thesis:** Cointegrated NSE stock pairs mean-revert at daily frequency. No co-location needed for 5-20 day holds.

**Specs:**
- Universe: Top 50 NSE futures stocks, screen for cointegration (Engle-Granger + Johansen)
- Entry: Spread deviates >2σ from rolling mean
- Exit: Spread returns to mean (0σ) or time stop at 20 days
- Instrument: Single stock futures (tighter spread than delivery, no STT on delivery)
- Hold: 5-20 days (mean reversion half-life for Indian pairs)
- Sizing: 5-10 pairs simultaneously, ₹50K per leg at ₹50L
- Sectors: Auto, Realty, Banking (Sen 2022 found highest returns)
- Rescreen: Monthly cointegration re-test; drop pairs that break cointegration

**Evidence:** Sen (2022, arXiv/IEEE) — Auto and Realty sectors highest returns. Sen et al. (2023) — 12 sectors, dual validation. Bhattacharya et al. — 25 of 231 bank pairs cointegrated. After-cost Sharpe 0.5-1.2.
**Expected:** Sharpe 0.5-1.0, max DD 10-25%, low correlation with directional strategies.
**Kill condition:** Sharpe < 0.3 over 6 months.
**Confidence:** MEDIUM — evidence exists but transaction costs often excluded from studies.

---

## Correlation Matrix (Expected)

|  | S1 ORB | S2 Overnight | S3 VWAP | S4 Momentum | S5 Expiry | S6 Vol Prem | S7 Pairs |
|---|---|---|---|---|---|---|---|
| S1 | 1.0 | 0.1 | -0.3 | 0.0 | 0.2 | -0.1 | 0.0 |
| S2 | | 1.0 | 0.0 | 0.1 | 0.0 | -0.2 | 0.0 |
| S3 | | | 1.0 | 0.0 | -0.1 | 0.1 | 0.0 |
| S4 | | | | 1.0 | 0.0 | 0.0 | -0.1 |
| S5 | | | | | 1.0 | 0.4 | 0.0 |
| S6 | | | | | | 1.0 | 0.0 |
| S7 | | | | | | | 1.0 |

S1/S3 anti-correlated (ORB vs mean reversion — regime switching). S5/S6 moderately correlated (both short vol). Rest near-zero. Portfolio diversification is genuine.

---

## Backtest Requirements (Non-Negotiable)

1. Event-driven simulation (not vectorized). Queue position and fill probability modeled.
2. Post-Apr-2025 cost rates. STT 0.15% options, 0.05% futures.
3. Spread modeled: ₹0.30/side options, ₹0.05/side futures. NOT spread=0.
4. Slippage: 1 tick additional per side.
5. Minimum 2 years of data for intraday strategies, 5+ years for daily+.
6. Walk-forward validation: train on 70%, test on 30%, rolling window.
7. Deflated Sharpe Ratio (DSR) applied to all results. Record every strategy tested, not just winners.
8. Kill threshold: after-cost Sharpe < 0.5 in backtest = don't paper trade.
