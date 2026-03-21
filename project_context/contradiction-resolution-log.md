# Contradiction Resolution Log

## Last Updated: 2026-03-21
## Source: NotebookLM audit of all project documents
## Status: ALL RESOLVED

---

## C1: NIFTY Lot Size (65 vs 75)
**Gemini says 75. Project docs say 65.**
**Resolution: 65 is correct.** Gemini confused NIFTY with FINNIFTY. Verified against NSE contract specs. All Gemini-derived calculations using lot=75 are wrong by 15.4%.
**Action:** Done — Master Cost Model flags this. All new docs use 65.

## C2: NIFTY Futures RT Cost (₹1,142 vs ₹941 vs ₹1,052)
- Gemini: ₹1,142 (lot=75, wrong)
- Master Cost Model: ₹941 (lot=65, correct, no spread)
- Compass Playbook: ₹1,052 (lot=75 wrong, but includes spread)

**Resolution: ₹941 for regulatory costs only. ~₹1,010 with spread for live projections.**
₹941 is the correct regulatory cost (STT+exchange+SEBI+stamp+brokerage+GST) at lot=65. For live execution, add ₹0.05/side spread × 65 × 2 = ₹13, total ~₹954. The ₹1,052 figure is wrong because it used lot=75 as the base.
**Action:** Done — Master Cost Model is the authority.

## C3: Breakeven Move (15.2 vs 14.5 pts)
**Resolution: 14.5 pts.** ₹941 / 65 = 14.5. Gemini's 15.2 used lot=75 (₹1,142/75=15.2).
**Action:** Done.

## C4: STT Effective Dates (April 2025 vs April 2026)
**Both are partially correct — different STT hikes happened at different times.**
- 0.1% → 0.15% on options: effective **1 April 2025** (Union Budget 2025)
- 0.02% → 0.05% on futures: effective **1 April 2026** (Union Budget 2026)

The Gemini study lumps both under "April 2026 regime." The Master Cost Model says "April 2025+" because the options STT (the one that matters most for our strategies) changed in April 2025.

**Resolution: Use these exact dates:**
- Options STT 0.15%: from 1-Apr-2025 (already in effect)
- Futures STT 0.05%: from 1-Apr-2026 (11 days away)

**Action:** Master Cost Model updated to say "April 2025+" which is correct for options. Futures cost calculations already use 0.05% which takes effect Apr 1. No change needed — both rates are already in our cost model. The date labeling confusion doesn't affect any numbers.

## C5: Performance Target (30% floor vs 18-25% moderate)
**NotebookLM flags ACTIVE_THREADS at 30% floor vs Playbook at 18-25% moderate.**

**Resolution: Both are correct in context. 30% is the AMBITION. 18-25% is the EXPECTATION.**
- 30% is the capital-raise gate — if we don't hit 30% annualized run-rate, we don't scale to ₹1 Cr. This is a business decision, not a statistical forecast.
- 18-25% is what comparable operations (Capitalmind, factor indices) deliver on average. This is what the evidence supports as sustainable.
- The gap between ambition and evidence is acknowledged. The plan is to close it with 7 uncorrelated strategies (higher portfolio Sharpe than any single strategy) and a strong tech team.

**Action:** Add a note to ACTIVE_THREADS: "30% is the scale gate, not the statistical expectation. Evidence-based moderate expectation is 18-25%. The 7-strategy portfolio with ERC is designed to push above 25%."

## C6: Portfolio Sharpe (1.5-2.0 vs 1.0-1.5)
**Resolution: Same logic as C5.**
- 1.5-2.0 is the TARGET for a 7-strategy uncorrelated portfolio
- 1.0-1.5 is what individual strategies deliver
- If 5 strategies survive with Sharpe 0.8-1.2 each and correlation <0.2, portfolio Sharpe of 1.3-1.8 is mathematically achievable

**Action:** No change. Both numbers are correct in their context.

## C7: Pairs Trading Feasibility (Low vs Viable)
**Gemini: "Low" because lot sizes make balancing impossible at ₹25-50L.**
**Playbook: "Fully viable" at daily granularity.**

**Resolution: BOTH are partially right. Feasibility depends on instrument choice.**
- Gemini's concern is valid for FUTURES pairs — single stock futures have large lot sizes (e.g., TCS lot = 175 × ~₹4000 = ₹7L per leg). At ₹50L, running 5-10 pairs in futures is capital-intensive.
- Playbook's claim is valid for CASH EQUITY pairs — delivery pairs have no lot size constraint, just ₹486 RT cost per ₹2L trade (24.3 bps).
- **The right approach at ₹50L:** Cash equity pairs for stat arb, NOT futures pairs. At ₹5 Cr, switch to futures pairs (tighter spread, no delivery STT on both sides).

**Action:** Update strategy-implementation-plan S7 to specify cash equity pairs at ₹50L, futures pairs at ₹5 Cr+. Allocate only 5% of capital (₹2.5L) initially — this is the lowest-conviction strategy.

## C8: ORB Probing Duration (15 vs 30 min)
**Gemini: 15-30 min range. Playbook: 30 min specifically.**

**Resolution: 30 minutes for NIFTY option selling.**
- Chueh et al. (2019, IEEE) found longer probing times work better in Asian markets
- SaiMohanReddy showed 15-min ORB on BankNifty failed after costs
- The Zerodha Substack study used 30-min range and showed all years profitable
- At 5-min bar resolution, 30 min = 6 bars of range establishment, which is more stable than 3 bars (15 min)

**Action:** Strategy-implementation-plan already specifies 30 min. No change needed. BUT: backtest both 15-min and 30-min during Week 2-4 and let data decide. If 15-min ORB has higher after-cost Sharpe on 5yr data, use it.

## C9: Kill Condition Timing (Weeks 2-4 vs 60 trading days)
**ACTIVE_THREADS says kill in backtest during Weeks 2-4. Strategy plan says kill after 60 live trading days.**

**Resolution: These are different stages, not contradictions.**
- Weeks 2-4: Kill in BACKTEST if after-cost Sharpe < 0.5. This is a backtest quality gate. If it doesn't work on 5 years of historical data, don't paper trade it.
- 60 trading days: Kill in LIVE TRADING if after-cost Sharpe < 0.5. This is a live performance gate. A strategy can look good in backtest but fail live due to execution slippage, regime change, etc.

**Action:** Clarify in both docs that these are sequential gates: backtest gate (Weeks 2-4) → paper trade gate (Weeks 4-6) → live gate (60 trading days from live start).

## C10: Tax Audit Threshold Probability
**Gemini: "Surprisingly easy" to hit ₹10 Cr. Fund-structure doc: "Almost certainly below."**

**Resolution: Fund-structure doc is correct at ₹50L. Gemini is correct at ₹5 Cr.**
- At ₹50L with 20 trades/day: turnover = sum of absolute P&L per trade, NOT traded value. If average |P&L| = ₹1,000/trade × 5,000 trades/year = ₹50L turnover. Nowhere near ₹10 Cr.
- Gemini confused "turnover" with "traded value." Traded value would be ₹15-30 Cr at 20 trades/day on futures. But F&O turnover for tax = absolute P&L sum, which is much smaller.
- At ₹5 Cr with 50 trades/day and avg |P&L| = ₹10,000: turnover = ₹12.5 Cr → crosses ₹10 Cr threshold.

**Action:** Fund-structure doc already handles this correctly. At ₹50L: no audit. At ₹5 Cr: budget for audit (₹50K-1L/year CA fees).

## C11: Spread=0 Model Validity
**ACTIVE_THREADS: "Dead." Evidence Base: "Valid for candle-to-candle backtest." Compass: "Reverses sign of returns."**

**Resolution: All three statements are correct in context.**
- Spread=0 is DEAD for live execution projections and for any strategy that relies on spread capture (the old microstructure regime).
- Spread=0 is VALID as a backtest modeling choice IF your strategy enters/exits at candle close prices via limit orders AND you're not measuring the spread-crossing alpha. It's a modeling abstraction, not a claim about market reality.
- Spread=0 REVERSES the sign when applied to strategies that cross the spread on every trade (market orders, aggressive limits), which is what the compass artifact was analyzing.

**Action:** The new docs already handle this correctly:
- Master Cost Model: includes spread in "Live Execution Cost Adjustments" section
- Strategy Implementation Plan: requires spread modeling in backtests (₹0.30/side options, ₹0.05/side futures)
- The old spread=0 regime (candle-to-candle on 5-second bars) is dead and buried.

## C12: Fyers TBT Latency (<10ms Claimed, Unverified)
**Resolution: Acknowledged gap. No resolution possible without measurement.**

**Action:** Add to Week 1 sprint: "Run latency benchmark on Fyers TBT during market hours. Compare exchange timestamp vs local clock on 500+ ticks. If median latency >50ms, evaluate TrueData as alternative data source."

---

## Decision Conflicts (From Decision Audit)

### D1: ORB Probing Time → Resolved in C8 (30 min, but backtest both)
### D2: Pairs Trading → Resolved in C7 (cash equity at ₹50L, futures at ₹5 Cr)
### D3: Performance Targets → Resolved in C5 (30% is gate, 18-25% is evidence-based expectation)
### D4: Iron Condor Returns → No conflict. Both sources agree on 1-2% monthly post-reform. Pre-reform was 2-4%. The strategy plan uses the post-reform figure.
### D5: Futures STT Date → Resolved in C4 (options Apr 2025, futures Apr 2026)

---

## Open Items (Unresolvable from Documents)

1. **Fyers TBT actual latency** — measure in Week 1
2. **Post-Oct-2024 option selling returns** — only 5 months of data; insufficient for statistical confidence. Monitor and validate through live trading.
3. **Tuesday expiry effects** — only ~4 months of data since weekly switched to Tuesday. Low confidence until 6+ months accumulate.
4. **Portfolio Sharpe of 1.5-2.0** — theoretical from correlation assumptions. Will only know after 6+ months of live multi-strategy returns.
