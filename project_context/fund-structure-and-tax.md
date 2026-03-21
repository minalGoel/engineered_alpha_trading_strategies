# Fund Structure, Tax & Scaling Path

## Last Updated: 2026-03-21

---

## Current Phase: Individual Prop Trading (₹50L)

### Tax Treatment

F&O income = **non-speculative business income** under Section 43(5) of Income Tax Act. Taxed at individual slab rates (up to 30% + 4% cess + applicable surcharge).

| Income Head | Classification | Tax Rate | Set-Off Rules |
|---|---|---|---|
| F&O profits/losses | Non-speculative business | Slab (up to ~31.2%) | Against any income except salary |
| Equity intraday | Speculative business | Slab | Only against speculative income |
| Equity delivery <1yr | Short-term capital gain | 20% flat | Against STCG only |
| Equity delivery >1yr | Long-term capital gain | 12.5% (above ₹1.25L) | Against LTCG only |

**Loss carry-forward:** F&O losses carry forward 8 assessment years against non-speculative business income. Must file ITR on time (ITR-3).

### F&O Turnover Calculation (Critical for Audit)

**Turnover = absolute sum of profits and losses per trade** (NOT total traded value).

Example at ₹50L with 7 strategies:
- 20 trades/day × 250 days = 5,000 trades/year
- Average absolute P&L per trade: ₹500-2,000
- Estimated turnover: ₹25L - ₹1 Cr
- **Section 44AB audit threshold: ₹10 Cr** (for 95%+ digital transactions)
- At ₹50L capital, you're almost certainly below audit threshold

**Presumptive taxation (Section 44AD):** If turnover < ₹3 Cr, declare minimum 6% of turnover as profit → no audit required. At ₹1 Cr turnover, minimum declared profit = ₹6L. If actual profit exceeds this, declare actual. If actual profit is below 6%, audit becomes mandatory.

### Business Expense Deductions (Section 37(1))

All of these are deductible as "expenses wholly and exclusively for business":

| Expense | Deductible? | Section |
|---|---|---|
| AWS / cloud compute | Yes | 37(1) |
| Data feed subscriptions | Yes | 37(1) |
| Broker API fees | Yes | 37(1) |
| Internet (proportionate) | Yes | 37(1) |
| Hardware (40% WDV depreciation) | Yes | 32(1) |
| Trading software licenses | Yes | 37(1) |
| Professional fees (CA, legal) | Yes | 37(1) |
| Home office rent (proportionate) | Yes | 37(1) |
| STT paid | Yes (deductible as business expense since 2008) | 36(1)(xv) |
| Team salaries/contractor payments | Yes | 37(1) / 192 TDS |

**File ITR-3** with separate P&L statements for F&O business and capital gains.

### Advance Tax

Quarterly advance tax required for business income:
- 15% by Jun 15, 45% by Sep 15, 75% by Dec 15, 100% by Mar 15
- Interest under Section 234B/C for shortfall
- At ₹50L capital targeting 30% = ₹15L profit → ~₹4.5L tax liability → budget ₹1.1L/quarter

### GST

**Not required for prop trading.** Securities excluded from "goods" and "services" under CGST Act 2017. GST applies only when providing advisory/management services to others.

---

## Scaling Path

### Month 4: ₹50L → ₹1 Cr

- Continue as **individual prop trader**
- Additional ₹50L is own/pledged capital — no regulatory implications
- Slab rate tax: at ₹30L+ profit, you're at 30% bracket
- No structural change needed

### Month 7: ₹1 Cr → ₹5 Cr

**Decision point: Is this all own capital or outside money?**

**If own/family capital:** Continue individual or form **LLP**.
- LLP: 2 partners minimum, ₹10-20K setup, 2-4 weeks
- LLP tax: 30% flat on profits (~34.94% effective with surcharge+cess)
- Partner profit share exempt under Section 10(2A)
- Liability protection — personal assets shielded
- LLP can trade F&O without SEBI registration (prop trading)

**If outside investor capital:** Need SEBI-registered structure.
- PMS or AIF required to manage others' money
- Start PMS registration process NOW (takes 6-12 months)

### Year 2: ₹5 Cr → ₹10 Cr

| Structure | When | Requirements | Tax | Setup Cost | Timeline |
|---|---|---|---|---|---|
| **Individual** | ₹50L-₹1 Cr | None | Slab (up to 30%) | ₹0 | Immediate |
| **LLP** | ₹1-5 Cr own capital | 2 partners | 30% flat | ₹10-20K | 2-4 weeks |
| **SEBI PMS** | ₹2-5 Cr+ outside capital | ₹5 Cr net worth (entity); PO with 5yr exp + NISM XXI-A/B; ₹10L SEBI fee | Pass-through to clients | ₹15-25L | 6-12 months |
| **AIF Cat III** | ₹20 Cr+ corpus | ₹5 Cr sponsor net worth; ₹1 Cr min per investor; 5% sponsor commitment | Fund-level 30-42.74% (SEVERE) | ₹20-35L | 6-18 months |
| **Smallcase** | Equity-only strategies | SEBI RA registration; NISM cert | Pass-through (client executes) | ₹1-3L | 3-6 months |

### Recommended Progression

1. **Now → Month 6:** Individual. Slab rates are lower than LLP's 30% for income under ~₹15L profit.
2. **Month 6 (if scaling with partners):** Form LLP. Liability protection, can add partners.
3. **Month 6 (start in parallel):** Begin SEBI PMS registration process. It takes 6-12 months — start before you need it.
4. **Month 12-18:** PMS live. Accept outside capital at ₹50L minimum per client. Pass-through taxation is a massive advantage over AIF.
5. **Avoid AIF Cat III** unless you need leverage (up to 2x NAV per SEBI circular) or pooled fund structure is essential. The fund-level taxation (30-42.74%) is devastating compared to PMS pass-through.

### PMS Registration Requirements (Start Preparing Now)

| Requirement | Detail |
|---|---|
| Entity | Company or LLP (individual cannot register PMS) |
| Net worth | ₹5 Cr (entity level) |
| Principal Officer | 5 years securities market experience; NISM Series XXI-A (PO) and XXI-B (Distributors) |
| SEBI fee | ₹10L (application) |
| Infrastructure | Office, compliance officer, risk management |
| Min per client | ₹50L |
| Track record | Not mandatory but practically essential for client acquisition |
| Timeline | 6-12 months from application to approval |

**Key insight:** Build your audited track record via ITR-3 from Day 1. When you apply for PMS, the first question investors and SEBI ask is "show me your returns." 12 months of audited 30%+ returns on your own ₹50L-5Cr is the strongest possible application.

### Smallcase (Parallel Revenue Stream)

Smallcase cannot do F&O — equity/ETF baskets only. But your **cross-sectional momentum strategy (S4)** is pure equity delivery. Once validated:
- Register as SEBI Research Analyst (NISM cert + ₹1-3L setup)
- Publish momentum strategy as Smallcase
- Subscription revenue: ₹7K-20K/year per subscriber
- Builds audience and AUM pipeline for eventual PMS
- Zero additional capital requirement

---

## Risk: Turnover Exceeding ₹10 Cr at Scale

At ₹5 Cr capital with 7 strategies:
- Possible 50-100 trades/day × 250 days = 12,500-25,000 trades/year
- If average absolute P&L per trade rises to ₹5,000-10,000
- Turnover: ₹6.25 Cr - ₹25 Cr → **may cross ₹10 Cr audit threshold**
- At that point: mandatory tax audit under Section 44AB
- Budget ₹50K-1L/year for CA audit fees
- This is manageable — just plan for it

---

## Regulatory Risk Register (Updated)

| # | Risk | Prob | Impact | Mitigation |
|---|---|---|---|---|
| 1 | OPS exceeds 10 at ₹5 Cr scale | Medium | Medium — need algo registration | Start registration at ₹1 Cr scale, don't wait |
| 2 | SEBI tightens algo rules further | Low | Medium | Classify algos as White Box; maintain documentation |
| 3 | Broker not compliant by Apr 1 | Low | High | Multi-broker setup; Dhan is most compliant |
| 4 | PMS registration delayed | Medium | Medium — can't take outside capital | Start process at Month 6 |
| 5 | Tax audit triggered early | Low | Low — just costs CA fees | Maintain clean books from Day 1 |
| 6 | Algo malfunction causes market impact | Very Low at ₹50L | High if at ₹5 Cr | Kill switches, position limits, max loss per day |
| 7 | Capital drawdown exceeds pledged schedule | Medium | High — scaling path breaks | Enforce kill conditions per strategy; 15% portfolio DD → half all positions |
