# Cost Model Decisions

## Summary
Two fundamentally different cost regimes: equity spot signals (costs excluded) and direct option trading (costs included). The transition from excluded to included costs revealed that most strategies have no edge after costs. This document captures the cost structures, the decision logic, and the breakeven analysis.

## When to Exclude Costs (Equity Phase)

The equity pipeline generated SPOT signals (BUY RELIANCE at ₹2487) which were then used to trigger option buys in a SEPARATE OMS. The spot signal itself has no execution cost — no STT, no brokerage, no spread. The option execution costs are handled by the OMS, not the signal backtester.

Therefore: equity signal backtesting excludes all costs. Sharpe, PnL, profit factor — all gross.

## When to Include Costs (Options Phase)

When trading options directly (BUY_CE NIFTY 24400 at ₹300), every trade incurs real costs:

### Upstox Cost Structure (Current Broker)

| Component | Rate | Scales With |
|-----------|------|-------------|
| Brokerage | ₹20 per ORDER (flat) | Fixed per order, regardless of lots |
| STT | 0.0625% of sell-side premium | premium × quantity |
| Exchange charges | ~₹3-5 per lot per side | lots |
| SEBI turnover fee | 0.0001% | turnover |
| Stamp duty | varies by state | turnover |
| GST | 18% on (brokerage + exchange charges) | derived |

### Key Insight: Brokerage is Per ORDER, Not Per Lot
- 1 lot (75 units) → brokerage = ₹20 per order
- 10 lots (750 units) → brokerage = ₹20 per order (SAME)
- Round trip: 2 orders = ₹40 fixed

This means brokerage per unit drops with scale:
- 1 lot: ₹40/75 = ₹0.53/unit
- 10 lots: ₹40/750 = ₹0.053/unit

### Sample Trade Calculation

NIFTY ATM CE: entry ₹300, exit ₹303, 1 lot (75 units):
```
Gross PnL:      (303 - 300) × 75           = ₹225
Spread cost:    1.0 × 2 sides × 75         = ₹150
STT:            303 × 0.000625 × 75        = ₹14.20
Brokerage:      20 × 2 orders              = ₹40
Exchange:       5 × 2 sides × 1 lot        = ₹10
Net PnL:        225 - 150 - 14.20 - 40 - 10 = ₹10.80
```

Same trade at 7 lots:
```
Gross PnL:      (303 - 300) × 75 × 7       = ₹1,575
Spread cost:    1.0 × 2 × 75 × 7           = ₹1,050
STT:            303 × 0.000625 × 75 × 7    = ₹99.37
Brokerage:      20 × 2 (SAME — per order)  = ₹40
Exchange:       5 × 2 × 7                   = ₹70
Net PnL:        1,575 - 1,050 - 99.37 - 40 - 70 = ₹315.63
```

Net per lot improves from ₹10.80 (1 lot) to ₹45.09 (7 lots) — brokerage dilution.

## Spread Cost — The Dominant Factor

Spread accounts for ~79% of total costs in the 1-lot case. The ₹1.00/side assumption is an estimate — no bid/ask columns exist in the data. Validation:
- High-low proxy (intra-bar range) for ATM NIFTY CE: median 1.40 pts, mean 1.66 pts
- Roughly consistent with ₹1.0-1.5 per side assumption
- This is NOT the true bid-ask spread — it's a proxy

### Spread Sensitivity (Critical)
For the best gross-positive strategy tested:
- At spread ₹0.50/side: net positive, Sharpe > 0
- At spread ₹1.00/side: marginal
- At spread ₹1.50/side: net negative
- At spread ₹2.50/side: deeply negative

The viability of the entire approach depends on execution quality.

## Breakeven Analysis

For a strategy to be net profitable after costs, it must clear a minimum number of option points per trade.

At 1 lot, ₹1.00/side spread, Upstox:
```
Minimum gross PnL per lot = spread(150) + STT(~14) + brokerage(40) + exchange(10) ≈ ₹214
214 / 75 units = 2.85 points minimum
```

Add a margin of safety → **minimum ~4 points per trade** to be reliably profitable.

At 7 lots: minimum drops to ~₹192/lot = 2.56 points (brokerage diluted).

### What 4 Points Means
- NIFTY 30-second average move: ~5-8 spot points
- At delta 0.5: ~2.5-4 option points per 30-second move
- So you need to capture nearly the FULL 30-second directional move to break even
- Only ~10.9% of consecutive up-move runs last 3+ bars (the minimum for breakeven)

This is tight. It means only strategies with high directional accuracy AND good timing are viable.

## Decisions Made (March 2026 overhaul)
- **Spread = 0**: Limit order execution at candle close prices eliminates bid-ask spread entirely
- **STT corrected to 0.15%**: Was 0.025% in code, hiked to 0.1% from Oct 2024, then to 0.15% from 1 Apr 2025
- **Capital per entry = ₹1,00,000**: Determines lot count. At ₹200 premium, NIFTY gets 7 lots (455 units)
- Brokerage is flat per order (₹40 RT) — diluted across lots at scale
- STT (0.15%) is the dominant cost at scale — percentage-based, not dilutable
- Breakeven at ₹200 premium: ~0.58 pts (NIFTY, 7 lots) / ~0.57 pts (BANKNIFTY, 16 lots)
- Report BOTH gross and net metrics everywhere

## Pitfalls & Anti-patterns
- **Using gross Sharpe for optimization objective**: When costs are on, the optimizer must use after-cost Sharpe.
- **Ignoring STT at scale**: STT (0.15% of sell turnover) doesn't dilute with more lots — it's the floor cost. At 7 lots NIFTY (₹200 premium), STT alone is ~₹139/trade.
- **Assuming 1-lot trades**: With ₹1L capital, most trades are 5-30 lots. Cost structure is dominated by percentage fees, not flat brokerage.

## Corrections Log
- ~~Cost model charged brokerage per lot~~ → Brokerage is flat per ORDER (₹20 regardless of lots)
- ~~Spread assumed at ₹1.50/side~~ → ~~Revised to ₹1.00/side~~ → **Removed entirely** (limit orders at candle close)
- ~~NIFTY lot size = 75~~ → Corrected to 65
- ~~BANKNIFTY lot size = 15~~ → Corrected to 30
- ~~STT = 0.025%~~ → ~~0.1% (Oct 2024)~~ → **0.15%** (from 1 Apr 2025)
- Added CAPITAL_PER_ENTRY = ₹1,00,000 for realistic lot sizing
