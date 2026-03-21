# Cost Model Decisions

## Summary
Two fundamentally different cost regimes: equity spot signals (costs excluded) and direct option trading (costs included). The transition from excluded to included costs revealed that most strategies have no edge after costs. This document captures the cost structures, the decision logic, and the breakeven analysis.

## When to Exclude Costs (Equity Phase)

The equity pipeline generated SPOT signals (BUY RELIANCE at ₹2487) which were then used to trigger option buys in a SEPARATE OMS. The spot signal itself has no execution cost — no STT, no brokerage, no spread. The option execution costs are handled by the OMS, not the signal backtester.

Therefore: equity signal backtesting excludes all costs. Sharpe, PnL, profit factor — all gross.

## When to Include Costs (Options Phase)

When trading options directly (BUY_CE NIFTY 24400 at ₹300), every trade incurs real costs:

### Execution Model (Fundamental)

Strategies trade **candle-to-candle**: enter at one candle's close price, exit at a subsequent candle's close price, holding for at most 120 seconds (24 bars at 5-second frequency). Entries and exits are modelled as **limit orders filling at candle close** — there is no bid-ask spread modelled. The P&L is purely the candle-to-candle premium movement.

This means: **spread cost = 0**. Any spread-based cost model is wrong for this regime.

### Upstox Cost Structure — Equity Options (Current)

| Component | Rate | Applied To | Scales With |
|-----------|------|-----------|-------------|
| Brokerage | ₹20 per ORDER (flat) | Per order, any size | Fixed per order |
| STT | **0.15%** on sell side | exit_premium × qty | qty (not dilutable) |
| Exchange charges (NSE) | 0.053% | total premium turnover (both sides) | turnover |
| SEBI turnover fee | 0.0001% | total premium turnover | turnover |
| Stamp duty | 0.003% on buy side | entry_premium × qty | qty |
| GST | 18% | brokerage + exchange + SEBI | derived |

**Note**: STT is 0.15% from 1 April 2025 (hiked from 0.1% Oct 2024, and before that 0.0625%). Use 0.15% for all backtesting.

### Key Insight: Brokerage is Per ORDER, Not Per Lot
- 1 lot (65 units) → brokerage = ₹20 per order
- 10 lots (650 units) → brokerage = ₹20 per order (SAME)
- Round trip: 2 orders = ₹40 fixed

Brokerage per unit drops with scale; STT does NOT:
- 1 lot: ₹40/65 = ₹0.62/unit brokerage | STT scales linearly
- 10 lots: ₹40/650 = ₹0.062/unit brokerage | STT 10× larger

### Sample Trade Calculation — ₹1L Capital, NIFTY

**Entry ₹200, exit ₹205** (5-point move, ₹1L capital):
```
lots     = floor(100000 / (200 × 65))     = 7
qty      = 7 × 65                          = 455 units
entry_tv = 200 × 455                       = ₹91,000 (buy side turnover)
exit_tv  = 205 × 455                       = ₹93,275 (sell side turnover)
total_tv = 91,000 + 93,275                = ₹1,84,275

Gross PnL:        (205 - 200) × 455        = ₹2,275.00
STT (sell side):  93,275 × 0.0015          =   ₹139.91
Exchange (NSE):   1,84,275 × 0.00053       =    ₹97.67
SEBI:             1,84,275 × 0.000001      =     ₹0.18
Stamp (buy side): 91,000 × 0.00003         =     ₹2.73
Brokerage:        20 × 2                   =    ₹40.00
GST 18% on (brokerage + exchange + SEBI):
  = 0.18 × (40 + 97.67 + 0.18)            =    ₹24.81
                                           ----------
Total costs:                               =   ₹305.30
Net PnL:          2,275 − 305.30           = ₹1,969.70
```

**Entry ₹300, exit ₹305** (5-point move, ₹1L capital):
```
lots     = floor(100000 / (300 × 65))     = 5
qty      = 325
Gross:   (305-300) × 325                  = ₹1,625.00
STT:     305 × 325 × 0.0015               =   ₹148.69
Exchange: (300+305) × 325 × 0.00053       =   ₹104.21
SEBI:     same × 0.000001                 =     ₹0.20
Stamp:   300 × 325 × 0.00003              =     ₹2.93
Brokerage:                                =    ₹40.00
GST 18%: 0.18 × (40 + 104.21 + 0.20)     =    ₹26.00
                                          ----------
Total costs:                              =   ₹322.03
Net PnL:  1,625 − 322                     = ₹1,302.97
```

## STT — The Dominant Cost at Scale

STT (0.15% of exit turnover) is the largest single cost component (~46% of total). Unlike brokerage, it does NOT dilute with more lots — it scales linearly with qty. At ₹1L capital:

| Entry Premium | Lots (NIFTY, lot=65) | Qty | STT | Total RT Costs | Breakeven (pts) |
|--------------|----------------------|-----|-----|----------------|-----------------|
| ₹50  | 30 | 1,950 | ₹146 | ~₹280 | ~0.14 pts |
| ₹100 | 15 | 975   | ₹146 | ~₹280 | ~0.29 pts |
| ₹200 | 7  | 455   | ₹137 | ~₹262 | ~0.58 pts |
| ₹300 | 5  | 325   | ₹147 | ~₹281 | ~0.86 pts |

*Breakeven = total_costs / qty (option points needed to cover all round-trip costs)*

The breakeven is a small fraction of a point — achievable within 1-3 bars of directional movement. The approach is viable with directional accuracy above ~53-55%.

## Breakeven Analysis

For a strategy to be net profitable after costs, it must move at least the breakeven points per trade (see table above). The breakeven numbers depend dynamically on the actual entry premium for each trade — the pipeline computes them per-trade, not from this table.

**What these breakevens mean**:
- NIFTY 5-second average move: ~0.5-2 spot points
- At delta 0.5: ~0.25-1 option point per 5-second bar
- At ₹200 premium: need 0.58 pts ≈ covered by 1 bar of directional movement
- This is **~7× easier to clear** than the old spread-included breakeven of ~4 pts

Directional accuracy needed: ~53-55% (achievable with good signal quality).

## Decisions Made (March 2026 overhaul)
- **Spread = 0**: Limit order execution at candle close prices. Entry = candle close, exit = subsequent candle close. No bid-ask spread.
- **Execution regime**: Candle-to-candle direction bets, hold ≤120 seconds (≤24 bars at 5s). NOT intra-candle spread capture.
- **STT = 0.15%**: Hiked from 0.1% (Oct 2024) and before that 0.0625%. Use 0.15% from 1 Apr 2025.
- **Capital per entry = ₹1,00,000**: Determines lot count dynamically per trade based on actual entry premium.
- **Lot count = floor(capital / (entry_premium × lot_size))**: Integer lots only; actual deployed ≤ ₹1L.
- Brokerage is flat ₹40 RT — diluted at scale, secondary vs STT.
- STT (0.15%) is the floor cost — percentage-based, cannot be reduced.
- Report BOTH gross and net metrics everywhere.
- Dynamic costs: `net_pnl_quick(entry_prem, exit_prem, lot_size, capital)` computes all costs per-trade using actual premiums. The breakeven table is guidance only.

## Pitfalls & Anti-patterns
- **Modelling spread as a cost**: Wrong for this regime. Entries and exits are at candle close prices. Spread = 0.
- **Strategies targeting bid-ask spread**: Fundamentally wrong. Strategies predict CANDLE DIRECTION; the option is the leveraged instrument.
- **Using gross Sharpe for optimization objective**: When costs are on, optimizer must use after-cost Sharpe.
- **Ignoring STT at scale**: STT (0.15% of sell turnover) doesn't dilute with more lots. It's the floor cost.
- **Assuming 1-lot trades**: With ₹1L capital, trades are 5-30 lots. Cost structure is dominated by STT, not flat brokerage.
- **Using old STT rate (0.0625% or 0.1%)**: Will massively understate costs. Use 0.15%.

## Corrections Log
- ~~Cost model charged brokerage per lot~~ → Brokerage is flat per ORDER (₹20 regardless of lots)
- ~~Spread assumed at ₹1.50/side~~ → ~~Revised to ₹1.00/side~~ → **Removed entirely (spread = 0)**. Trades execute at candle close prices, not on bid-ask spread.
- ~~Strategies modelled as exploiting bid-ask spread~~ → **Fundamentally wrong**. Strategies predict candle direction; options are the leveraged instrument.
- ~~NIFTY lot size = 75~~ → Corrected to **65**
- ~~BANKNIFTY lot size = 15~~ → Corrected to **30**
- ~~STT = 0.0625%~~ → ~~0.1% (Oct 2024)~~ → **0.15%** (from 1 Apr 2025)
- Added CAPITAL_PER_ENTRY = ₹1,00,000 for realistic lot sizing
- Breakeven revised from ~4 pts (with spread) to ~0.14-0.86 pts (spread=0) depending on premium level
