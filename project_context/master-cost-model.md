# Master Cost Model — Single Source of Truth

## Last Updated: 2026-03-21 | Regime: Post-April 2025 STT rates

## Critical Corrections

| Source | Error | Correct |
|---|---|---|
| Gemini Feasibility Study | NIFTY lot = 75 | **65** |
| Gemini study | Futures RT = ₹1,142 | **₹941** (recalc with lot 65) |
| Gemini study | Breakeven = 15.2 pts | **14.5 pts** |

## Contract Specs (March 2026)

| Index | Lot Size | Weekly Expiry | Monthly Expiry |
|---|---|---|---|
| NIFTY 50 | **65** | Tuesday (NSE only) | Last Thursday |
| BANKNIFTY | **30** | **Discontinued Nov 2024** | Last Tuesday |
| FINNIFTY | 65 | Discontinued | Monthly |

## Cost Rates (April 2025+)

| Component | Equity Delivery | Equity Intraday | Index Futures | Index Options |
|---|---|---|---|---|
| STT (buy) | 0.1% | Nil | Nil | Nil |
| STT (sell) | 0.1% | 0.025% | 0.05% of value | **0.15%** of premium |
| Brokerage | ₹20/order flat | same | same | same |
| Exchange (NSE) | 0.00173% | same | same | 0.053% |
| SEBI fee | 0.0001% | same | same | same |
| Stamp (buy) | 0.015% | 0.003% | 0.003% | 0.003% |
| GST | 18% on (brok+exch+SEBI) | same | same | same |

## Round-Trip Cost by Instrument (1 Lot)

| Instrument | Setup | RT Cost | Breakeven Move | Min Viable Hold |
|---|---|---|---|---|
| NIFTY futures (1 lot) | NIFTY@24000, 65 units | ₹941 | 14.5 pts | 15-30 min |
| NIFTY options buy (1 lot, ₹200 prem) | 65 units | ₹84 | 1.3 opt pts | 5-15 min |
| NIFTY options buy (7 lots, ₹200 prem, ₹1L cap) | 455 units | ₹305 | 0.67 opt pts | 5-15 min |
| NIFTY option sell (strangle, 1+1 lot) | 130 units | ~₹127 | 1.2% of collected prem | 1-7 days |
| Equity delivery (₹2L trade) | varies | ₹486 | 24.3 bps | Monthly+ |

## Instrument Decision Matrix

| Hold Period | Best Instrument | Why |
|---|---|---|
| <5 min | Not viable (no colo) | Latency eats >1% of hold |
| 5-30 min | NIFTY monthly options | Lower STT per delta; gamma convexity |
| 30 min - 4 hr | NIFTY options (sell opposite for ORB) | Theta + directional |
| Overnight | NIFTY futures | No theta, no vega, tighter spread |
| Multi-day | NIFTY futures | Linear exposure |
| Monthly rebal | Cash equities delivery | LTCG 12.5% if >1yr |

## Live Execution Cost Adjustments (NOT in backtest)

Add these to all live projections on top of backtest costs:
- Options spread: ₹0.30/side (ATM liquid hours), ₹0.50/side (illiquid)
- Futures spread: ₹0.05/side (liquid), ₹0.15/side (illiquid)
- Slippage at ₹5L order: ~0.5-1 tick additional
- Latency adverse selection: budget 1-2 bps for intraday strategies

## Tax Rates (Applies to PnL, Not Cost Model)

| Income Type | Rate | Applies To |
|---|---|---|
| F&O business income | Slab (up to 30% + 4% cess) | All F&O PnL |
| STCG equity (<1 yr) | 20% | Momentum rebalance sells |
| LTCG equity (>1 yr) | 12.5% (above ₹1.25L) | Long-hold equity |
| F&O turnover for audit | Absolute sum of profits + losses | Section 44AB threshold ₹10Cr |
