# ACTIVE_THREADS.md — Current Project State

## Last Updated: 2026-03-21

## What We Are
Pre-seed quant fund. ₹50L deployed capital, pledged scale to ₹10Cr by Year 2. Strong tech team (concurrency, infra). No co-location Year 1 — everything else in scope.

## Capital Trajectory
| Month | Capital | Gate |
|---|---|---|
| 0-3 | ₹50L | Build + validate all strategies live |
| 4 | ₹1 Cr | Pass 30% annualized run-rate |
| 4-6 | ₹1 Cr | Validate at scale |
| 7 | ₹5 Cr | Pass 30% gate at ₹1 Cr |
| 7-12 | ₹5 Cr | Validate at scale |
| 13 | ₹10 Cr | Year 2 begins |

**Year 1 target: 30% minimum.** Benchmark: Capitalmind 30.2% CAGR with momentum alone. We run 5-7 uncorrelated strategies with a tech team. 30% is the floor.

## The Portfolio

| # | Strategy | Instrument | Hold Time | Trades/Day | Capital % | Exp. Sharpe |
|---|---|---|---|---|---|---|
| 1 | 30-min ORB (sell opposite) | NIFTY monthly options | 45-180 min | 1-3 | 20% | 0.8-1.2 |
| 2 | Overnight gap capture | NIFTY futures | Overnight | 1 | 15% | 1.0-1.5 |
| 3 | VWAP mean reversion | NIFTY futures/options | 15-90 min | 2-5 | 15% | 0.6-1.0 |
| 4 | Cross-sectional momentum | NIFTY200 equities | Monthly rebal | ~10/mo | 20% | 0.8-1.2 |
| 5 | Expiry day structures | NIFTY weekly options | Expiry (Tue) | 2-5 | 10% | 0.5-1.0 |
| 6 | Vol premium (iron condors) | NIFTY monthly options | 3-7 days | 2-4/wk | 15% | 0.8-1.5 |
| 7 | Pairs / stat arb | Top NSE futures pairs | 5-20 days | 2-4 | 5% | 0.5-1.0 |

Portfolio Sharpe target: 1.5-2.0. ERC allocation, vol targeting 15% annualized.

## Infrastructure
| Component | Choice | Status |
|---|---|---|
| Data feed | Fyers TBT (50-level DOM) | Pending account |
| Primary execution | Dhan (25 OPS, 200-depth) | Pending account + static IP |
| Secondary execution | Upstox (50 OPS, HFT endpoint) | Existing |
| Backup | Angel One | Pending |
| Compute | AWS ap-south-1 | Ready |
| Multi-broker router | OpenAlgo or custom | Build Week 2 |

## 90-Day Sprint
- **Week 1-2:** Infra — accounts, static IPs, monitoring, data pull, tick recording
- **Week 2-4:** Backtest all 7 strategies, kill anything with after-cost Sharpe < 0.5
- **Week 4-6:** Paper trade survivors, measure signal degradation vs backtest
- **Week 6-12:** Live at ₹50L, scale up strategy by strategy, weekly risk reviews
- **Month 4:** Scale gate → ₹1 Cr if 30% run-rate achieved

## What's Dead
- 5-second microstructure on options
- Spread=0 model for live trading
- 291 codegen strategies
- Anything requiring co-location in Year 1
