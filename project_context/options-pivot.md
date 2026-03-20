# Options Pivot — From Equity Spot to 5-Second Index Options

## Summary
The project pivoted from 1-minute OHLCV bars on 206 NSE equity stocks (using spot signals to trigger option trades externally) to 5-second candle data on NIFTY50/BANKNIFTY indices, trading their options directly. This document captures what changed, what survived, what was thrown out, and the critical reframing that happened mid-pivot.

## What Changed

| Dimension | Before (Equity) | After (Options) |
|-----------|-----------------|-----------------|
| Asset class | 206 individual stocks | 2 indices (NIFTY50, BANKNIFTY) |
| Timeframe | 1-minute bars | 5-second bars (12x density) |
| Trade execution | Spot signal → option bought externally | Buy CE/PE directly |
| PnL source | Spot price movement | Option premium change |
| Cost model | Excluded (irrelevant for spot signals) | Included (spread + STT + brokerage) |
| Hold time | 2-60 minutes | 5-120 seconds |
| Data | OHLCV per stock | Spot + full option chain + VIX |
| Lot sizes | N/A | NIFTY=75, BANKNIFTY=15 |
| Expiry handling | None | Track per position, force close |
| Greeks | None | Delta affects expected option move |

## What Survived from Equity Pipeline
- **Numba state machine** pattern (adapted for CE/PE states and premium-based PnL)
- **Per-strategy folders** with strategy.json, optimization.json, verdict.json, report.md
- **Optuna optimization** with Chan's principles (Sharpe objective, min trade count, default-first)
- **Walk-forward CV** (adapted to leave-one-day-out for thin data)
- **Autonomous batch processing** with Claude Code CLI
- **Crash resilience** with incremental saves and checkpoint files

## What Was Thrown Out
- All 358 equity strategy implementations (initially)
- The spot-signal-to-option-proxy model
- Equity-specific indicators (delivery %, SLB, promoter holdings)
- Stock filtering logic (irrelevant with 2 indices)
- The cost model exclusion

## The Critical Reframing

### Wrong Frame: "Options Microstructure"
The initial pivot asked: "Which strategies exploit options-specific pricing inefficiencies at 5-second frequency?"

This produced 6 survivors: IV-RV spread, put-call skew, vol surface, vol pattern, IV dislocation, and expiry day pattern. All focused on option-specific mispricing (IV too high, skew too steep, etc.).

**Result**: 0 of 6 were net profitable after costs. The option-specific mispricing is ₹1-2 per trade. Costs are ₹3-4 per trade. The math doesn't work.

### Correct Frame: "Spot Direction Prediction"
The actual goal: **predict which direction NIFTY/BANKNIFTY spot will move in the next 15-60 seconds**, then trade ATM options as leveraged directional bets.

A 10-point NIFTY move → ~5 point ATM CE/PE move (delta 0.5) → ₹325 gross per lot. Costs ~₹180. Net ~₹145. This is viable with 55-58% directional accuracy.

Under this frame, any strategy that predicts price direction on a single instrument qualifies: momentum, mean reversion, RSI, Bollinger, breakout, volume spike, VWAP, etc. The 352 "discarded" equity strategies came back into play.

### What This Means for Strategy Design
- Indicators should be computed on SPOT data, not option data
- The option is the INSTRUMENT, not the SIGNAL SOURCE
- `buy_ce` = "I predict spot goes up." `buy_pe` = "I predict spot goes down."
- Stop/target in option premium points (because that's your actual P&L)
- Delta ≥ 0.35 filter (far OTM options don't track spot reliably)

## The 12-Day Data Limitation

The 5-second dataset spans only 12 trading days (2026-03-04 to 2026-03-19). This is:
- Enough to build and debug the pipeline
- Enough to identify which strategies produce signals at all
- NOT enough for walk-forward validation (need 50+ trades per strategy)
- NOT enough to cover multiple VIX regimes, expiry cycles, or macro events

**Minimum viable history**: 6 months (~125 trading days) for meaningful backtesting. 12 months for regime coverage.

## Decisions Made
- Direct option trading (buy CE/PE) — no option writing/selling
- Cost model is ON for all option trades
- Leave-one-day-out CV for 12-day dataset (acknowledged as insufficient)
- Strategy must clear ~₹4 option points per trade after costs to be viable
- The mechanism field is mandatory — forces structural thinking about WHY
- 30-50 strategies target for new regime (quality over quantity)

## Corrections Log
- ~~First triage looked for "options microstructure" strategies → 6 survivors~~ → Correct triage looks for "spot direction prediction" → 291 survivors
- ~~Cost model excluded (as in equity)~~ → Cost model mandatory (direct option trading, costs matter)
- ~~vol_surface_signal had Sharpe +7.17~~ → Entirely phantom from the strike-switching bug. After fix: Sharpe deeply negative.
- ~~All 6 vol/IV strategies had edge~~ → 0 of 6 are net profitable after corrected strike locking and costs
- ~~NIFTY lot size = 75~~ → Corrected to 65 in recent audit (lot sizes change periodically — verify current values)
- ~~BANKNIFTY lot size = 15~~ → Corrected to 30 in recent audit
