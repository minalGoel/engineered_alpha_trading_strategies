# Strategy Triage Methodology

## Summary
How to classify an existing strategy library for a new trading regime. Two triages were performed: the first was wrong (filtered for "options microstructure" → 6 survivors, 0 profitable), the second was correct (filtered for "spot direction prediction" → 291 survivors). The difference was a framing error about what the edge source actually is.

## The Wrong Triage: Options Microstructure

### Framing
"We're trading options. Therefore we need strategies that exploit option-specific pricing inefficiencies."

### Criteria
- IV-RV spread dislocation
- Put-call skew mean reversion
- Volatility surface mispricing
- Gamma hedging effects on expiry
- Options order flow imbalance

### Result
- 6 of 358 strategies survived
- All focused on option-specific phenomena (IV skew, vol surface, implied-realized spread)
- After correcting the strike-locking bug: 0 of 6 were net profitable
- The option-specific mispricing is ₹1-2 per trade. Costs are ₹3-4 per trade. No room.

### Why It Failed
The mispricing edge in option premiums at 5-second frequency is tiny — measured in fractions of a point. With the correct cost model (STT=0.15%, spread=0, ₹1L capital), breakeven is ~0.14-0.86 pts — but the option-chain mispricing signal itself is still unreliable at 5-second resolution and produced no consistent directional edge.

## The Correct Triage: Spot Direction Prediction

### Framing
"We're predicting which direction NIFTY/BANKNIFTY spot will move in the next 15-60 seconds. The option is a leveraged directional bet, not the signal source."

A 10-point NIFTY spot move → ~5 point ATM option move (delta 0.5) → ₹325 gross per lot (65 units × 5 pts). At ₹1L capital (~5 lots at ₹300 entry), costs ~₹65/lot (STT=0.15%, spread=0, brokerage diluted). Net ~₹260/lot. This is viable with ~53-55% directional accuracy.

### Criteria
**QUALIFY** — Any strategy that predicts price direction on a single instrument:
- Momentum (continuation, trend following, acceleration)
- Mean reversion (RSI, Bollinger, z-score, overextension)
- Breakout (range, volatility, volume, opening range)
- Volume-based (spike detection, relative volume, OBV)
- VIX-conditional direction
- Time-of-day patterns (opening drive, closing flow, lunch fade)
- Microstructure (order flow, tick momentum, trade arrival)
- VWAP as reference (reversion to VWAP, breakout from VWAP)
- Moving average strategies (crossovers, trend detection)
- ADX, MACD, ATR-based strategies
- NIFTY-BANKNIFTY spread (the one valid pair)

**DISCARD** — Structurally can't work on 2 indices:
- Cross-sectional (requires ranking across stock universe)
- Pair trading between individual equities
- Sector rotation (requires sector indices)
- Equity-specific data (delivery %, SLB, promoter holdings, earnings, PE ratio)
- Stock selection ("buy the 10 most oversold") — can't select from 2 items

### Result
- 291 of 358 qualified
- ~67 discarded (cross-sectional, pair trading, equity-specific data)
- Massive increase in strategy diversity: momentum, mean reversion, breakout, volume, time-of-day

## The Mechanism Quality Gate

Even after the correct triage, a second filter is needed: every strategy must have a plausible **microstructure mechanism** explaining WHY the predicted move happens.

### Good Mechanisms (Pass)
- "Institutional order flow arrives in bursts, creating 30-120 second persistent buying pressure before liquidity absorbs it"
- "Stop-loss cascades from retail algos hitting clustered stops at round-number levels create 5-8 point overshoots that revert in 20-40 seconds"
- "Opening gap triggers overnight order accumulation; the first 15-minute range represents the absorption of this flow"

### Bad Mechanisms (Fail)
- "RSI below 30 means oversold"
- "Moving averages crossed"
- "Price hit the lower Bollinger band"

These describe WHAT the indicator shows, not WHY the market moves. Without a mechanism, the strategy is fitting noise.

### The Template Problem
When batch-converting 291 strategies, the LLM wrote ~30 mechanism templates and copy-pasted:
- Zero mechanism failures out of 291 → the filter was fake
- Fix: force unique mechanisms by auditing for >50% word overlap between any two strategies

## Lookback Adaptation Rules

When converting a 1-minute strategy to 5-second bars:

### Default: Preserve Time
RSI(14) on 1-min = 14-minute lookback → RSI(168) on 5s bars. Same market information, different bar count.

### Exception: Preserve Bars
When the original thesis is about bar-count persistence: "14 consecutive up bars" → RSI(14) on 5s = 70-second lookback. The thesis is about persistence in bars, not minutes.

### Neither: Rethink Entirely
Some strategies need a shorter lookback for 5-second trading. A 14-minute RSI captures the "slow recovery." For 30-second option holds, you want the "fast oversold" — maybe RSI(36) = 3-minute lookback. Document the reasoning.

## Stop/Target Adaptation

Equity stops (0.3% of stock price) don't translate directly to option points.

### The Conversion
```
spot_stop_pct × underlying_price × delta ≈ option_stop_points
0.3% × 24400 × 0.5 = 36.6 option points
```
36 points is absurd for a 30-second hold. Don't mechanically convert.

### The Correct Approach
Think about the strategy type:
- **Momentum**: Tight stop (2-3 pts). If it doesn't continue immediately, exit.
- **Mean reversion**: Wider stop (4-6 pts). Reversions overshoot before bouncing.
- **Breakout**: Medium stop (3-5 pts). Below the breakout level.
- **Each strategy gets its own rationale** — no category defaults.

## Decisions Made
- Spot direction prediction is the correct framing, not options microstructure
- 291 qualified (vs 6) — massive improvement in strategy diversity
- Mechanism field mandatory with uniqueness enforcement
- Lookback default: preserve time (12x scaling), with exceptions
- Stops/targets: individual reasoning, not category assignment

## Pitfalls & Anti-patterns
- **Framing the triage by instrument instead of edge source**: "We trade options → need option strategies" is wrong. "We predict spot direction → options are the instrument" is right.
- **Template mechanisms**: 30 templates × 10 strategies each = 291 "unique" mechanisms that are all the same. Audit with word-overlap analysis.
- **Category-default stops**: "Scalp: 2/3, breakout: 4/7" applied uniformly. Useless. Each strategy needs individual reasoning.
- **Blind 12x lookback scaling**: Produces technically correct but strategically wrong lookback periods.
- **Not killing enough**: Sonnet kills ~10-15% of strategies. Opus kills ~20-30%. If kill rate is below 15%, the triage is too lenient.

## Corrections Log
- ~~First triage: options microstructure framing → 6 survivors~~ → Corrected to spot direction prediction → 291 survivors
- ~~All 6 vol/IV strategies had edge~~ → 0 of 6 profitable after corrected strike locking
- ~~291 strategies all passed mechanism filter~~ → Zero failures = fake filter. Mechanism uniqueness must be audited separately.
- ~~Stops assigned by category (scalp/breakout/reversion)~~ → Each strategy requires individual stop reasoning
