# Signal Language Standard

## Summary
Two signal vocabularies: one for equity spot trading and one for index options. Each defines exact state machine transitions, valid state changes, and the output schema for downstream execution systems.

## Equity Signals

```
BUY    — Open a new long position
SELL   — Close an existing long position
SHORT  — Open a new short position
COVER  — Close an existing short position
FLAT   — No position, no action (default state)
```

**Why not BUYBACK?** BUYBACK is ambiguous (corporate action? closing a short?). COVER is the universal term for closing a short — every broker API from Zerodha to Interactive Brokers uses it.

### State Machine Rules (Equity)
- BUY only valid from FLAT → enters LONG state
- SELL only valid from LONG → returns to FLAT
- SHORT only valid from FLAT → enters SHORT state
- COVER only valid from SHORT → returns to FLAT
- One position at a time per stock per strategy. Always.
- EOD flatten: force SELL or COVER by 15:20 IST. No overnight.

### Equity Signal Schema
```json
{
    "timestamp": "2025-01-15T10:32:00+05:30",
    "symbol": "RELIANCE",
    "action": "BUY",
    "price": 2487.50,
    "stop_loss": 2480.05,
    "target": 2500.00,
    "metadata": {
        "strategy": "vwap_mean_reversion_001",
        "confidence": 1.0,
        "reason": "vwap_zscore=-1.83, rel_volume=1.62",
        "indicators": {"vwap_zscore": -1.83, "rel_volume": 1.62, "vix": 14.2}
    }
}
```

## Options Signals

```
BUY_CE   — Buy a Call option (bullish on underlying)
SELL_CE  — Close an existing Call position
BUY_PE   — Buy a Put option (bearish on underlying)
SELL_PE  — Close an existing Put position
FLAT     — No position (default state)
```

### State Machine Rules (Options)
- BUY_CE only valid from FLAT → enters HOLDING_CE state
- SELL_CE only valid from HOLDING_CE → returns to FLAT
- BUY_PE only valid from FLAT → enters HOLDING_PE state
- SELL_PE only valid from HOLDING_PE → returns to FLAT
- **Cannot hold CE and PE simultaneously**
- Cannot receive BUY_PE while HOLDING_CE (ignored)
- EOD flatten: force SELL_CE or SELL_PE by 15:25 IST
- Expiry day: flatten by 15:20 IST
- One position at a time. No stacking.

### Options Signal Schema
Each signal must include: timestamp, underlying, strike, expiry, action, option_premium_at_signal, stop_points, target_points, delta_at_entry, metadata.

### Numba Signal Codes
```python
FLAT    = 0
BUY_CE  = 1
SELL_CE = 2
BUY_PE  = 3
SELL_PE = 4
```

These must be consistent across state_machine.py, base.py, config.py, and every module.

## Confidence Score
Set to **1.0 for all signals**. There is no probability model — every signal that passes all conditions fires at full confidence. The field exists for the downstream OMS to use in position sizing later.

## Stop/Target in Signals
- BUY and SHORT (equity) / BUY_CE and BUY_PE (options): include stop_loss and target prices
- SELL and COVER / SELL_CE and SELL_PE: set both to `null` — closing signals don't need stops

## Decisions Made
- COVER over BUYBACK — industry standard, no ambiguity
- Confidence always 1.0 — don't engineer a probability model without the data to calibrate it
- One position at a time — simplifies state machine, eliminates portfolio-level complexity during strategy evaluation

## Corrections Log
- ~~Original proposal used BUY/SELL/SHORT/BUYBACK~~ → Corrected to BUY/SELL/SHORT/COVER
- ~~Confidence was shown as 0.72 in example~~ → Fixed to 1.0 everywhere (no probability model)
