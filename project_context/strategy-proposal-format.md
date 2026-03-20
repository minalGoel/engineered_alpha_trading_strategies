# Strategy Proposal Format

## Summary
A standardized JSON schema used to define trading strategies in a machine-readable format that any AI agent (Claude, GPT, Gemini) can fill out. The format evolved from an equity spot version to an options-specific version when the project pivoted to 5-second NIFTY/BANKNIFTY options.

## The Equity Format (Original)

The core insight: a strategy proposal is a **spec sheet**, not prose. Every field must have a value. If a field can't be filled, the idea isn't ready.

### Schema

```json
{
    "name": "unique_slug_no_spaces_v1",
    "version": "0.1.0",
    "author": "who/what proposed this",
    "tags": ["mean_reversion", "momentum", "breakout"],

    "thesis": "One paragraph answering: (1) What inefficiency? (2) WHY does it exist? (3) Why hasn't it been arbitraged away?",

    "universe": "nifty50 | nifty200 | fno | nifty500 | custom",
    "stock_filter": "how to pick specific stocks",
    "timeframe": "1min | 5min | 15min",
    "session": "09:20-15:15",
    "hold_time": "5-30 bars",
    "max_hold_bars": 30,
    "trades_per_day": "1-3",
    "max_trades_per_day": 6,

    "data": {
        "ohlcv_1min": true,
        "vwap": false,
        "vix": false,
        "index": "nifty50 | banknifty | none",
        "option_chain": false,
        "order_book": false,
        "other": []
    },

    "indicators": [
        {
            "name": "indicator_slug",
            "formula": "EMA(close, 20)",
            "lookback": 20,
            "purpose": "trend filter"
        }
    ],

    "entry": {
        "long": {
            "conditions": ["close < vwap", "vwap_zscore < -1.5", "rel_volume > 1.3"],
            "confirmation": ""
        },
        "short": {
            "conditions": [],
            "confirmation": ""
        },
        "entry_price": "market | limit at vwap | close of signal bar"
    },

    "exit": {
        "target": "vwap touch | 1.5x ATR from entry | fixed 0.5%",
        "stop_loss": "0.3% from entry | 1.5x ATR | below ORB low",
        "trailing_stop": "none | move SL to breakeven after 0.3%",
        "time_stop": "exit after 30 bars",
        "eod_rule": "flatten",
        "signal_exit": "exit when zscore crosses 0"
    },

    "risk": {
        "capital_per_trade": 100000,
        "max_risk_per_trade_pct": 0.3,
        "position_sizing": "fixed | vol_adjusted | confidence_scaled",
        "max_open_positions": 1,
        "max_daily_loss_inr": 5000,
        "max_portfolio_heat_pct": 2.0
    },

    "filters": {
        "vix_filter": "vix < 20",
        "time_filter": "no trades in first 5 min",
        "volume_filter": "skip if volume < 50% of 20-bar avg",
        "trend_filter": "only long if index_return_15 > 0",
        "spread_filter": "skip if bid-ask > 0.1%",
        "event_filter": "skip on expiry day",
        "other_filters": []
    },

    "edge": {
        "expected_bps": 20,
        "win_rate_estimate": 0.55,
        "avg_winner_to_loser": 1.2,
        "profit_factor_estimate": 1.5,
        "edge_decay_risk": "low — structural | medium — behavioral | high — crowded"
    },

    "costs": {
        "slippage_bps_per_side": 3,
        "brokerage": "zerodha_intraday",
        "stt_bps": 2.5,
        "exchange_charges_bps": 0.5
    },

    "weaknesses": ["minimum 2 specific failure modes"],
    "prior_art": "Classic VWAP reversion | Connors RSI variant | Original",
    "notes": ""
}
```

### Agent System Prompt Rules
When giving this form to an AI agent, the prompt enforces:
- **Thesis first**: Must answer what, why, and why-not-arbitraged
- **Machine-readable conditions**: `"close < vwap AND zscore < -1.5"` not `"enter when stock is oversold"`
- **Exact indicator parameters**: `"EMA(close, 9)"` not `"short-term moving average"`
- **Mandatory stop loss**: Every entry must define a stop
- **Honest weaknesses**: Zero weaknesses = rejected proposal
- **Internally consistent edge math**: `edge ≈ (win_rate × avg_winner) - ((1-win_rate) × avg_loser) - costs`
- **No overfitting bait**: Max 5 tunable parameters without justification

## The Options Format (Current)

When the project pivoted to 5-second NIFTY/BANKNIFTY options, the schema changed significantly:

```json
{
    "name": "underlying_lag_ce_scalp_v1",
    "version": "1.0.0",
    "author": "adapted from [original]",
    "original_strategy": "[original name]",
    "original_source": "trading_strategies/unique_strategies_all/[file]",

    "underlying": "NIFTY | BANKNIFTY | BOTH",
    "timeframe": "5s",
    "session": "09:18-15:25 IST",
    "hold_time_seconds": [15, 60],
    "max_hold_bars": 24,
    "max_trades_per_day": 20,

    "thesis": "...",
    "mechanism": "MANDATORY: Why this works at the microstructure level on THIS index",

    "indicators": [
        {
            "name": "spot_return_6bar",
            "formula": "(close - close[6]) / close[6]",
            "lookback": 6,
            "computed_on": "spot | option | vix",
            "purpose": "30-second momentum"
        }
    ],

    "entry": {
        "buy_ce": {"conditions": ["spot_return_6bar > 0.0003"]},
        "buy_pe": {"conditions": ["spot_return_6bar < -0.0003"]},
        "strike_selection": "ATM",
        "delta_filter": "abs(delta) > 0.35"
    },

    "exit": {
        "target_points": 5,
        "stop_points": 3,
        "time_stop_bars": 12,
        "signal_exit": "...",
        "eod_flatten": "15:25 IST"
    },

    "cost_model": {
        "spread_per_side": 1.0,
        "broker": "upstox",
        "min_edge_points_after_costs": 4
    },

    "tunable_params": [
        {"name": "threshold", "default": 0.0003, "low": 0.0001, "high": 0.001}
    ],

    "weaknesses": ["..."],
    "adaptation_notes": "What changed from original and why"
}
```

### Key Differences from Equity Format
- `mechanism` field is **mandatory** — must explain WHY the edge exists in microstructure terms, not just indicator logic
- `computed_on` per indicator — distinguishes spot vs option vs VIX data
- Stops/targets in **option premium points**, not spot % or bps
- `entry` uses `buy_ce`/`buy_pe` instead of `long`/`short`
- `cost_model` is explicit (was excluded in late equity phase, now included for direct option trading)
- `original_source` traces back to the source strategy

## Decisions Made
- The format must be rigid enough that strategies from different AI agents are directly comparable
- Conditions must be machine-readable boolean expressions, not English prose
- Weaknesses are mandatory (minimum 2) — zero weaknesses = rejected
- `mechanism` field was added specifically because the original triage used thesis alone, which wasn't sufficient to filter quality

## Pitfalls & Anti-patterns
- **Template mechanisms**: When batch-converting strategies, AI agents write ~30 generic mechanism templates and copy-paste across hundreds of strategies. The `mechanism` field only works as a quality gate if each is genuinely unique.
- **Category-default stops**: Assigning stops by strategy type ("scalp: 2/3, breakout: 4/7") instead of per-strategy reasoning produces useless results.
- **Blind lookback scaling**: When converting from 1-min to 5-second bars, multiplying all lookback periods by 12 preserves time but may not preserve the strategy's intent. Each lookback needs individual reasoning.

## Corrections Log
- ~~Equity format had no `mechanism` field~~ → Added `mechanism` as mandatory quality gate in the options format after the first triage produced 291 strategies with zero mechanism failures (proving the gate was fake)
- ~~Cost model was excluded from equity backtesting~~ → Cost exclusion was correct for equity (spot signals triggering option buys externally) but costs MUST be included for direct option trading
- ~~`computed_on` field didn't exist~~ → Added to distinguish which data source each indicator uses (critical when you have spot, option chain, and VIX data)
