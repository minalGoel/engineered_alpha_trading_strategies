# Strategy Implementation Pattern

## Summary
Why generic condition parsers fail, the LLM-per-strategy approach, the BaseStrategy interface for both equity and options, and the role of CONVENTIONS.md as the quality bridge between Opus (design) and Sonnet (implementation) sessions.

## Why Generic Parsers Fail

The first approach was building a regex-based condition parser that reads English conditions from strategy JSONs and translates them to code:
```
"close < vwap AND vwap_zscore < -1.5" → df["close"] < df["vwap"]) & (df["vwap_zscore"] < -1.5)
```

**Failure rate: 81%**. Reasons:
- Complex conditions with nested logic, OR branches, and lookback references
- Indicator names don't match DataFrame column names
- Ambiguous phrasing: "price touches the lower band" — which price? which band?
- Cross-references: "when RSI exits oversold" — needs state tracking, not a boolean
- Strategy-specific edge cases: "only on the first occurrence per day"

## The LLM-Per-Strategy Approach

Each strategy gets a dedicated Python implementation written by an LLM that reads the strategy JSON and writes real code. No parser, no interpreter. The LLM understands the INTENT behind "when RSI exits oversold" and writes appropriate state-tracking code.

**Implementation flow**:
1. LLM reads CONVENTIONS.md (interface spec + examples)
2. LLM reads base.py (exact class interface)
3. LLM reads ONE strategy JSON
4. LLM writes pipeline/strategies/{name}.py with actual indicator computation and boolean mask logic
5. LLM smoke-tests on real data
6. If compute() fails → LLM fixes and retests

**Result**: 358 strategies implemented this way. After audit, 357 of 358 produced valid compute() output on real data (3 strategies verified in detail: all PASS).

## The BaseStrategy Interface — Equity Version

```python
@dataclass
class StrategySignals:
    long_entry: np.ndarray     # bool
    short_entry: np.ndarray    # bool
    long_exit: np.ndarray      # bool
    short_exit: np.ndarray     # bool
    stop_prices: np.ndarray    # float64 — 0.0 means no stop
    target_prices: np.ndarray  # float64 — 0.0 means no target

class BaseStrategy:
    name: str
    is_long_only: bool
    session_start: int         # minutes from midnight IST
    session_end: int
    unparseable: bool = False  # True = return all zeros

    def tunable_params(self) -> list[TunableParam]: ...
    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals: ...
```

## The BaseStrategy Interface — Options Version

```python
@dataclass
class OptionSignals:
    buy_ce: np.ndarray         # bool
    buy_pe: np.ndarray         # bool
    sell_ce: np.ndarray        # bool
    sell_pe: np.ndarray        # bool
    stop_points: np.ndarray    # float64 — option premium stop
    target_points: np.ndarray  # float64 — option premium target
    strike_offset: np.ndarray  # int — 0=ATM, 1=ATM+1, -1=ATM-1
    time_stop_bars: int
    max_trades_per_day: int

class BaseStrategy:
    name: str
    underlying: str            # "NIFTY" or "BANKNIFTY" or "BOTH"
    session_start_minutes: int
    session_end_minutes: int

    def tunable_params(self) -> list[TunableParam]: ...
    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals: ...
```

## CONVENTIONS.md — The Quality Bridge

CONVENTIONS.md is the single most important file in the repo. It's the contract between:
- The Opus session that designs the architecture
- The Sonnet sessions that mass-produce implementations
- Future human developers reading the code

### What CONVENTIONS.md Must Contain
1. **Exact interface**: copy of the dataclass and class signatures
2. **Column definitions**: every column in the input DataFrames, its type, its meaning
3. **NaN handling rules**: what to do with NaN in each array type
4. **Stop/target conventions**: what 0.0 means, units (option points vs spot %)
5. **Session filtering**: how time_minutes works, session boundaries
6. **Common indicator snippets**: RSI, VWAP, ATR as copy-paste templates
7. **2-3 COMPLETE example implementations**: full .py files that work, not pseudocode
8. **Anti-patterns**: things explicitly NOT to do

### Why This Matters for Automation
When Sonnet implements strategy #147 in a batch script, it reads CONVENTIONS.md fresh. The quality of that implementation depends entirely on the quality of the examples and rules in CONVENTIONS.md. Bad examples → bad implementations × 291. Good examples → good implementations × 291.

The gold-standard Opus session that writes CONVENTIONS.md is the highest-leverage single session in the entire project.

## TunableParam Convention

Parameters are divided into:
- **Tunable**: Numeric thresholds that Optuna can search. `{"name": "rsi_threshold", "default": 30, "low": 20, "high": 40}`
- **Fixed**: Indicator periods (lookback windows) that define the strategy's identity. NOT tunable.

This split allows indicator pre-computation: compute RSI(14) ONCE, then each Optuna trial only changes the threshold comparison (`< 30` vs `< 25` vs `< 35`).

## NaN Safety Rules

1. **stop_points / target_points**: NaN → 0.0 (means "no stop/target this bar"). NaN in Numba state machine causes corruption.
2. **Boolean masks** (buy_ce, buy_pe, etc.): NaN → False. NaN is truthy in numpy — it would trigger entries on every NaN bar.
3. **Indicators**: Strategy-specific defaults. RSI NaN → 50, z-score NaN → 0, volume_ratio NaN → 0, VIX NaN → 99 (extremely high = no trades).
4. **Price arrays**: Forward-fill in Polars BEFORE converting to numpy. Never leave NaN in price data.

## Decisions Made
- LLM writes code, not a parser — 81% failure rate settled this permanently
- CONVENTIONS.md is the single source of truth — all sessions read it first
- Tunable = thresholds only, never indicator periods
- Indicators computed ONCE, not per Optuna trial
- `unparseable = True` is a valid verdict — return all-zero arrays, don't crash

## Pitfalls & Anti-patterns
- **Building a generic condition parser**: Fails on 81% of strategies. Don't even start.
- **Sonnet writing implementations without CONVENTIONS.md**: The quality collapses. Examples in CONVENTIONS.md are the single most important quality lever.
- **Batching 20+ strategies per LLM session**: Quality degrades after ~10. For highest quality, 1 strategy per session (fresh context each time).
- **Template mechanisms across strategies**: When batch-converting, the LLM writes ~30 mechanism templates and copy-pastes. Force unique mechanisms by including a self-check rule.

## Corrections Log
- ~~Initial plan: build a regex condition parser for all strategies~~ → Abandoned after 81% failure rate. LLM per strategy is the correct approach.
- ~~Batch size of 10-20 per Sonnet session~~ → Reduced to 5 for conversion quality, ultimately 1-per-session for diamond standard
- ~~CONVENTIONS.md written by Sonnet during first implementation session~~ → Must be written by Opus during gold-standard session (it's the quality template for everything downstream)
