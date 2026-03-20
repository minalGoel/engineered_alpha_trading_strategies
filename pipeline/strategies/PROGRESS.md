# Strategy Implementation Progress

## Stats
- Total JSON strategies: 358
- Implemented: 358
- Smoke test: 358 passed, 0 failed (87 WARN no-signals on synthetic data — expected)
- Unparseable: 0

## Source Breakdown
| Source | Count | Status |
|--------|-------|--------|
| GPT (10) | 10 | Done |
| Grok (9) | 9 | Done |
| Gemini (20) | 20 | Done |
| Claude Project (25) | 25 | Done |
| Opus (48) | 48 | Done |
| Cursor GPT54 (10) | 10 | Done |
| Cursor Gemini31Pro (33) | 33+ | Done |
| Cursor Opus46Max (200) | 200 | Done |

## Implementation Notes
- All strategies follow BaseStrategy/StrategySignals interface in `base.py`
- Each strategy has dedicated `compute()` with hand-written indicator logic
- No regex parser used — every strategy is a bespoke Python implementation
- VWAP uses daily reset via `.over("day_id")` in Polars expression context
- ATR uses Wilder's smoothing (loop-based)
- RSI uses Wilder's smoothing with proper warmup
- NaN handling: VIX→99.0, volume→1.0, indicators→neutral values
- Exit params are scalar (matching state machine interface)
- One performance fix: cursor_opus46max_143 volume profile computed per-day instead of incrementally
