#!/bin/bash
# ============================================================================
# AUTONOMOUS STRATEGY IMPLEMENTATION RUNNER
#
# Feeds batches of strategies to Claude Code CLI (Sonnet) in non-interactive mode.
# Each batch: read JSONs → implement → syntax check → commit → next.
#
# PREREQUISITES:
#   - Claude Code CLI installed and authenticated (`claude` command available)
#   - Batch 1 done by Opus (base.py, CONVENTIONS.md, phases.py rewired,
#     10+ example strategies exist in pipeline/strategies/)
#   - Run from repo root
#
# USAGE:
#   cd ~/Coding\ Projects/engineered_alpha_trading_strategies
#   chmod +x scripts/auto_implement.sh
#   nohup ./scripts/auto_implement.sh > logs/runner.log 2>&1 &
#   tail -f logs/runner.log
# ============================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

STRATEGIES_DIR="$REPO_ROOT/unique_strategies"
IMPL_DIR="$REPO_ROOT/pipeline/strategies"
LOG_DIR="$REPO_ROOT/logs"
BATCH_SIZE=10
MODEL="claude-sonnet-4-20250514"
MAX_TURNS=50
RETRY_LIMIT=2
SLEEP_BETWEEN=5

mkdir -p "$LOG_DIR" "$IMPL_DIR"

# ── Preflight checks ────────────────────────────────────────────────────

echo "$(date '+%Y-%m-%d %H:%M:%S') ═══ PREFLIGHT ═══"

if ! command -v claude &>/dev/null; then
    echo "FATAL: 'claude' CLI not found."
    exit 1
fi

if [[ ! -f "$IMPL_DIR/base.py" ]]; then
    echo "FATAL: pipeline/strategies/base.py missing. Run Opus batch first."
    exit 1
fi

if [[ ! -f "$IMPL_DIR/CONVENTIONS.md" ]]; then
    echo "FATAL: pipeline/strategies/CONVENTIONS.md missing. Run Opus batch first."
    exit 1
fi

EXAMPLE_COUNT=$(find "$IMPL_DIR" -maxdepth 1 -name "*.py" ! -name "base.py" ! -name "__init__.py" | wc -l | tr -d ' ')
if [[ $EXAMPLE_COUNT -lt 5 ]]; then
    echo "FATAL: Only $EXAMPLE_COUNT example strategies. Need ≥5 from Opus."
    exit 1
fi
echo "Examples found: $EXAMPLE_COUNT"

# ── Collect remaining strategies ────────────────────────────────────────

REMAINING=()
for json_file in $(ls "$STRATEGIES_DIR"/*.json 2>/dev/null | sort); do
    name=$(basename "$json_file" .json)
    if [[ ! -f "$IMPL_DIR/${name}.py" ]]; then
        REMAINING+=("$json_file")
    fi
done

TOTAL_STRATS=$(ls "$STRATEGIES_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
ALREADY_DONE=$((TOTAL_STRATS - ${#REMAINING[@]}))

echo "Total JSONs:    $TOTAL_STRATS"
echo "Done:           $ALREADY_DONE"
echo "Remaining:      ${#REMAINING[@]}"
echo "Batches:        $(( (${#REMAINING[@]} + BATCH_SIZE - 1) / BATCH_SIZE ))"

if [[ ${#REMAINING[@]} -eq 0 ]]; then
    echo "All done. Exiting."
    exit 0
fi

# ── Clear previous failure log ──────────────────────────────────────────

rm -f "$LOG_DIR/failed_strategies.txt"

# ── Batch loop ──────────────────────────────────────────────────────────

BATCH_NUM=1
RUN_IMPLEMENTED=0
RUN_FAILED=0
START_TIME=$(date +%s)

for ((i=0; i<${#REMAINING[@]}; i+=BATCH_SIZE)); do
    BATCH=("${REMAINING[@]:i:BATCH_SIZE}")
    BATCH_COUNT=${#BATCH[@]}

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  BATCH $BATCH_NUM — items $((i+1))-$((i+BATCH_COUNT)) of ${#REMAINING[@]}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════"

    # ── Strategy list for prompt
    STRAT_LIST=""
    STRAT_NAMES=""
    for s in "${BATCH[@]}"; do
        STRAT_LIST="$STRAT_LIST- $(basename $s)"$'\n'
        STRAT_NAMES="$STRAT_NAMES $(basename $s .json)"
    done

    # ── Prompt
    read -r -d '' PROMPT <<ENDPROMPT || true
You are implementing trading strategy backtests in the repo at ~/Coding Projects/engineered_alpha_trading_strategies/

## STEP 1: Read these files (in this exact order, completely)

1. pipeline/strategies/CONVENTIONS.md — full interface spec, column definitions, stop/target conventions, NaN handling, indicator snippets, complete example implementations.
2. pipeline/strategies/base.py — BaseStrategy class and StrategySignals dataclass.

These two files contain EVERYTHING you need. Do not read other pipeline modules.

## STEP 2: Implement these $BATCH_COUNT strategies

For each strategy JSON listed below:
1. Read the JSON from unique_strategies/{filename}
2. Understand the thesis — what market behavior is it exploiting?
3. Write pipeline/strategies/{name}.py with a Strategy class extending BaseStrategy
4. compute() must: compute all indicators from raw OHLCV, evaluate entry/exit conditions as boolean numpy masks, compute stop/target prices, return StrategySignals

Strategies to implement:
$STRAT_LIST

## RULES (critical — from CONVENTIONS.md)

- Input df columns: datetime, open, high, low, close, volume, vix, index_close, day_id, time_minutes. Nothing else exists. Compute every indicator yourself.
- VWAP resets daily using .over("day_id") for cumulative sums.
- NaN handling: zscore→0, vix→99, volume_ratio→0, rsi→50, atr→0. stop_prices=0.0 means no stop that bar.
- tunable_params(): ONLY numeric thresholds (the 30 in RSI<30). NEVER indicator periods (the 14 in RSI(14)).
- If conditions are genuinely vague English with no precise logic: set unparseable=True, return all-zero arrays.
- If you make interpretation choices: record in assumptions=[].
- Polars for indicator computation, convert to numpy for boolean masks and StrategySignals arrays.
- Arrays in StrategySignals must have length == len(df). No exceptions.

## OUTPUT

Write ONLY the strategy .py files. Do not modify other pipeline files. Do not run anything.
ENDPROMPT

    # ── Execute with retries
    SUCCESS=false
    for attempt in $(seq 1 $RETRY_LIMIT); do
        LOGFILE="$LOG_DIR/batch${BATCH_NUM}_a${attempt}_$(date +%H%M%S).log"
        echo "  Attempt $attempt/$RETRY_LIMIT → $(basename $LOGFILE)"

        if claude -p "$PROMPT" \
            --model "$MODEL" \
            --max-turns "$MAX_TURNS" \
            --output-format text \
            > "$LOGFILE" 2>&1; then
            SUCCESS=true
            break
        else
            echo "  ⚠ Exit code $?"
            if [[ $attempt -lt $RETRY_LIMIT ]]; then
                echo "  Retrying in 10s..."
                sleep 10
            fi
        fi
    done

    if [[ "$SUCCESS" == false ]]; then
        echo "  ✗ FAILED after $RETRY_LIMIT attempts"
        for s in "${BATCH[@]}"; do
            echo "$(basename $s .json)" >> "$LOG_DIR/failed_strategies.txt"
        done
        RUN_FAILED=$((RUN_FAILED + BATCH_COUNT))
        ((BATCH_NUM++))
        continue
    fi

    # ── Verify: file exists + valid Python syntax
    BATCH_OK=0
    BATCH_BAD=""
    for s in "${BATCH[@]}"; do
        name=$(basename "$s" .json)
        f="$IMPL_DIR/${name}.py"
        if [[ -f "$f" ]]; then
            if python3 -c "import ast; ast.parse(open('$f').read())" 2>/dev/null; then
                ((BATCH_OK++))
            else
                echo "  ✗ Syntax error: ${name}.py"
                mv "$f" "${f}.broken" 2>/dev/null || true
                BATCH_BAD="$BATCH_BAD $name"
            fi
        else
            BATCH_BAD="$BATCH_BAD $name"
        fi
    done

    BATCH_FAIL=$((BATCH_COUNT - BATCH_OK))
    RUN_IMPLEMENTED=$((RUN_IMPLEMENTED + BATCH_OK))
    RUN_FAILED=$((RUN_FAILED + BATCH_FAIL))

    echo "  ✓ $BATCH_OK/$BATCH_COUNT OK"
    if [[ -n "$BATCH_BAD" ]]; then
        echo "  ✗ Failed:$BATCH_BAD"
        for name in $BATCH_BAD; do
            echo "$name" >> "$LOG_DIR/failed_strategies.txt"
        done
    fi

    # ── Commit
    git add pipeline/strategies/ 2>/dev/null
    if ! git diff --cached --quiet 2>/dev/null; then
        git commit -m "Auto batch $BATCH_NUM: ${BATCH_OK}/${BATCH_COUNT} strategies" --quiet 2>/dev/null || true
        echo "  Committed."
    fi

    # ── ETA
    NOW=$(date +%s)
    ELAPSED_SEC=$((NOW - START_TIME))
    ELAPSED_MIN=$((ELAPSED_SEC / 60))
    DONE_TOTAL=$((ALREADY_DONE + RUN_IMPLEMENTED))
    LEFT=$((TOTAL_STRATS - DONE_TOTAL - RUN_FAILED))
    if [[ $BATCH_NUM -gt 1 ]]; then
        SEC_PER_BATCH=$((ELAPSED_SEC / (BATCH_NUM - 1)))
        BATCHES_LEFT=$(( (LEFT + BATCH_SIZE - 1) / BATCH_SIZE ))
        ETA_MIN=$(( (SEC_PER_BATCH * BATCHES_LEFT) / 60 ))
        echo "  Progress: $DONE_TOTAL/$TOTAL_STRATS | ${ELAPSED_MIN}m elapsed | ~${ETA_MIN}m remaining"
    fi

    ((BATCH_NUM++))
    sleep $SLEEP_BETWEEN
done

# ── Summary ─────────────────────────────────────────────────────────────

TOTAL_ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))
FINAL_COUNT=$(find "$IMPL_DIR" -maxdepth 1 -name "*.py" ! -name "base.py" ! -name "__init__.py" | wc -l | tr -d ' ')

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  COMPLETE"
echo ""
echo "  Runtime:          ${TOTAL_ELAPSED} min"
echo "  Implementations:  $FINAL_COUNT / $TOTAL_STRATS"
echo "  New this run:     $RUN_IMPLEMENTED"
echo "  Failed this run:  $RUN_FAILED"
if [[ -f "$LOG_DIR/failed_strategies.txt" ]]; then
    echo "  Failed list:      $LOG_DIR/failed_strategies.txt"
fi
echo "════════════════════════════════════════════════════════════"

# ── Final push
git add pipeline/strategies/
git diff --cached --quiet 2>/dev/null || \
    git commit -m "Auto-implementation done: $FINAL_COUNT/$TOTAL_STRATS strategies" --quiet 2>/dev/null
git push origin main 2>/dev/null && echo "Pushed." || echo "⚠ Push failed — run 'git push' manually."