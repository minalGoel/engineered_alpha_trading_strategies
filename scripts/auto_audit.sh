#!/bin/bash
# ============================================================================
# AUTONOMOUS STRATEGY AUDIT & FIX RUNNER
# ============================================================================

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

IMPL_DIR="$REPO_ROOT/pipeline/strategies"
LOG_DIR="$REPO_ROOT/logs"
AUDIT_TRACKER="$REPO_ROOT/logs/audited_strategies.txt"
BATCH_SIZE=10
MODEL="sonnet"
MAX_TURNS=60
RETRY_LIMIT=2
SLEEP_BETWEEN=5

mkdir -p "$LOG_DIR"
touch "$AUDIT_TRACKER"

echo "$(date '+%Y-%m-%d %H:%M:%S') ═══ PREFLIGHT ═══"

command -v claude &>/dev/null || { echo "FATAL: claude CLI not found."; exit 1; }
[[ -f "$IMPL_DIR/base.py" ]] || { echo "FATAL: base.py missing."; exit 1; }
[[ -f "$IMPL_DIR/CONVENTIONS.md" ]] || { echo "FATAL: CONVENTIONS.md missing."; exit 1; }

SAMPLE_STOCK=$(find "$REPO_ROOT/data" -maxdepth 1 -name "*.parquet" -type f 2>/dev/null | head -1)
[[ -n "$SAMPLE_STOCK" ]] || { echo "FATAL: No parquet files in data/."; exit 1; }
echo "Sample stock: $(basename "$SAMPLE_STOCK")"

ALL_STRATS=()
while IFS= read -r f; do
    ALL_STRATS+=("$(basename "$f" .py)")
done < <(find "$IMPL_DIR" -maxdepth 1 -name "*.py" -type f \
    ! -name "base.py" ! -name "__init__.py" ! -name "loader.py" ! -name "smoke_test.py" \
    2>/dev/null | sort)

REMAINING=()
for name in "${ALL_STRATS[@]}"; do
    grep -qxF "$name" "$AUDIT_TRACKER" 2>/dev/null || REMAINING+=("$name")
done

echo "Total strategies:   ${#ALL_STRATS[@]}"
echo "Already audited:    $(( ${#ALL_STRATS[@]} - ${#REMAINING[@]} ))"
echo "Remaining:          ${#REMAINING[@]}"
echo "Batches:            $(( (${#REMAINING[@]} + BATCH_SIZE - 1) / BATCH_SIZE ))"

[[ ${#REMAINING[@]} -gt 0 ]] || { echo "All done."; exit 0; }

BATCH_NUM=1
RUN_PASSED=0
RUN_FIXED=0
RUN_UNFIXABLE=0
START_TIME=$(date +%s)

for ((i=0; i<${#REMAINING[@]}; i+=BATCH_SIZE)); do
    BATCH=("${REMAINING[@]:i:BATCH_SIZE}")
    BATCH_COUNT=${#BATCH[@]}

    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  AUDIT BATCH $BATCH_NUM — strategies $((i+1))-$((i+BATCH_COUNT)) of ${#REMAINING[@]}"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "════════════════════════════════════════════════════════════"

    STRAT_LIST=""
    for name in "${BATCH[@]}"; do
        STRAT_LIST="${STRAT_LIST}- ${name}.py
"
    done

    # Write prompt to temp file to avoid shell quoting issues
    PROMPT_FILE=$(mktemp /tmp/claude_audit.XXXXXX)
    cat > "$PROMPT_FILE" << ENDPROMPT
IMPORTANT: This project uses a Python virtualenv. Before running ANY Python code, always run: source .venv/bin/activate

You are auditing and fixing trading strategy implementations in ~/Coding_Projects/engineered_alpha_trading_strategies/

## STEP 1: Activate the virtualenv and read reference files

First run: source .venv/bin/activate

Then read these files:
1. pipeline/strategies/CONVENTIONS.md — the interface spec, column definitions, all conventions.
2. pipeline/strategies/base.py — the BaseStrategy class and StrategySignals dataclass.

## STEP 2: For EACH of these ${BATCH_COUNT} strategies, do ALL of the following

${STRAT_LIST}

### 2a. Read and understand

Read pipeline/strategies/{name}.py. Understand what the strategy is trying to do.

### 2b. Verify it imports and instantiates

Run in Python (using the activated venv):
import importlib
mod = importlib.import_module("pipeline.strategies.{name}")
strat = mod.Strategy()

If this fails, fix the import/syntax error.

### 2c. Run compute() on real data

Load real 1-min parquet data for one stock from data/. Load it with Polars. Make sure the DataFrame has the required columns: datetime, open, high, low, close, volume, vix, index_close, day_id, time_minutes. If some columns need to be constructed (like day_id, time_minutes, or vix/index_close need to be joined), construct them.

Then run:
params = {p.name: p.default for p in strat.tunable_params()}
signals = strat.compute(df, params)

### 2d. Verify the output

Check ALL of these. If ANY fails, it is a bug:

1. Array lengths: Every array in StrategySignals must have len == len(df). No exceptions.
2. Dtypes: long_entry/short_entry/long_exit/short_exit must be bool or int8. stop_prices/target_prices must be float64.
3. No NaN in stop_prices or target_prices. Must be 0.0 or a valid price.
4. No NaN in entry/exit masks. NaN in boolean is truthy in numpy.
5. Signal sanity: long_entry not True on EVERY bar or ZERO bars (unless unparseable=True).
6. Stop prices: When > 0, should be below close (for longs).
7. Target prices: When > 0, should be above close (for longs).
8. is_long_only: If True, short_entry must be all False.
9. Session window: Entries only within session_start to session_end.
10. VWAP: If used, must reset daily.

### 2e. Fix any bugs found

For each bug:
- Add comment at top: # AUDIT FIX: [description]
- Fix it
- Re-run compute() to verify
- If unfixable, set unparseable = True and return all-zero arrays.

### 2f. Log the result

After auditing each strategy, print EXACTLY this line on its own (no backticks, no markdown):
AUDIT_RESULT: {name} STATUS=PASS SIGNALS=X,Y ISSUES=0
or
AUDIT_RESULT: {name} STATUS=FIXED SIGNALS=X,Y ISSUES=N
or
AUDIT_RESULT: {name} STATUS=UNFIXABLE SIGNALS=0,0 ISSUES=N

These lines must be printed as raw text, not inside code blocks.

## RULES

- Activate .venv/bin/activate before any Python.
- You MUST actually run the code on real data.
- Fix bugs in place.
- Do not modify base.py, CONVENTIONS.md, or any other pipeline file.
- Work through ALL ${BATCH_COUNT} strategies. Do not stop early.
ENDPROMPT

    SUCCESS=false
    for attempt in $(seq 1 $RETRY_LIMIT); do
        LOGFILE="$LOG_DIR/audit_batch${BATCH_NUM}_a${attempt}_$(date +%H%M%S).log"
        echo "  Attempt $attempt/$RETRY_LIMIT → $(basename "$LOGFILE")"

        if claude -p "$(cat "$PROMPT_FILE")" \
            --dangerously-skip-permissions \
            --model "$MODEL" \
            --max-turns "$MAX_TURNS" \
            --output-format text \
            < /dev/null \
            > "$LOGFILE" 2>&1; then
            SUCCESS=true
            break
        else
            echo "  ⚠ Exit code $?"
            [[ $attempt -lt $RETRY_LIMIT ]] && { echo "  Retrying in 10s..."; sleep 10; }
        fi
    done

    rm -f "$PROMPT_FILE"

    if [[ "$SUCCESS" == false ]]; then
        echo "  ✗ BATCH FAILED after $RETRY_LIMIT attempts"
        ((BATCH_NUM++))
        continue
    fi

    # Parse results
    BATCH_PASS=0; BATCH_FIXED=0; BATCH_UNFIXABLE=0; BATCH_UNKNOWN=0

    for name in "${BATCH[@]}"; do
        RESULT=$(grep "^AUDIT_RESULT: $name " "$LOGFILE" 2>/dev/null | tail -1 || true)
        if [[ -n "$RESULT" ]]; then
            STATUS=$(echo "$RESULT" | sed -n 's/.*STATUS=\([A-Z]*\).*/\1/p')
            case "$STATUS" in
                PASS)      ((BATCH_PASS++)); ((RUN_PASSED++)) ;;
                FIXED)     ((BATCH_FIXED++)); ((RUN_FIXED++)) ;;
                UNFIXABLE) ((BATCH_UNFIXABLE++)); ((RUN_UNFIXABLE++)) ;;
                *)         ((BATCH_UNKNOWN++)) ;;
            esac
            echo "$name" >> "$AUDIT_TRACKER"
        else
            echo "  ? No result for $name"
            ((BATCH_UNKNOWN++))
            # Still mark as audited so we don't retry forever
            echo "$name" >> "$AUDIT_TRACKER"
        fi
    done

    echo "  Results: $BATCH_PASS pass, $BATCH_FIXED fixed, $BATCH_UNFIXABLE unfixable, $BATCH_UNKNOWN unknown"

    git add pipeline/strategies/ 2>/dev/null
    if ! git diff --cached --quiet 2>/dev/null; then
        git commit -m "Audit batch $BATCH_NUM: ${BATCH_PASS}p ${BATCH_FIXED}f ${BATCH_UNFIXABLE}u ${BATCH_UNKNOWN}?" --quiet 2>/dev/null || true
        echo "  Committed."
    fi

    NOW=$(date +%s)
    ELAPSED_MIN=$(( (NOW - START_TIME) / 60 ))
    DONE_SO_FAR=$((i + BATCH_COUNT))
    LEFT=$((${#REMAINING[@]} - DONE_SO_FAR))
    if [[ $BATCH_NUM -gt 0 && $DONE_SO_FAR -gt 0 ]]; then
        SEC_PER_BATCH=$(( (NOW - START_TIME) / BATCH_NUM ))
        ETA_MIN=$(( (SEC_PER_BATCH * ( (LEFT + BATCH_SIZE - 1) / BATCH_SIZE )) / 60 ))
        echo "  Progress: $DONE_SO_FAR/${#REMAINING[@]} | ${ELAPSED_MIN}m elapsed | ~${ETA_MIN}m left"
    fi

    ((BATCH_NUM++))
    sleep $SLEEP_BETWEEN
done

TOTAL_ELAPSED=$(( ($(date +%s) - START_TIME) / 60 ))
AUDITED_COUNT=$(wc -l < "$AUDIT_TRACKER" | tr -d ' ')

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  AUDIT COMPLETE"
echo "  Runtime:      ${TOTAL_ELAPSED} min"
echo "  Audited:      $AUDITED_COUNT / ${#ALL_STRATS[@]}"
echo "  Passed:       $RUN_PASSED"
echo "  Fixed:        $RUN_FIXED"
echo "  Unfixable:    $RUN_UNFIXABLE"
echo "  Deployable:   $((RUN_PASSED + RUN_FIXED))"
echo "════════════════════════════════════════════════════════════"

git add pipeline/strategies/
git diff --cached --quiet 2>/dev/null || \
    git commit -m "Audit complete: ${RUN_PASSED}p ${RUN_FIXED}f ${RUN_UNFIXABLE}u" --quiet 2>/dev/null
git push origin main 2>/dev/null && echo "Pushed." || echo "Push failed."