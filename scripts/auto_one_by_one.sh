#!/bin/bash
# ============================================================================
# ONE-AT-A-TIME STRATEGY: CONVERT → IMPLEMENT → BACKTEST → VERIFY
#
# Fresh Sonnet session per strategy. Full pipeline per strategy.
#
# NOTE: `-e` intentionally omitted from `set` flags. The main loop handles
# failures manually per-strategy. Adding `-e` would abort the entire run on
# the first strategy that has a non-zero exit from any subcommand (e.g.,
# grep finding no match = exit 1). This is a deliberate design choice.
#
# USAGE:
#   cd ~/Coding_Projects/engineered_alpha_trading_strategies
#   chmod +x scripts/auto_one_by_one.sh
#   mkdir -p logs/per_strategy strategy_results
#
#   # Normal run:
#   nohup ./scripts/auto_one_by_one.sh > logs/one_by_one_runner.log 2>&1 &
#
#   # Retry previously failed strategies:
#   nohup ./scripts/auto_one_by_one.sh --retry-failed > logs/one_by_one_runner.log 2>&1 &
#
# MONITOR (use watch, not tail -f):
#   watch -n 10 'tail -20 logs/one_by_one_runner.log'
#
# STOP:
#   pkill -f auto_one_by_one.sh; pkill -f "claude -p"
# ============================================================================

set -uo pipefail
# `-e` intentionally omitted — see note above

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ── Paths ───────────────────────────────────────────────────────────────
SOURCE_DIR="$REPO_ROOT/trading_strategies/unique_strategies_all"
JSON_DIR="$REPO_ROOT/unique_strategies"
IMPL_DIR="$REPO_ROOT/pipeline/strategies"
LOG_DIR="$REPO_ROOT/logs/per_strategy"
TRACKER="$REPO_ROOT/logs/completed_strategies.txt"
FAILED_FILE="$REPO_ROOT/logs/failed_strategies.txt"
KILLS_FILE="$REPO_ROOT/outputs/conversion_kills.json"
RESULTS_DIR="$REPO_ROOT/strategy_results"

# ── Settings ────────────────────────────────────────────────────────────
MODEL="sonnet"
MAX_TURNS=40
SLEEP_BETWEEN=3
COMMIT_EVERY=5
HEARTBEAT_INTERVAL=60
RETRY_FAILED=false

# Parse CLI args
for arg in "$@"; do
    case "$arg" in
        --retry-failed) RETRY_FAILED=true ;;
    esac
done

# ── Setup dirs ──────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR" "$JSON_DIR" "$RESULTS_DIR" "$(dirname "$KILLS_FILE")"
touch "$TRACKER" "$FAILED_FILE"

# Clean stale temp files from prior crashed runs
rm -f /tmp/claude_one.*

# Validate/init kills file (protects against half-written JSON from prior crash)
if [[ -s "$KILLS_FILE" ]]; then
    if ! python3 -c "import json; json.load(open('$KILLS_FILE'))" 2>/dev/null; then
        echo "[]" > "$KILLS_FILE"
    fi
else
    echo "[]" > "$KILLS_FILE"
fi

# ── Logging ─────────────────────────────────────────────────────────────
log() {
    local msg="${1:-}"
    [[ -z "$msg" ]] && return
    echo "$(date '+%Y-%m-%d %H:%M:%S') $msg"
}

# ── Cleanup trap ────────────────────────────────────────────────────────
heartbeat_pid=""
cleanup() {
    if [[ -n "$heartbeat_pid" ]]; then
        kill "$heartbeat_pid" 2>/dev/null
        wait "$heartbeat_pid" 2>/dev/null
    fi
    rm -f /tmp/claude_one.*
    log "INTERRUPTED — progress saved to $TRACKER"
    exit 130
}
trap cleanup INT TERM

# ── Heartbeat ───────────────────────────────────────────────────────────
start_heartbeat() {
    local strat_name="$1"
    local parent_pid=$$
    (
        while true; do
            sleep "$HEARTBEAT_INTERVAL"
            # Check if a claude child of THIS script is still running
            if pgrep -P "$parent_pid" -f "claude" > /dev/null 2>&1; then
                log "  [HEARTBEAT] $strat_name — claude running"
            else
                break
            fi
        done
    ) &
    heartbeat_pid=$!
}

stop_heartbeat() {
    if [[ -n "$heartbeat_pid" ]]; then
        kill "$heartbeat_pid" 2>/dev/null
        wait "$heartbeat_pid" 2>/dev/null
        heartbeat_pid=""
    fi
}

# ── Preflight ───────────────────────────────────────────────────────────
log "═══ PREFLIGHT ═══"

if ! command -v claude &>/dev/null; then
    log "FATAL: 'claude' CLI not found."
    exit 1
fi

if [[ ! -f "$IMPL_DIR/base.py" ]]; then
    log "FATAL: base.py missing."
    exit 1
fi

if [[ ! -f "$IMPL_DIR/CONVENTIONS.md" ]]; then
    log "FATAL: CONVENTIONS.md missing."
    exit 1
fi

if [[ ! -f "$REPO_ROOT/outputs/strategy_retriage.json" ]]; then
    log "FATAL: strategy_retriage.json missing."
    exit 1
fi

if ! find "$REPO_ROOT/5second_data" -name "*.parquet" -type f 2>/dev/null | head -1 | grep -q .; then
    log "FATAL: No parquet files in 5second_data/."
    exit 1
fi

log "All preflight checks passed."

# ── Build work list ─────────────────────────────────────────────────────
QUALIFIED=()
while IFS=$'\t' read -r qname qsrc; do
    [[ -n "$qname" ]] && QUALIFIED+=("${qname}	${qsrc}")
done < <(python3 -c "
import json, sys
try:
    with open('$REPO_ROOT/outputs/strategy_retriage.json') as f:
        data = json.load(f)
    for s in data.get('qualified_strategies', []):
        print(s['name'] + '\t' + s['source_file'])
except Exception as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
")

if [[ ${#QUALIFIED[@]} -eq 0 ]]; then
    log "FATAL: No qualified strategies in retriage.json."
    exit 1
fi

# Build remaining list based on mode
REMAINING=()
if [[ "$RETRY_FAILED" == true ]]; then
    log "MODE: --retry-failed — retrying previously failed strategies"
    while IFS= read -r fname; do
        if [[ -n "$fname" ]]; then
            # Look up source_file for this name from QUALIFIED
            for entry in "${QUALIFIED[@]}"; do
                entry_name="${entry%%	*}"
                if [[ "$entry_name" == "$fname" ]]; then
                    REMAINING+=("$entry")
                    break
                fi
            done
        fi
    done < "$FAILED_FILE"
    # Clear failed file so retries get a fresh slate
    > "$FAILED_FILE"
else
    for entry in "${QUALIFIED[@]}"; do
        entry_name="${entry%%	*}"
        if ! grep -qxF "$entry_name" "$TRACKER" 2>/dev/null; then
            REMAINING+=("$entry")
        fi
    done
fi

log "Qualified total: ${#QUALIFIED[@]}"
log "Already done:    $(wc -l < "$TRACKER" | tr -d ' ')"
log "Failed (prior):  $(wc -l < "$FAILED_FILE" | tr -d ' ')"
log "Remaining:       ${#REMAINING[@]}"
log "Mode:            $(if [[ "$RETRY_FAILED" == true ]]; then echo "retry-failed"; else echo "normal"; fi)"

if [[ ${#REMAINING[@]} -eq 0 ]]; then
    log "Nothing to do."
    exit 0
fi

# ── Prompt template ─────────────────────────────────────────────────────
# Written to temp file with QUOTED heredoc (no variable expansion inside).
# Strategy name and source path injected via sed AFTER writing.
# This prevents prompt injection if a strategy name contains shell metacharacters.

write_prompt() {
    local prompt_file="$1"
    local strat_name="$2"
    local strat_src="$3"

    cat > "$prompt_file" << 'ENDPROMPT'
IMPORTANT: Run "source .venv/bin/activate" before any Python command.

You are working in ~/Coding_Projects/engineered_alpha_trading_strategies/

You have ONE strategy to process end-to-end: convert, implement, backtest, verify.

## READ FIRST (mandatory, in this order):
1. pipeline/strategies/CONVENTIONS.md — format spec, conversion principles, gold-standard examples
2. pipeline/strategies/base.py — OptionSignals interface and BaseStrategy class
3. 5second_data/SCHEMA.md — data schema. Timestamps are UTC. Convert to IST for all trading logic.

## THE STRATEGY TO PROCESS:
File: __STRATEGY_SOURCE__

Read it now. This is an equity spot strategy for 1-minute bars. You are converting it to 5-second NIFTY/BANKNIFTY index options.

## STEP 1: DECIDE — CONVERT OR KILL

Goal: predict SPOT DIRECTION on NIFTY/BANKNIFTY at 15-60 second horizons. Trade ATM options (buy CE for bullish, buy PE for bearish). A 10-point NIFTY move = ~5 point option move at delta 0.5.

KILL if:
- Requires universe of stocks (cross-sectional, sector rotation, pair trading between equities)
- Depends on equity-specific data (delivery %, SLB, promoter holdings, earnings)
- Only works at 15+ minute horizons with no faster version of the same idea
- Makes no sense on an index (stock-specific patterns)
- You cannot write a genuine, specific mechanism for WHY this works at 5s on NIFTY

If KILLED: read outputs/conversion_kills.json, append {"name": "__STRATEGY_NAME__", "reason": "your reason"}, write it back. Then print exactly:
RESULT: __STRATEGY_NAME__ STATUS=KILLED
Stop here.

## STEP 2: CONVERT THE JSON

Write unique_strategies/__STRATEGY_NAME__.json following CONVENTIONS.md format.

MECHANISM (mandatory, UNIQUE to this strategy):
What happens in the NIFTY/BANKNIFTY order book when these specific conditions are met? Not a generic paragraph. 2-4 sentences specific to THIS strategy on THIS index.

LOOKBACK (must have rationale — not just 12x the original):
For a 30-60 second option hold, what lookback gives useful signal? Write why.

STOPS/TARGETS (individual rationale, not category default):
Typical NIFTY 30s move: 5-8 spot points = 2.5-4 option points at delta 0.5. Write a 1-sentence rationale for your specific choice.

## STEP 3: IMPLEMENT THE PYTHON

Write pipeline/strategies/__STRATEGY_NAME__.py:
- class Strategy(BaseStrategy) with name, underlying, session times
- tunable_params() — thresholds only, never indicator periods
- compute(spot_df, option_df, vix_df, params) returning OptionSignals
- All indicators on spot_df unless explicitly option/vix
- NaN replaced with neutral values before boolean masks
- buy_ce for bullish, buy_pe for bearish
- Stops/targets in option premium points

## STEP 4: SMOKE TEST

Source the venv. Load real data from 5second_data/ for one session_date. Convert UTC to IST. VERIFY first bar hour == 9 in IST.

Run compute() with default params. Check:
- All arrays len == len(spot_df). No NaN in masks or stops.
- Signals within session window. Not all-True or all-False.
Print: buy_ce count, buy_pe count.
If compute() crashes: FIX and retest.

## STEP 5: END-TO-END BACKTEST

Run through state machine on ALL 12 days. Print first 3 trades:
| Entry (IST) | Action | Strike | Entry prem | Exit prem | Same strike? | Hold bars | Gross PnL | Exit reason |

## STEP 6: VERIFY STRIKE LOCKING

For trade #1: query option_candles.parquet directly. Confirm entry and exit premiums match raw parquet for the SAME strike. Print: "Strike lock verified: YES/NO"

## STEP 7: SIGNAL-CONDITION ALIGNMENT

For trade #1: print each condition and actual value at entry bar. All must PASS.

## STEP 8: COST MODEL AND SUMMARY

Apply Upstox costs (spread 1.00/side). Save strategy_results/__STRATEGY_NAME__/summary.json:
{"name": "__STRATEGY_NAME__", "status": "CONVERTED", "total_trades": N, "gross_pnl_inr": X, "net_pnl_inr": X, "avg_gross_per_trade": X, "avg_net_per_trade": X, "win_rate_gross": X, "win_rate_net": X, "signals_buy_ce": N, "signals_buy_pe": N, "strike_lock_verified": true, "condition_alignment_verified": true, "first_bar_hour_ist": 9, "mechanism_first_20_words": "..."}

## FINAL OUTPUT — print EXACTLY one of these lines:

RESULT: __STRATEGY_NAME__ STATUS=CONVERTED TRADES={n} GROSS={gross_pnl} NET={net_pnl} VERIFIED={YES/NO}
RESULT: __STRATEGY_NAME__ STATUS=KILLED
RESULT: __STRATEGY_NAME__ STATUS=FAILED REASON="{reason}"
ENDPROMPT

    # Inject strategy-specific values via sed (safe — no shell expansion in the heredoc)
    sed -i '' "s|__STRATEGY_NAME__|${strat_name}|g" "$prompt_file"
    sed -i '' "s|__STRATEGY_SOURCE__|${strat_src}|g" "$prompt_file"
}

# ── Main loop ───────────────────────────────────────────────────────────
DONE_COUNT=0
CONVERTED=0
KILLED=0
FAILED=0
START_TIME=$(date +%s)

for entry in "${REMAINING[@]}"; do
    name="${entry%%	*}"
    SRC="${entry#*	}"
    ((DONE_COUNT++))
    STRAT_START=$(date +%s)

    # Resolve source file path (make absolute if relative)
    [[ "$SRC" != /* ]] && SRC="$REPO_ROOT/$SRC"
    if [[ ! -f "$SRC" ]]; then
        log "[$DONE_COUNT/${#REMAINING[@]}] $name — ✗ SOURCE NOT FOUND: $SRC"
        echo "$name" >> "$TRACKER"
        echo "$name" >> "$FAILED_FILE"
        ((FAILED++))
        continue
    fi

    log "[$DONE_COUNT/${#REMAINING[@]}] $name — starting"

    # Write prompt (quoted heredoc + sed injection = no prompt injection)
    PROMPT_FILE=$(mktemp /tmp/claude_one.XXXXXX)
    write_prompt "$PROMPT_FILE" "$name" "$SRC"

    LOGFILE="$LOG_DIR/${name}.log"

    # Start heartbeat
    start_heartbeat "$name"

    # Run claude — fully headless
    # NOTE: $(cat ...) puts prompt in argv. Current prompt is ~3KB, well under
    # macOS ARG_MAX of ~1MB. If prompts grow significantly, switch to piping.
    CLAUDE_EXIT=0
    claude -p "$(cat "$PROMPT_FILE")" \
        --dangerously-skip-permissions \
        --model "$MODEL" \
        --max-turns "$MAX_TURNS" \
        --output-format text \
        < /dev/null \
        > "$LOGFILE" 2>&1 || CLAUDE_EXIT=$?

    # Cleanup
    stop_heartbeat
    rm -f "$PROMPT_FILE"

    # Timing
    STRAT_SEC=$(( $(date +%s) - STRAT_START ))
    STRAT_MIN=$(( STRAT_SEC / 60 ))
    STRAT_REM=$(( STRAT_SEC % 60 ))

    # ── Handle claude failure ───────────────────────────────────────────
    if [[ $CLAUDE_EXIT -ne 0 ]]; then
        log "  ✗ CLAUDE EXIT $CLAUDE_EXIT (${STRAT_MIN}m${STRAT_REM}s) — see $LOG_DIR/${name}.log"
        echo "$name" >> "$FAILED_FILE"
        echo "$name" >> "$TRACKER"
        ((FAILED++))
        sleep "$SLEEP_BETWEEN"
        continue
    fi

    if [[ ! -s "$LOGFILE" ]]; then
        log "  ✗ EMPTY LOG (${STRAT_MIN}m${STRAT_REM}s)"
        echo "$name" >> "$FAILED_FILE"
        echo "$name" >> "$TRACKER"
        ((FAILED++))
        sleep "$SLEEP_BETWEEN"
        continue
    fi

    # ── Parse RESULT line ───────────────────────────────────────────────
    RESULT=$(grep "^RESULT:" "$LOGFILE" 2>/dev/null | tail -1 || true)
    if [[ -z "$RESULT" ]]; then
        RESULT=$(grep "RESULT: ${name}" "$LOGFILE" 2>/dev/null | tail -1 || true)
    fi

    if echo "$RESULT" | grep -q "STATUS=CONVERTED"; then
        TRADES=$(echo "$RESULT" | sed -n 's/.*TRADES=\([0-9-]*\).*/\1/p')
        NET=$(echo "$RESULT" | sed -n 's/.*NET=\([^ ]*\).*/\1/p')
        VERIFIED=$(echo "$RESULT" | sed -n 's/.*VERIFIED=\([A-Z]*\).*/\1/p')
        log "  ✓ CONVERTED | trades=${TRADES:-?} | net=${NET:-?} | verified=${VERIFIED:-?} | ${STRAT_MIN}m${STRAT_REM}s"
        echo "$name" >> "$TRACKER"
        ((CONVERTED++))

    elif echo "$RESULT" | grep -q "STATUS=KILLED"; then
        log "  ✗ KILLED | ${STRAT_MIN}m${STRAT_REM}s"
        echo "$name" >> "$TRACKER"
        ((KILLED++))

    elif echo "$RESULT" | grep -q "STATUS=FAILED"; then
        REASON=$(echo "$RESULT" | sed -n 's/.*REASON="\([^"]*\)".*/\1/p')
        log "  ✗ FAILED: ${REASON:-unknown} | ${STRAT_MIN}m${STRAT_REM}s"
        echo "$name" >> "$FAILED_FILE"
        echo "$name" >> "$TRACKER"
        ((FAILED++))

    else
        log "  ? NO RESULT LINE | ${STRAT_MIN}m${STRAT_REM}s"
        tail -3 "$LOGFILE" 2>/dev/null | while IFS= read -r line; do
            log "    $line"
        done
        echo "$name" >> "$FAILED_FILE"
        echo "$name" >> "$TRACKER"
        ((FAILED++))
    fi

    # ── Periodic commit ─────────────────────────────────────────────────
    if (( DONE_COUNT % COMMIT_EVERY == 0 )); then
        git add unique_strategies/ pipeline/strategies/ strategy_results/ outputs/ 2>/dev/null
        if ! git diff --cached --quiet 2>/dev/null; then
            git commit -m "Auto [$DONE_COUNT/${#REMAINING[@]}]: ${CONVERTED}conv ${KILLED}kill ${FAILED}fail" \
                --quiet 2>/dev/null || true
            log "  [GIT] Committed"
        fi
    fi

    # ── ETA ──────────────────────────────────────────────────────────────
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    LEFT=$((${#REMAINING[@]} - DONE_COUNT))
    if [[ $DONE_COUNT -gt 0 && $ELAPSED -gt 0 ]]; then
        SEC_PER=$((ELAPSED / DONE_COUNT))
        ETA_SEC=$((SEC_PER * LEFT))
        ETA_HR=$((ETA_SEC / 3600))
        ETA_MIN=$(( (ETA_SEC % 3600) / 60 ))
        ELAPSED_MIN=$((ELAPSED / 60))
        log "  ── [$DONE_COUNT/${#REMAINING[@]}] ${ELAPSED_MIN}m elapsed | ~${ETA_HR}h${ETA_MIN}m left | ✓${CONVERTED} ✗${KILLED} !${FAILED}"
    fi

    sleep "$SLEEP_BETWEEN"
done

# ── Final commit + summary ──────────────────────────────────────────────
git add unique_strategies/ pipeline/strategies/ strategy_results/ outputs/ 2>/dev/null
git diff --cached --quiet 2>/dev/null || \
    git commit -m "DONE: ${CONVERTED}conv ${KILLED}kill ${FAILED}fail of ${#REMAINING[@]}" \
        --quiet 2>/dev/null

TOTAL_SEC=$(( $(date +%s) - START_TIME ))
TOTAL_HR=$((TOTAL_SEC / 3600))
TOTAL_MIN=$(( (TOTAL_SEC % 3600) / 60 ))
JSON_COUNT=$(find "$JSON_DIR" -maxdepth 1 -name "*.json" -type f 2>/dev/null | wc -l | tr -d ' ')
PY_COUNT=$(find "$IMPL_DIR" -maxdepth 1 -name "*.py" -type f \
    ! -name "base.py" ! -name "__init__.py" ! -name "loader.py" ! -name "smoke_test.py" \
    2>/dev/null | wc -l | tr -d ' ')
SUMMARY_COUNT=$(find "$RESULTS_DIR" -name "summary.json" -type f 2>/dev/null | wc -l | tr -d ' ')
FAIL_COUNT=$(wc -l < "$FAILED_FILE" | tr -d ' ')

log "════════════════════════════════════════════════════════════"
log "  COMPLETE"
log "  Runtime:      ${TOTAL_HR}h ${TOTAL_MIN}m"
log "  Converted:    $CONVERTED"
log "  Killed:       $KILLED"
log "  Failed:       $FAILED (retryable with --retry-failed)"
log "  Processed:    ${#REMAINING[@]}"
log "  ──"
log "  JSONs:        $JSON_COUNT in unique_strategies/"
log "  Python:       $PY_COUNT in pipeline/strategies/"
log "  Summaries:    $SUMMARY_COUNT in strategy_results/"
log "  Failed list:  $FAILED_FILE ($FAIL_COUNT entries)"
log "  Killed list:  $KILLS_FILE"
log "════════════════════════════════════════════════════════════"

git push origin main 2>/dev/null && log "Pushed to origin." || log "⚠ Push failed — run 'git push' manually."
