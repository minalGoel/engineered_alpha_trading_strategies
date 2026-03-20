# Claude Code Automation — Headless CLI Execution

## Summary
Battle-tested patterns for running Claude Code CLI unattended in batch scripts. Every issue below was encountered and resolved through trial and error across 3 different automation scripts (equity audit, options conversion, one-by-one processing).

## The Working Invocation

```bash
claude -p "$(cat "$PROMPT_FILE")" \
    --dangerously-skip-permissions \
    --model opus \
    --max-turns 40 \
    --output-format text \
    < /dev/null \
    > "$LOGFILE" 2>&1
```

Every flag matters:

| Flag | Why |
|------|-----|
| `-p "$(cat ...)"` | Print mode — non-interactive. Prompt as argument. |
| `--dangerously-skip-permissions` | Skips all "Allow this tool?" prompts. Required for unattended. |
| `--model opus` | Explicit model selection. |
| `--max-turns 40` | Safety cap on tool-use rounds. Prevents infinite loops. |
| `--output-format text` | Writes plain text to stdout. **Only writes when session completes** — log file is empty during execution. |
| `< /dev/null` | **CRITICAL**: Fully detaches stdin from TTY. Without this, claude grabs TTY and causes "tty output" suspension errors when run under nohup. |
| `> "$LOGFILE" 2>&1` | Captures both stdout and stderr. |

## Prompt Delivery

### The Quoted Heredoc + sed Pattern (Final, Correct)

```bash
write_prompt() {
    local prompt_file="$1"
    local strat_name="$2"
    local strat_src="$3"

    cat > "$prompt_file" << 'ENDPROMPT'
    ... prompt text with __STRATEGY_NAME__ placeholders ...
ENDPROMPT

    sed -i '' "s|__STRATEGY_NAME__|${strat_name}|g" "$prompt_file"
    sed -i '' "s|__STRATEGY_SOURCE__|${strat_src}|g" "$prompt_file"
}
```

**Why quoted heredoc** (`<< 'ENDPROMPT'`): Prevents shell expansion inside the prompt. If a strategy name contains `$()` or backticks, an unquoted heredoc would execute them. The quoted heredoc writes the literal text, then `sed` injects the safe variable values after.

### Failed Approaches

| Approach | Problem |
|----------|---------|
| Unquoted heredoc with `${VAR}` | Shell expansion inside prompt. Strategy names with `$` or backticks = injection. |
| `echo "$PROMPT" \| claude -p -` | Pipe to stdin conflicts with `< /dev/null`. Can't do both. |
| Inline prompt as string variable | Shell quoting hell with nested quotes in JSON examples within the prompt. |
| `cat "$PROMPT_FILE"` piped to stdin | `--output-format text` requires `-p` flag, which takes the prompt as an argument, not stdin. |

## Temp File Management

```bash
# macOS-compatible mktemp (no extension after pattern)
PROMPT_FILE=$(mktemp /tmp/claude_one.XXXXXX)

# NOT this — fails on macOS:
PROMPT_FILE=$(mktemp /tmp/claude_audit_XXXXXX.txt)
```

macOS `mktemp` requires the `X` pattern at the very end with no suffix. Adding `.txt` causes "File exists" errors.

**Cleanup**:
- Delete temp file immediately after claude exits: `rm -f "$PROMPT_FILE"`
- Clean stale files at script startup: `rm -f /tmp/claude_one.*`
- Trap handler for interrupts: `trap 'rm -f /tmp/claude_one.*' INT TERM`

## The Venv Problem

Claude Code spawns its own shell subprocess. Activating the venv in the calling script (`source .venv/bin/activate`) does NOT affect claude's subprocess. The fix: instruct claude in the PROMPT to activate the venv.

```
IMPORTANT: Run "source .venv/bin/activate" before any Python command.
```

This must be the FIRST line of every prompt that involves Python execution.

## Logging Architecture

### The Problem
`--output-format text` only writes when the entire session completes. The log file is empty during execution. This means:
- You can't `tail -f` the log to see progress
- You can't tell if claude is working or hung
- A 20-minute silent period is normal

### The Solution: Heartbeat Process

```bash
start_heartbeat() {
    local strat_name="$1"
    local parent_pid=$$
    (
        while true; do
            sleep 60
            if pgrep -P "$parent_pid" -f "claude" > /dev/null 2>&1; then
                log "  [HEARTBEAT] $strat_name — claude running"
            else
                break
            fi
        done
    ) &
    heartbeat_pid=$!
}
```

- Uses `pgrep -P "$parent_pid"` to check only children of THIS script (not other claude processes)
- Prints every 60 seconds to the runner log
- Killed via `stop_heartbeat()` after claude exits

### Monitoring Commands

```bash
# DON'T use tail -f (causes TTY suspension under nohup)
# DO use watch or periodic cat:
watch -n 10 'tail -20 logs/one_by_one_runner.log'

# Or on macOS without watch:
while true; do clear; tail -20 logs/one_by_one_runner.log; sleep 10; done

# Quick status counts:
grep -c "CONVERTED\|KILLED\|FAILED" logs/one_by_one_runner.log

# Read a specific strategy's full session:
cat logs/per_strategy/momentum_v1.log
```

## Checkpoint and Resume

### Tracker Pattern
```bash
TRACKER="logs/completed_strategies.txt"
touch "$TRACKER"

# Skip already completed:
for name in "${QUALIFIED[@]}"; do
    grep -qxF "$name" "$TRACKER" 2>/dev/null || REMAINING+=("$name")
done

# After each strategy completes:
echo "$name" >> "$TRACKER"
```

### Separate Failed File
```bash
FAILED_FILE="logs/failed_strategies.txt"

# Failures go to BOTH tracker (skip on normal run) and failed file (retry later)
echo "$name" >> "$TRACKER"
echo "$name" >> "$FAILED_FILE"

# Retry with:
./scripts/auto_one_by_one.sh --retry-failed
```

### Git Commits as Checkpoints
```bash
# Every N strategies:
if (( DONE_COUNT % 5 == 0 )); then
    git add unique_strategies/ pipeline/strategies/ strategy_results/ outputs/
    git diff --cached --quiet || git commit -m "Auto [$DONE_COUNT/${TOTAL}]" --quiet
fi
```

## Trap for Cleanup

```bash
cleanup() {
    if [[ -n "$heartbeat_pid" ]]; then
        kill "$heartbeat_pid" 2>/dev/null
        wait "$heartbeat_pid" 2>/dev/null
    fi
    rm -f /tmp/claude_one.*
    log "INTERRUPTED — progress saved"
    exit 130
}
trap cleanup INT TERM
```

Without this, killing the script mid-run leaves orphaned heartbeat processes and temp files.

## Preflight Checks

Run before the main loop to fail fast:

```bash
command -v claude &>/dev/null || { log "FATAL: claude not found"; exit 1; }
[[ -f "$IMPL_DIR/base.py" ]] || { log "FATAL: base.py missing"; exit 1; }
[[ -f "$IMPL_DIR/CONVENTIONS.md" ]] || { log "FATAL: CONVENTIONS.md missing"; exit 1; }
[[ -f "outputs/strategy_retriage.json" ]] || { log "FATAL: retriage.json missing"; exit 1; }
find "5second_data" -name "*.parquet" -type f | head -1 | grep -q . || { log "FATAL: no parquet"; exit 1; }
```

## `set -uo pipefail` Without `-e`

The script uses `set -uo pipefail` deliberately WITHOUT `-e`. Adding `-e` would abort the entire run when any command returns non-zero — which includes `grep` finding no match (exit 1), `git diff --cached --quiet` when there are staged changes (exit 1), and other innocent failures. The main loop handles errors per-strategy.

Document this intentional omission so a future reader doesn't "fix" it.

## Decisions Made
- One fresh session per strategy (no batching) eliminates context drift
- Opus for judgment-heavy tasks, Sonnet for mechanical tasks
- `--max-turns 40` is enough for convert + implement + backtest + verify per strategy
- Log every strategy to its own file in `logs/per_strategy/` for full audit trail
- Commit every 5 strategies (not every 1 — too many commits; not every 20 — too much loss on crash)

## Pitfalls & Anti-patterns
- **`tail -f` on nohup output**: Can cause TTY suspension. Use `watch` or periodic `cat`.
- **Nested Claude sessions**: Running `claude` inside an existing `claude` session fails with "cannot be launched inside another Claude Code session." Unset the `CLAUDECODE` environment variable to bypass.
- **`grep -oP` on macOS**: Not supported. Use `sed -n 's/.../\1/p'` instead.
- **Expecting streaming output**: `--output-format text` buffers everything until session end. The log file is empty during execution. This is not a bug.
- **ARG_MAX risk**: `$(cat "$PROMPT_FILE")` puts the prompt in argv. Current prompts are ~3KB, well under macOS ARG_MAX of ~1MB. If prompts grow to include full strategy JSON content, switch to a different delivery method.

## Corrections Log
- ~~Used `mktemp /tmp/claude_audit_XXXXXX.txt`~~ → macOS requires no suffix after X pattern: `mktemp /tmp/claude_one.XXXXXX`
- ~~Prompt passed via unquoted heredoc with `${VAR}` expansion~~ → Quoted heredoc + sed injection prevents shell injection
- ~~Heartbeat used `pgrep -f "claude.*-p"` (too broad)~~ → `pgrep -P "$parent_pid" -f "claude"` scopes to children of this script only
- ~~Failed strategies marked complete and lost forever~~ → Separate `failed_strategies.txt` + `--retry-failed` flag
- ~~No cleanup on interrupt~~ → Added `trap cleanup INT TERM`
- ~~KILLS_FILE corrupted by half-writes~~ → JSON validation on startup, reset to `[]` if corrupted
