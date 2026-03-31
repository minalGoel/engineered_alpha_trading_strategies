#!/usr/bin/env bash
# Sync EC2-produced artifacts into local pipeline paths.
#
# Default source: ./ec2_artifacts
# Copies:
#   ec2_artifacts/outputs                         -> ./outputs
#   ec2_artifacts/pipeline_artifacts/strategy_results (preferred)
#     OR ec2_artifacts/strategy_results           -> ./strategy_results
#   ec2_artifacts/*.log                           -> ./logs/ec2
#
# Usage:
#   ./scripts/sync_ec2_artifacts.sh
#   ./scripts/sync_ec2_artifacts.sh --source /path/to/ec2_artifacts
#   ./scripts/sync_ec2_artifacts.sh --dry-run

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_DIR="$ROOT_DIR/ec2_artifacts"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: --source requires a path argument" >&2
                exit 1
            fi
            SRC_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ! -d "$SRC_DIR" ]]; then
    echo "ERROR: Source directory not found: $SRC_DIR" >&2
    exit 1
fi

mkdir -p "$ROOT_DIR/outputs" "$ROOT_DIR/strategy_results" "$ROOT_DIR/logs/ec2"

sync_dir() {
    local from="$1"
    local to="$2"
    if [[ ! -d "$from" ]]; then
        return 0
    fi

    if command -v rsync >/dev/null 2>&1; then
        local rsync_flags=(-a --exclude ".DS_Store")
        if [[ "$DRY_RUN" -eq 1 ]]; then
            rsync_flags+=(-n -v)
        fi
        rsync "${rsync_flags[@]}" "$from"/ "$to"/
    else
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[dry-run] cp -R \"$from\"/. \"$to\"/"
        else
            cp -R "$from"/. "$to"/
        fi
    fi
}

STRATEGY_SRC=""
if [[ -d "$SRC_DIR/pipeline_artifacts/strategy_results" ]]; then
    STRATEGY_SRC="$SRC_DIR/pipeline_artifacts/strategy_results"
elif [[ -d "$SRC_DIR/strategy_results" ]]; then
    STRATEGY_SRC="$SRC_DIR/strategy_results"
fi

echo "Source artifacts: $SRC_DIR"
echo "Syncing outputs..."
sync_dir "$SRC_DIR/outputs" "$ROOT_DIR/outputs"

if [[ -n "$STRATEGY_SRC" ]]; then
    echo "Syncing strategy results from: $STRATEGY_SRC"
    sync_dir "$STRATEGY_SRC" "$ROOT_DIR/strategy_results"
else
    echo "No strategy_results directory found under $SRC_DIR (skipping)"
fi

echo "Syncing EC2 logs..."
if command -v rsync >/dev/null 2>&1; then
    rsync_flags=(-a --prune-empty-dirs --include "*/" --include "*.log" --include "*.out" --include "*.err" --exclude "*")
    if [[ "$DRY_RUN" -eq 1 ]]; then
        rsync_flags+=(-n -v)
    fi
    rsync "${rsync_flags[@]}" "$SRC_DIR"/ "$ROOT_DIR/logs/ec2"/
else
    while IFS= read -r -d '' f; do
        rel="${f#$SRC_DIR/}"
        dest="$ROOT_DIR/logs/ec2/$rel"
        mkdir -p "$(dirname "$dest")"
        if [[ "$DRY_RUN" -eq 1 ]]; then
            echo "[dry-run] cp \"$f\" \"$dest\""
        else
            cp "$f" "$dest"
        fi
    done < <(find "$SRC_DIR" -type f \( -name "*.log" -o -name "*.out" -o -name "*.err" \) -print0)
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry-run complete. No files were copied."
else
    OUT_COUNT=$(find "$ROOT_DIR/outputs" -type f | wc -l | tr -d ' ')
    STRAT_COUNT=$(find "$ROOT_DIR/strategy_results" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    LOG_COUNT=$(find "$ROOT_DIR/logs/ec2" -type f | wc -l | tr -d ' ')
    echo "Done."
    echo "outputs files: $OUT_COUNT"
    echo "strategy result dirs: $STRAT_COUNT"
    echo "ec2 logs files: $LOG_COUNT"
fi
