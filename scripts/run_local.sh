#!/bin/bash
# Run a single strategy locally for testing
set -euo pipefail

STRATEGY="${1:-}"

if [ -z "$STRATEGY" ]; then
    echo "Usage: ./scripts/run_local.sh <strategy_name>"
    echo "       ./scripts/run_local.sh --dry-run"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_local.sh vwap_mean_reversion_basic_v1"
    echo "  ./scripts/run_local.sh --dry-run"
    exit 1
fi

if [ "$STRATEGY" = "--dry-run" ]; then
    python pipeline/run_all.py --dry-run
else
    python pipeline/run_all.py \
        --strategy "$STRATEGY" \
        --parallel-strategies 1 \
        --cores-per-strategy 2
fi
