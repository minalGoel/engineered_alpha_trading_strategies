#!/usr/bin/env bash
# Serve the synced dashboard artifacts from ./outputs
#
# Usage:
#   ./scripts/start_dashboard.sh
#   ./scripts/start_dashboard.sh 8765

set -euo pipefail

PORT="${1:-8765}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/outputs"
DASH_FILE="$OUT_DIR/winning_strategies_dashboard.html"

if [[ ! -f "$DASH_FILE" ]]; then
    echo "Dashboard file not found: $DASH_FILE" >&2
    echo "Run ./scripts/sync_ec2_artifacts.sh first, or run pipeline to generate outputs." >&2
    exit 1
fi

echo "Serving dashboard from: $OUT_DIR"
echo "Open: http://127.0.0.1:${PORT}/winning_strategies_dashboard.html"
cd "$OUT_DIR"
python3 -m http.server "$PORT"
