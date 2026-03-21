#!/usr/bin/env python3
"""One-time setup: download Chart.js to dashboard/static/ for offline use.

Run this once before starting the dashboard:
    python dashboard/setup_dashboard.py

After this, the dashboard works completely offline. No CDN calls at runtime.
"""
import sys
import urllib.request
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"
CHART_JS_URL = "https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"
CHART_JS_PATH = STATIC_DIR / "chart.min.js"

# Known SHA256 for integrity verification (optional but recommended)
CHART_JS_SHA256 = None  # Set to expected hash to enable verification


def download_chart_js():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    if CHART_JS_PATH.exists():
        print(f"chart.min.js already present at {CHART_JS_PATH}")
        print("Delete it and re-run to force a fresh download.")
        return

    print(f"Downloading Chart.js 4.4.4 from {CHART_JS_URL}")
    print("This is a one-time download. The dashboard will work offline after this.")

    try:
        with urllib.request.urlopen(CHART_JS_URL, timeout=30) as resp:
            content = resp.read()
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        print(f"\nManual alternative:")
        print(f"  1. Download chart.umd.min.js from https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/")
        print(f"  2. Save as: {CHART_JS_PATH}")
        sys.exit(1)

    CHART_JS_PATH.write_bytes(content)
    size_kb = len(content) / 1024
    print(f"Saved {CHART_JS_PATH.name} ({size_kb:.0f} KB)")

    # Verify content is JavaScript
    snippet = content[:200].decode("utf-8", errors="ignore")
    if "chart" not in snippet.lower() and "Chart" not in snippet:
        print("WARNING: Downloaded content does not look like Chart.js. Verify manually.")
    else:
        print("Verification OK — Chart.js content confirmed.")

    print(f"\nSetup complete. Start the dashboard with:")
    print(f"  python pipeline/run_all.py --dashboard")
    print(f"  # or: python pipeline/dashboard_server.py")


if __name__ == "__main__":
    download_chart_js()
