#!/bin/bash
# =============================================================================
# EC2 Bootstrap + Pipeline Execution Script
# Target: c7a.16xlarge (64 vCPUs, 128GB RAM, AMD EPYC 4th gen)
# AMI: Ubuntu 24.04 LTS
# Volume: 200GB gp3
# Region: us-east-1
#
# Usage: Paste as EC2 User Data or run manually after SSH.
# =============================================================================

set -euo pipefail

# ── Configuration (edit these) ──────────────────────────────────────────────
REPO_URL="${REPO_URL:-https://github.com/YOUR_ORG/engineered_alpha_trading_strategies.git}"
GIT_BRANCH="${GIT_BRANCH:-claude/write-trading-pipeline-Yl57J}"
GIT_USER="${GIT_USER:-your-github-username}"
GIT_TOKEN="${GIT_TOKEN:-ghp_your_github_pat_here}"

# Pipeline settings
PARALLEL_STRATEGIES="${PARALLEL_STRATEGIES:-16}"
CORES_PER_STRATEGY="${CORES_PER_STRATEGY:-4}"

# Working directory
WORK_DIR="/home/ubuntu/pipeline"

# ── System setup ────────────────────────────────────────────────────────────
echo "[$(date)] Starting EC2 bootstrap..."

# Update system
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3.12-dev git git-lfs curl

# Install git LFS BEFORE any clone
git lfs install

# ── Clone repository ────────────────────────────────────────────────────────
echo "[$(date)] Cloning repository..."
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Configure git credentials
CLONE_URL="https://${GIT_USER}:${GIT_TOKEN}@${REPO_URL#https://}"
git clone --branch "$GIT_BRANCH" "$CLONE_URL" repo
cd repo

# Pull LFS files (data)
echo "[$(date)] Pulling LFS data files..."
git lfs pull

echo "[$(date)] Repository ready. Data files:"
ls -lh data/

# ── Python environment ─────────────────────────────────────────────────────
echo "[$(date)] Setting up Python environment..."
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

echo "[$(date)] Python environment ready."
python3 --version
pip list | grep -E "polars|numba|optuna|orjson"

# ── Run pipeline ───────────────────────────────────────────────────────────
echo "[$(date)] Starting pipeline..."
echo "[$(date)] parallel=$PARALLEL_STRATEGIES, cores_per=$CORES_PER_STRATEGY"

nohup python3 pipeline/run_all.py \
    --parallel-strategies "$PARALLEL_STRATEGIES" \
    --cores-per-strategy "$CORES_PER_STRATEGY" \
    > ~/pipeline.log 2>&1 &

PIPELINE_PID=$!
echo "[$(date)] Pipeline started with PID=$PIPELINE_PID"
echo "[$(date)] Monitor with: tail -f ~/pipeline.log"

# Wait for pipeline to complete (disable set -e so we can capture exit code)
set +e
wait $PIPELINE_PID
PIPELINE_EXIT=$?
set -e
echo "[$(date)] Pipeline exited with code=$PIPELINE_EXIT"

# ── Push results to git ────────────────────────────────────────────────────
echo "[$(date)] Pushing results to git..."
git lfs install
git add strategy_results/ outputs/
git commit -m "Pipeline results $(date -I)" || echo "Nothing to commit"

# Retry push with backoff
MAX_RETRIES=3
RETRY_DELAY=30
for i in $(seq 1 $MAX_RETRIES); do
    echo "[$(date)] Push attempt $i/$MAX_RETRIES..."
    if git push origin "$GIT_BRANCH"; then
        echo "[$(date)] Push successful!"
        break
    fi
    if [ "$i" -lt "$MAX_RETRIES" ]; then
        echo "[$(date)] Push failed, retrying in ${RETRY_DELAY}s..."
        sleep $RETRY_DELAY
        RETRY_DELAY=$((RETRY_DELAY * 2))
    else
        echo "[$(date)] Push FAILED after $MAX_RETRIES attempts"
        touch ~/PUSH_FAILED
        echo "[$(date)] Results saved on disk. Instance left running."
        exit 1
    fi
done

# ── Shutdown on success ────────────────────────────────────────────────────
echo "[$(date)] All done. Shutting down..."
sudo shutdown -h now
