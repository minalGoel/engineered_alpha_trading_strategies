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

# ── Clone repository (with retry) ──────────────────────────────────────────
echo "[$(date)] Cloning repository..."
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

CLONE_URL="https://${GIT_USER}:${GIT_TOKEN}@${REPO_URL#https://}"
CLONE_OK=false
for attempt in 1 2 3 4; do
    echo "[$(date)] Clone attempt $attempt/4..."
    if git clone --branch "$GIT_BRANCH" "$CLONE_URL" repo; then
        CLONE_OK=true
        break
    fi
    delay=$((2 ** attempt))
    echo "[$(date)] Clone failed, retrying in ${delay}s..."
    rm -rf repo  # clean partial clone
    sleep "$delay"
done
if [ "$CLONE_OK" = false ]; then
    echo "[$(date)] FATAL: Clone failed after 4 attempts"
    exit 1
fi

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

# ── Run pipeline + push inside nohup (survives SSH disconnect) ─────────────
# The entire run-and-push sequence is inside nohup so that if the user
# connected via SSH and the session drops, the push still happens.
echo "[$(date)] Starting pipeline (nohup-protected)..."
echo "[$(date)] parallel=$PARALLEL_STRATEGIES, cores_per=$CORES_PER_STRATEGY"

nohup bash -c '
set -eo pipefail
cd '"$WORK_DIR/repo"'
source .venv/bin/activate

echo "[$(date)] Pipeline starting..."
python3 pipeline/run_all.py \
    --parallel-strategies '"$PARALLEL_STRATEGIES"' \
    --cores-per-strategy '"$CORES_PER_STRATEGY"'
PIPELINE_EXIT=$?
echo "[$(date)] Pipeline exited with code=$PIPELINE_EXIT"

# ── Push results to git ──────────────────────────────────────────────────
echo "[$(date)] Pushing results to git..."
git lfs install
git add strategy_results/ outputs/
git commit -m "Pipeline results $(date -I)" || echo "Nothing to commit"

MAX_RETRIES=4
RETRY_DELAY=2
for i in $(seq 1 $MAX_RETRIES); do
    echo "[$(date)] Push attempt $i/$MAX_RETRIES..."
    if git push origin '"$GIT_BRANCH"'; then
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

echo "[$(date)] All done. Shutting down..."
sudo shutdown -h now
' > ~/pipeline.log 2>&1 &

PIPELINE_PID=$!
echo "[$(date)] Pipeline+push started with PID=$PIPELINE_PID"
echo "[$(date)] Monitor with: tail -f ~/pipeline.log"
echo "[$(date)] Safe to disconnect SSH — nohup protects the entire flow."
