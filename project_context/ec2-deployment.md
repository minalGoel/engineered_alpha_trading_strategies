# EC2 Deployment

## Summary
Unattended pipeline execution on EC2 with git-based I/O. Instance runs the pipeline, commits results, pushes, and self-terminates. Designed for the equity pipeline's ~6-7 hour full run across 358 strategies × 206 stocks.

## Instance Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Instance | c7a.16xlarge | 64 vCPUs, 128GB RAM. Compute-optimized AMD. |
| Region | us-east-1 | Cheapest spot pricing |
| Storage | 200GB gp3 root | Data + results + git history |
| AMI | Ubuntu 24.04 LTS | Standard, well-supported |
| Cost | ~$2.42/hr on-demand, ~$12 total | 6-7 hours runtime |

### Why c7a.16xlarge
- 64 vCPUs supports 8 parallel strategy workers × 4 cores each = 32 active cores
- 128GB RAM handles 8 workers × 8GB peak = 64GB with 64GB headroom
- The 32xlarge variant (128 vCPUs, 256GB) is overkill — the pipeline is memory-bound at 16 parallel workers, not CPU-bound

## Bootstrap Script (ec2_run.sh)

```bash
#!/bin/bash
set -euo pipefail

# Timezone
export TZ=Asia/Kolkata
sudo timedatectl set-timezone Asia/Kolkata
sudo apt-get update && sudo apt-get install -y tzdata

# Python
sudo apt-get install -y python3 python3-pip python3-venv git-lfs
python3 -m venv /opt/venv
source /opt/venv/bin/activate

# Clone repo with LFS
git lfs install
git clone https://${GH_TOKEN}@github.com/${REPO}.git /opt/pipeline
cd /opt/pipeline

# Install dependencies
pip install -r requirements.txt

# Set Numba threads (CRITICAL for parallel workers)
export NUMBA_NUM_THREADS=1

# Run pipeline
python -m pipeline.run_all \
    --workers 8 \
    --phase 1 \
    2>&1 | tee logs/ec2_run.log

# Commit results
git add strategy_results/ outputs/ logs/
git commit -m "EC2 pipeline run: $(date +%Y-%m-%d)"
git push origin main

# Self-terminate
sudo shutdown -h now
```

### Key Details
- `NUMBA_NUM_THREADS=1` MUST be set before importing numba. Otherwise thread contention kills performance.
- `git push` MUST be inside nohup or the main script body. A prior bug had it after the pipeline but outside nohup — if SSH disconnected, push never happened.
- `shutdown -h now` terminates the instance. Use `--no-wall` to suppress broadcast warning.
- Git LFS required for parquet files. `.gitattributes` must track `*.parquet` before the first push.

## Data Transfer

### Option A: Git LFS (Used)
- `.gitattributes`: `data/*.parquet filter=lfs diff=lfs merge=lfs`
- Parquet files stored in LFS, cloned with `git lfs install && git clone`
- Works for <5GB total data. Above that, consider S3.

### Option B: S3 (Not Used, For Scale)
- Upload parquet files to S3 bucket
- Bootstrap script downloads with `aws s3 sync`
- Requires IAM role on the EC2 instance

## Monitoring

From your laptop:
```bash
# SSH in and tail the log
ssh -i key.pem ubuntu@<ip> 'tail -f /opt/pipeline/logs/ec2_run.log'

# Quick check: how many strategies done?
ssh ubuntu@<ip> 'ls /opt/pipeline/strategy_results/ | wc -l'
```

If SSH disconnects, the pipeline continues (running under nohup or as the main process of a user-data script).

## Self-Termination

The script ends with `sudo shutdown -h now`. This stops the instance (not terminates — stopped instances don't incur compute charges but EBS volumes still do). For full cleanup:
- Use `aws ec2 terminate-instances` instead, or
- Set the instance to "terminate on stop" before launching

## Git Push Failure Handling

```bash
git push origin main 2>/dev/null && echo "Pushed." || {
    echo "Push failed. Results saved locally."
    # Retry with exponential backoff
    for i in 1 2 3; do
        sleep $((i * 30))
        git push origin main && break
    done
}
```

If push fails after retries, results are still on the EBS volume. SSH in to retrieve.

## Decisions Made
- Git-based I/O (clone, commit, push) over S3 — simpler, full provenance via git history
- 8 parallel workers (not 16) — memory safety margin
- Self-termination — prevents forgotten instances running for days
- Ubuntu 24.04 over Amazon Linux — more familiar, better Python support

## Pitfalls & Anti-patterns
- **Forgetting `NUMBA_NUM_THREADS=1`**: Without this, 8 workers × N Numba threads = thread contention. Performance drops 3-5x.
- **Git push outside nohup/main script**: If SSH disconnects before push, all results are lost.
- **Not setting timezone**: EC2 defaults to UTC. All IST-based pipeline logic fails silently.
- **Using t-series instances**: T-series (t3, t4g) have CPU burst credits. They throttle to baseline after credits deplete — pipeline slows 10x. Always use c-series (compute-optimized).
- **Forgetting git lfs install before clone**: Parquet files download as pointer files (tiny text files), not actual data. Pipeline fails silently with empty DataFrames.

## Corrections Log
- ~~16 parallel workers on 128GB~~ → 8 workers for memory safety
- ~~git push was after pipeline but outside nohup~~ → Moved inside main script body
- ~~No timezone export~~ → Added `export TZ=Asia/Kolkata` and `timedatectl`
