# Performance Engineering

## Summary
Specific library choices, parallelism patterns, and compute sizing that make the difference between a 5-hour pipeline run and a 3-week one. Every choice was motivated by measured performance on the actual workload.

## Library Stack

| Layer | Library | Why Not the Alternative |
|-------|---------|----------------------|
| DataFrames | **Polars** | pandas is 10-50x slower for filter/groupby/join. Arrow columnar format is dramatically more memory-efficient. |
| Position State Machine | **Numba @njit** | The bar-by-bar loop tracking FLAT→LONG→SHORT can't be vectorized. Numba compiles to native code: 281K bars in <5ms vs 500ms in pure Python. 100x difference. |
| JSON I/O | **orjson** | 10x faster than stdlib json. But beware: rejects NaN and Infinity (sanitize before writing). |
| Optimization | **Optuna** (TPESampler) | Bayesian search, not random. No pruner enabled — let all trials complete. |
| Parallelism | **joblib** | Parallel + delayed. Two-level: outer (strategy) + inner (instrument). |
| Tabular output | **Parquet** | 5-10x faster reads than CSV, 3-5x smaller files. |

## The Numba State Machine

This is the single hottest function. It runs billions of times across all strategies × instruments × Optuna trials.

**Requirements**:
- Takes numpy arrays, returns numpy arrays
- All inputs must be float64, int8, or int32 (no Python objects inside @njit)
- Trailing stops are stateful — track highest high / lowest low since entry, update each bar — this MUST be inside the Numba loop
- Set `NUMBA_NUM_THREADS=1` when using parallel workers (prevents Numba spawning internal threads per joblib worker)

**Key guards inside the loop**:
- `stop_price == 0.0` → skip stop check (not "compare against 0")
- `atr > 0` before any ATR-based stop calculation
- Force-close open position on last bar of data
- Day boundary detection via `day_id` array

## Parallelism Architecture

**Outer pool**: Strategy-level, loky backend
**Inner pool**: Instrument-level, threading backend

**CRITICAL**: Both pools cannot use loky. loky = process-based. Process inside process = deadlock on some systems.

### EC2 (c7a.16xlarge: 64 vCPUs, 128GB RAM)
- 8 parallel strategies × 4 cores per strategy = 32 cores active
- Originally planned 16×4=64, but 16 workers × ~8GB peak = 128GB, dangerously close to OOM
- 8×4=32 gives 64GB peak usage with 64GB headroom

### Laptop (MacBook Air M4)
- 1 parallel strategy × 2 cores = 2 cores active
- For single-strategy testing only

## Memory Management

Each worker loads stock data independently. With Polars lazy scanning, not all data materializes simultaneously. But during indicator computation (when rolling windows force materialization), peak memory per worker = ~6-8GB.

**VIX and index data**: Small (~1MB each). Just reload in each worker. Don't bother with shared memory — IPC coordination overhead isn't worth it for files this small.

**Forward-fill NaN**: Use `pl.col(c).forward_fill()` in Polars BEFORE converting to numpy. A Python-loop forward-fill adds ~33 minutes to the full run.

## Data Format

Convert all CSV to Parquet ONCE at pipeline start (Phase 1). All subsequent reads use Parquet. This is a one-time cost that pays for itself thousands of times over.

## EC2 Instance Selection

| Instance | vCPUs | RAM | Cost/hr | Use Case |
|----------|-------|-----|---------|----------|
| c7a.16xlarge | 64 | 128GB | ~$2.42 | Full pipeline (8 parallel strategies) |
| c7a.8xlarge | 32 | 64GB | ~$1.21 | Budget option (4 parallel) |
| MacBook Air M4 | 8 | 16GB | $0 | Single-strategy testing only |

Storage: 200GB gp3 root volume. 358 strategies × signals.parquet + trades.parquet ≈ 3.6-21.5GB total output.

## Decisions Made
- Polars over pandas — non-negotiable for this data volume
- Numba for inner loop — the 100x speedup is the difference between hours and weeks
- Parquet over CSV — one-time conversion cost, permanent speedup
- 8 parallel (not 16) on 128GB instance — safer memory margin
- No Optuna pruning — Bayesian sampler needs the full evaluation landscape from all 100 trials

## Pitfalls & Anti-patterns
- **pandas in hot path**: Any pandas operation in the per-bar or per-trial loop kills performance
- **Python-loop forward-fill**: O(n) Python vs vectorized Polars. 33-minute difference on full run.
- **Both joblib pools as loky**: Deadlock. Outer=loky, inner=threading.
- **Numba compiling in parallel**: First call to @njit compiles. If 16 workers all compile simultaneously → contention. Warm up the Numba function once before spawning workers.
- **Loading all stocks into memory per worker**: With 16 workers × full dataset = 800GB virtual memory. Use lazy loading or reduce parallelism.

## Corrections Log
- ~~16 parallel strategies on 128GB~~ → Reduced to 8 to avoid OOM during peak indicator computation
- ~~Pre-load VIX/index into shared memory~~ → Just reload per worker. The files are 1MB each, shared memory adds complexity for no gain.
- ~~10-minute timeout per strategy optimization~~ → 30-minute safety net. 10 minutes was a laptop constraint.
