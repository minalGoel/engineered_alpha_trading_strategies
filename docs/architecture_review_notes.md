# Architecture Review Notes

Adversarial review findings that shaped the architecture design. These are retained for context but the main architecture document (`live_trading_architecture.md`) incorporates all fixes seamlessly.

## Findings (by severity)

### CRITICAL

1. **Redis SPOF** — If Redis dies, IPC between strategy processes and OMS breaks. Open positions go unmanaged. Fix: Redis Sentinel from day 1. OMS degraded mode (operates via broker REST/WS directly). Mandatory server-side SL on every entry as crash protection.

2. **Broker failover useless for exits** — Can't close positions on one broker from another. Fix: Cross-broker hedging (place opposing positions on backup broker to neutralize exposure). Plus server-side SL as baseline protection.

3. **Ghost position window** — If a fill notification is lost (WS packet drop), local state shows FLAT but broker has a position. Strategy opens a SECOND position. Fix: post-order reconciliation timer (not just 3x/day). Pre-trade broker position check in risk gate.

### HIGH

4. **10 OPS rate limit math** — Concurrent modify loops + REST polling burns OPS too fast. Fix: no REST polling when WS is healthy. Priority token bucket (exits > modifications > new orders).

5. **Option chain staleness** — 5s polling means signal fires with up to 4.5s stale pricing data, potentially wrong strike. Fix: always on-demand API call at signal time. Background poll is for monitoring only.

6. **asyncio tick buffering** — Redis pub/sub drops ticks when strategy's event loop is blocked in on_bar_close. Fix: Redis Streams (persistent, consumer groups). on_bar_close runs in thread executor.

7. **DuckDB concurrent access** — DuckDB doesn't support multi-process writes. Fix: Position Tracker is sole DuckDB writer. Others read via Redis cache.

8. **threading.Lock in asyncio** — Blocks entire event loop. Fix: asyncio.Lock.

### MEDIUM

9. **EOD flatten too slow** — Sequential exits with 10 OPS = 50s for 5 positions. Fix: batch cancels + parallel exit placement + IOC escalation at 15:22.

10. **OMS modify race condition** — Order fills between status check and modify call. Fix: per-order asyncio.Lock + sequence numbers. Handle DhanOrderNotModifiable gracefully.

11. **Parquet crash data loss** — 30s buffer lost on crash (SEBI audit gap). Fix: tick WAL (mmap'd, survives SIGKILL) + 5s flush interval.

12. **Static holiday calendar** — Misses emergency closures. Fix: NSE API + broker market status check at startup. All-instrument-silence detector during session.
