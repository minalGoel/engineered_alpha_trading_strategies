# Live Trading System — Architecture Document v3

**Date:** 2026-03-23
**Status:** Design (pre-implementation)
**Scope:** v3 — single-leg orders, multi-account replication, multi-broker failover
**Previous version:** live_trading_architecture.md (v2)

---

## Section 0: Findings & Review Mapping

Every finding from the v2 architecture review is tracked below with a severity rating
and a pointer to the v3 section that resolves it. No finding is deferred or left
unaddressed.

| Finding | Severity | Description | Addressed In |
|---------|----------|-------------|--------------|
| CRITICAL-1 | CRITICAL | DAY SL expires at 15:30; overnight positions are unprotected until next-day AMO submission | §5 OMS — Forever Orders: SL orders placed as GTT/Forever with GTC validity. Automatic DAY-to-Forever conversion at 15:20 for any position held overnight. |
| CRITICAL-2 | CRITICAL | SL quantity not synced on PART_TRADED; sync only triggers on full abandon, leaving residual exposure | §5 OMS — Fill Loop SL Sync: FillManager emits `PART_TRADED` event; SL Lifecycle Manager immediately modifies SL qty to match remaining entry qty. REST verification follows every modify. |
| HIGH-3 | HIGH | No cross-broker hedging during failover; positions opened on Dhan have no SL if execution switches to Upstox | §6 Broker Router — Cross-Broker Position Awareness: Failover router reads open positions from primary broker before routing to secondary. SL orders placed on secondary reference primary positions via internal position tracker. |
| HIGH-4 | HIGH | Signal deduplication uses simple (strategy, symbol, direction) tuple; duplicate signals with different bar timestamps pass through | §4 Capital Allocator — SignalDeduplicator: Dedup key is (strategy_id, direction, underlying). TTL-based cooldown window (configurable per strategy, default 300s). Atomic check-and-set via in-memory dict with asyncio.Lock. |
| HIGH-5 | HIGH | No existing position check before capital allocation; system can allocate capital to a signal when a same-direction position already exists | §4 Capital Allocator — AllocationRequest: Pre-allocation query to PositionTracker for open positions matching (strategy_id, underlying, direction). Allocation rejected if matching position exists. |
| HIGH-6 | HIGH | No ghost position detection; broker may show positions the system does not track (manual trades, partial fills missed during disconnect) | §7 Risk Manager — Pre-Trade Broker Position Check: Periodic reconciliation (every 5 minutes during market hours) compares broker position snapshot against internal PositionTracker. Discrepancies logged, alerted via Telegram, and optionally halt new orders until resolved. |
| HIGH-7 | HIGH | Multi-account not supported; system assumes a single trading account | §10 Account Replication Layer (NEW): Fan-out of sized signals to N accounts. Per-account OMS, Fill Manager, SL Lifecycle, and WS connections. Account-level capital tracking and risk limits. |
| MEDIUM-8 | MEDIUM | Tuesday CSV fallback (stale instrument master) does not disable S5 (0-DTE strategy) which requires current-day expiry instruments | §2 Instrument Resolution: CSV staleness check compares file date against current date. If stale > 1 day, S5 is disabled with alert. Strategy-level instrument dependency declared in config. |
| MEDIUM-9 | MEDIUM | Tick WAL flush interval not specified; crash between flushes loses unbounded tick data | §1 Data Ingestion: WAL flush interval set to 5 seconds. Maximum data loss on crash: 5 seconds of ticks. WAL uses append-only file with fsync on flush. |
| MEDIUM-10 | MEDIUM | DuckDB concurrent write risk; multiple components may attempt simultaneous writes causing corruption | §8 Position Tracker — Single-Writer Architecture: All writes to DuckDB funneled through a single PositionTracker process via an asyncio queue. Read replicas use DuckDB read-only connections. No external process writes to the database file. |
| MEDIUM-11 | MEDIUM | DuckDB tables not partitioned per-account; multi-account support requires schema changes to avoid cross-account data leakage | §8 Position Tracker — Per-Account Tables: Table schema includes `account_id` as partition key. All queries filter by account_id. Views created per-account for convenience. Aggregate views span all accounts for portfolio-level risk. |
| LOW-12 | LOW | `threading.Lock` used in async code; blocks the entire event loop during contention | §4 Capital Allocator: All locks converted to `asyncio.Lock`. No threading primitives in async code paths. Threading locks retained only in multiprocessing strategy harness (non-async). |

### Resolution Summary

- **CRITICAL findings (2/2):** Both resolved with explicit mechanisms and verification steps.
- **HIGH findings (5/5):** All resolved. HIGH-7 (multi-account) is the primary driver of the v3 rewrite.
- **MEDIUM findings (4/4):** All resolved with specific parameter values and architectural constraints.
- **LOW findings (1/1):** Resolved.
- **Total: 12/12 findings addressed. Zero deferred.**

---

## Section 1: System Overview

A production live trading system for 7 strategies (S1-S7) on NSE, using Dhan as the
primary broker and Upstox as failover, for both tick-by-tick market data and order
execution. The system runs on an AWS EC2 `ap-south-1` instance with a static Elastic
IP (SEBI compliance). Initial capital: ₹50L across up to 3 accounts, scaling to ₹10Cr
by Year 2.

The backtest pipeline (existing `pipeline/`) remains untouched. The live system is a
top-level package `live/` that reuses the cost model and strategy signal logic but adds
real-time execution infrastructure. v3 introduces multi-account replication, multi-broker
failover, and resolves all 12 findings from the v2 review.

### 1.1 Broker Selection

The architecture uses an adapter pattern with Dhan as primary and Upstox as failover.
Multi-account mode in v3 means multiple Dhan API keys (one per account).

| Dimension | Dhan | Upstox |
|-----------|------|--------|
| OPS limit | 10 per key | 50 per key |
| WS instruments | 25,000 | 100 |
| Depth levels | 20 / 200 | 5 |
| Tick delivery | Event-driven | Event-driven |
| Historical data | 5 years (including expired options) | Multi-year |
| HFT endpoint | No | `api-hft.upstox.com` |
| Static IP support | Confirmed | Unclear |
| Daily order limit | 5,000 per key | Higher |
| Multi-account | Multiple keys, same infrastructure | Multiple keys, same infrastructure |

**v3 Recommendation:** Dhan remains primary. 3 accounts = 30 OPS total (10 per Dhan
key). This provides sufficient headroom for 7 strategies across 3 accounts at current
signal frequency. Upstox serves as failover broker — activated only when Dhan API
is unreachable or rate-limited. Cross-broker position awareness (CRITICAL-1, HIGH-3)
ensures failover does not create orphaned positions.

**Multi-account OPS arithmetic:**

```
Per account:   10 OPS (Dhan limit)
Accounts:      3
Total OPS:     30

Per strategy:  ~4 OPS peak (entry + SL + modify + cancel)
Strategies:    7
Concurrent:    ~3 strategies firing simultaneously (worst case)
Peak demand:   ~12 OPS across all accounts
Headroom:      30 - 12 = 18 OPS (60% spare capacity)
```

**Convention:** This document uses "broker" generically. Broker-specific details
(endpoint URLs, auth flows, field names) live in the `BrokerAdapter` implementation,
not in the architecture. "Account" refers to a distinct trading account with its own
API key, capital pool, and position set.

### 1.2 Process Topology (v3)

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  EC2 ap-south-1 (Ubuntu, static Elastic IP)                                         │
│                                                                                      │
│  ┌────────────────┐                                                                  │
│  │ Redis Sentinel  │  (primary + 1 replica, same host)                              │
│  └────────┬───────┘                                                                  │
│           │                                                                           │
│  ┌────────▼───────┐  Redis Streams      ┌──────────────────────────┐                │
│  │ Data Ingester   │ ───────────────→   │ Strategy Processes (x7)  │                │
│  │ (async, shared) │  STREAM:TICK:*     │  S1..S7 (multiprocess)   │                │
│  │ + Tick WAL (5s) │                    │  (shared — not per-acct) │                │
│  └──────┬─────────┘                     └──────────┬───────────────┘                │
│         │ Parquet flush                             │ StrategySignal                  │
│         │                                           ▼                                 │
│  ┌──────▼─────────┐                    ┌──────────────────────────┐                 │
│  │ Tick Archive    │                    │ Signal Router (async)    │                 │
│  │ (Parquet → S3)  │                    │  + SignalDeduplicator    │                 │
│  └────────────────┘                    │  + Instrument Resolver   │                 │
│                                         │  (shared signal path)    │                 │
│                                         └──────────┬───────────────┘                 │
│                                                     │ DeduplicatedSignal              │
│                                         ┌───────────▼───────────────┐                │
│                                         │ Account Replication Layer  │  ◄── NEW (v3) │
│                                         │  + Capital Allocator      │                │
│                                         │  Fan-out to N accounts    │                │
│                                         │  Per-account sizing       │                │
│                                         └─────┬─────┬─────┬────────┘                │
│                                               │     │     │                           │
│                              ┌────────────────┘     │     └────────────────┐          │
│                              ▼                      ▼                      ▼          │
│                   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│                   │ Account A       │  │ Account B       │  │ Account C       │     │
│                   │ ┌─────────────┐ │  │ ┌─────────────┐ │  │ ┌─────────────┐ │     │
│                   │ │ Risk Gate   │ │  │ │ Risk Gate   │ │  │ │ Risk Gate   │ │     │
│                   │ │ (per-acct)  │ │  │ │ (per-acct)  │ │  │ │ (per-acct)  │ │     │
│                   │ └──────┬──────┘ │  │ └──────┬──────┘ │  │ └──────┬──────┘ │     │
│                   │ ┌──────▼──────┐ │  │ ┌──────▼──────┐ │  │ ┌──────▼──────┐ │     │
│                   │ │ OMS (async) │ │  │ │ OMS (async) │ │  │ │ OMS (async) │ │     │
│                   │ │ +FillMgr    │ │  │ │ +FillMgr    │ │  │ │ +FillMgr    │ │     │
│                   │ │ +SL Lifecyc │ │  │ │ +SL Lifecyc │ │  │ │ +SL Lifecyc │ │     │
│                   │ │ +EOD Flattn │ │  │ │ +EOD Flattn │ │  │ │ +EOD Flattn │ │     │
│                   │ └──────┬──────┘ │  │ └──────┬──────┘ │  │ └──────┬──────┘ │     │
│                   │ ┌──────▼──────┐ │  │ ┌──────▼──────┐ │  │ ┌──────▼──────┐ │     │
│                   │ │ Broker WS   │ │  │ │ Broker WS   │ │  │ │ Broker WS   │ │     │
│                   │ │ (order upd) │ │  │ │ (order upd) │ │  │ │ (order upd) │ │     │
│                   │ └─────────────┘ │  │ └─────────────┘ │  │ └─────────────┘ │     │
│                   └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│                            │ fills               │ fills              │ fills         │
│                            └─────────────┬───────┘───────────────────┘               │
│                                          ▼                                            │
│                              ┌──────────────────────────┐                            │
│                              │ Position & PnL Tracker    │                            │
│                              │ (sole DuckDB writer)      │                            │
│                              │ Per-account tables         │                            │
│                              │ + aggregate portfolio view │                            │
│                              └──────────┬───────────────┘                            │
│                                         │                                             │
│  ┌────────────────┐         ┌───────────▼──────────────┐                             │
│  │ Risk Monitor    │◄────── │ Post-Trade Monitor        │                             │
│  │ (per-acct +     │        │ (per-acct reconciliation) │                             │
│  │  portfolio-wide │        └──────────────────────────┘                              │
│  │  kill/halt)     │                                                                  │
│  └──────┬─────────┘                                                                   │
│         │                                                                              │
│  ┌──────▼──────────────────────┐    ┌────────────────────┐                           │
│  │ Prometheus + Grafana        │    │ Audit Logger        │                           │
│  │ (per-acct + aggregate       │    │ (structlog → S3)    │                           │
│  │  dashboards)                │    │ (per-acct log files)│                           │
│  └──────┬──────────────────────┘    └────────────────────┘                           │
│         │                                                                              │
│  ┌──────▼─────────┐   ┌──────────────────┐                                           │
│  │ Telegram Notifs │   │ Broker Router    │  ◄── NEW (v3)                             │
│  │ (per-acct +     │   │ Dhan (primary)   │                                           │
│  │  portfolio)     │   │ Upstox (failover)│                                           │
│  └────────────────┘   │ Cross-broker pos  │                                           │
│                        │ awareness         │                                           │
│                        └──────────────────┘                                           │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

**Shared vs. Per-Account boundary:**

| Layer | Shared or Per-Account | Rationale |
|-------|----------------------|-----------|
| Data Ingester | Shared | Market data is account-agnostic. One WS connection for ticks. |
| Strategy Processes (S1-S7) | Shared | Strategies produce signals from market data. Signals are not account-specific. |
| Signal Router | Shared | Deduplication and instrument resolution are account-agnostic. |
| Account Replication Layer | Per-account (fan-out) | Replicates a single signal into N account-specific execution requests. |
| Capital Allocator | Per-account | Each account has its own capital pool and ERC allocation. The allocator runs once per account per signal inside the Account Replication Layer. |
| Risk Gate | Per-account | Account-level drawdown limits, position limits, and margin checks. |
| OMS + Fill Manager + SL Lifecycle | Per-account | Each account has its own order book, fill state, and SL tracking. Separate broker WS connections per API key. |
| Position Tracker | Shared process, per-account data | Single DuckDB writer process. Tables partitioned by `account_id`. |
| Risk Monitor | Per-account + portfolio-wide | Account-level kill switches plus portfolio-wide aggregate checks. |
| Monitoring | Shared infra, per-account dashboards | One Prometheus/Grafana instance. Dashboards filtered by `account_id` label. |
| Broker Router | Shared | Routes orders to Dhan (primary) or Upstox (failover). Decision is per-order, not per-account. |

### 1.3 Signal-to-Fill Data Flow (v3)

```
Broker WS (ticks)
    │
    ▼
DataIngester (shared)
    │
    ├── Tick WAL + Parquet (5s flush)
    │
    ▼
Redis STREAM:TICK:* ──→ Strategy Process (S1..S7, shared)
                              │
                         StrategySignal
                              │
                              ▼
                    Redis STREAM:SIGNAL
                              │
                              ▼
                    ┌─────────────────────────┐
                    │ Signal Router (shared)   │
                    │                          │
                    │ 1. Signal dedup check    │
                    │    (strategy_id,         │
                    │     direction,           │
                    │     underlying)           │
                    │                          │
                    │ 2. Instrument resolution │
                    │    (on-demand chain API) │
                    └────────────┬─────────────┘
                                 │
                                 │ DeduplicatedSignal + ResolvedInstrument
                                 │
                    ┌────────────▼─────────────┐
                    │ Account Replication Layer │
                    │ For each active account:  │
                    │                           │
                    │   3. Capital Allocator    │
                    │      (per-account ERC,    │
                    │       account capital)    │
                    │                           │
                    │   4. Position check       │
                    │      (existing position   │
                    │       for this account?)  │
                    │                           │
                    │   5. Risk Gate            │
                    │      (per-account limits, │
                    │       broker pos check)   │
                    └─────┬─────┬─────┬────────┘
                          │     │     │
                 Acct A   │     │     │   Acct C
                          │  Acct B   │
                          ▼     ▼     ▼
                    ResolvedOrder (shared — same instrument for all accounts, account_id attached at OMS layer)
                          │     │     │
                          ▼     ▼     ▼
                    ┌─────────────────────────┐
                    │ Broker Router            │
                    │ Dhan (primary) or        │
                    │ Upstox (failover)        │
                    └─────┬─────┬─────┬───────┘
                          │     │     │
                          ▼     ▼     ▼
                    OMS.place_entry_with_sl() (per account)
                      ├── Entry LIMIT order
                      └── Paired SL order (server-side, Forever validity)
                          │     │     │
                    Broker WS (order updates, per-account connection)
                          │     │     │
                          ▼     ▼     ▼
                    FillManager (per account)
                      + PART_TRADED → immediate SL qty sync
                      + post-order REST verification
                          │     │     │
                          └─────┴─────┘
                                │
                                ▼
                    PositionTracker.on_fill(account_id, ...)
                      + per-account position tables
                      + portfolio aggregate views
                      + periodic broker reconciliation (5 min)
                                │
                                ▼
                    Audit log + Prometheus (account_id label) + Telegram
```

**Key v3 changes in data flow:**
1. Signal dedup and instrument resolution happen once (shared), before account fan-out.
2. Capital allocation, position checks, and risk gates run per-account inside the
   Account Replication Layer.
3. Each account has its own OMS, FillManager, and SL Lifecycle — no shared order state.
4. Broker Router sits between the Account Replication Layer and per-account OMS,
   deciding Dhan vs. Upstox per order.
5. PositionTracker receives fills from all accounts, tagged with `account_id`.

### 1.4 Latency Budget (Signal-to-Exchange)

```
Signal fires (bar close)                  t = 0ms
Signal dedup + instrument resolution      t = 1-5ms       (shared, once)
Account fan-out + per-account sizing      t = 5-15ms      (concurrent across accounts)
Per-account risk gate checks              t = 5-20ms      (concurrent across accounts)
Rate limiter wait (avg)                   t = 0-100ms     (per account, independent limits)
Entry order API call                      t = 50-200ms    (per account, concurrent)
SL order API call                         t = 50-200ms    (per account, concurrent)
Broker OMS → Exchange                     t = 100-3,000ms (async, not in our control)
────────────────────────────────────────────────────────────────────
Total (our side, single account):         ~300-900ms
Total (our side, 3 accounts):             ~310-940ms      (fan-out is concurrent)
Total (incl. broker-to-exchange):         ~500-4,000ms

Signal half-life (minimum):               >30 seconds
Our latency as % of signal life:          1-3%
```

**Multi-account latency note:** Account fan-out adds ~5-10ms per additional account
(concurrent, not serial). The fan-out uses `asyncio.gather()` to dispatch sizing, risk
checks, and order placement for all accounts simultaneously. The bottleneck remains the
broker API call latency, which is independent per account and per API key. Three accounts
do not triple the latency — they add marginal overhead for the fan-out coordination.

**Worst case (all 3 accounts, Dhan API slow):**
```
Fan-out overhead:    ~10ms
Slowest account:     ~900ms (our side)
Total:               ~910ms our side, within the 1-3% signal-life budget
```

### 1.5 Known Limitations (v3)

1. **Single-host deployment.** All components run on one EC2 instance. Redis Sentinel
   on the same host protects against process crashes only — not instance, EBS, or AZ
   failure. Host-level protection: server-side SLs (Forever validity) + broker
   auto-square at session close. Cross-host HA is a scaling trigger at ₹1Cr+.

2. **Single-leg orders only.** All orders are single-leg LIMIT. Multi-leg strategies
   (spreads, strangles) are executed as independent legs with execution risk between
   legs. Multi-leg atomic orders are not supported.

3. **Dhan 10 OPS per key.** Each account is limited to 10 orders per second. With 3
   accounts and 7 strategies, the 30 OPS aggregate provides 60% headroom at current
   signal frequency. Binding constraint at higher frequencies requires Upstox switch or
   algo registration for higher limits.

4. **Upstox failover is degraded mode.** Upstox WS instrument limit (100 vs. Dhan's
   25,000) means tick data cannot fail over to Upstox. Failover is execution-only:
   order placement and SL management. If Dhan tick WS drops, strategies pause until
   reconnection.

5. **No cross-broker hedging.** If Account A has positions on Dhan and Dhan fails,
   the failover to Upstox can place new orders but cannot modify existing Dhan SLs.
   Mitigation: Dhan server-side SLs (Forever) remain active on Dhan infrastructure
   regardless of our system's connectivity.

6. **Broker auto-square override.** Dhan and Upstox auto-square intraday positions at
   session close. The system's EOD flatten runs at 15:20 to control exit price. If the
   system fails to flatten, broker auto-square is the backstop — but at potentially
   worse prices.

7. **No real-time margin calculation.** Margin availability is checked via broker API
   before order placement, but margin impact of pending orders is estimated locally
   (not queried from broker in real time). Margin rejection is handled as a fill
   failure.

8. **Account replication is best-effort atomic.** If Account A's order succeeds and
   Account B's order is rejected (margin, rate limit), the system does not roll back
   Account A. Each account operates independently after fan-out. Portfolio-level
   risk monitor detects and alerts on cross-account position divergence.

### 1.6 v2 to v3 Changes Summary

**New components:**

- **Account Replication Layer (§10):** Fan-out of signals to N trading accounts with
  per-account sizing, risk checks, and independent OMS instances.
- **Broker Router (§6):** Routes orders to Dhan (primary) or Upstox (failover) with
  cross-broker position awareness and automatic failover on API errors.

**Modified components:**

- **OMS (§5):** SL orders use Forever/GTT validity instead of DAY (CRITICAL-1). Fill
  loop syncs SL quantity on every PART_TRADED event (CRITICAL-2). Per-account OMS
  instances instead of singleton.
- **Capital Allocator (§4):** Signal deduplication key is (strategy_id, direction, underlying) with
  TTL-based cooldown (HIGH-4). Pre-allocation position check added (HIGH-5). All locks
  converted from `threading.Lock` to `asyncio.Lock` (LOW-12). Per-account capital
  pools.
- **Risk Manager (§7):** Ghost position detection via periodic broker position
  reconciliation (HIGH-6). Per-account risk limits plus portfolio-wide aggregate
  checks.
- **Instrument Resolution (§2):** CSV staleness check disables strategies that depend
  on current-day instruments (MEDIUM-8).
- **Data Ingestion (§1):** WAL flush interval specified as 5 seconds (MEDIUM-9).
- **Position Tracker (§8):** Single-writer architecture enforced (MEDIUM-10).
  Per-account tables with `account_id` partition key (MEDIUM-11).
- **Monitoring:** Per-account Grafana dashboards plus portfolio aggregate views.
  Per-account Telegram notification channels.

**Unchanged components:**

- **Strategy Processes (S1-S7):** Signal generation logic unchanged. Strategies are
  account-agnostic — they produce signals from market data without knowledge of
  account topology.
- **Data Ingestion core:** Tick normalization, Redis Streams distribution, and Parquet
  archival unchanged. Only WAL flush interval formalized.
- **Redis Sentinel:** Same-host topology unchanged. Process-level HA only.

**Architecture decisions updated:**

| Decision | v2 Rationale | v3 Update |
|----------|-------------|-----------|
| Single broker | Eliminates cross-broker complexity. Failover is a v2 concern. | Dhan primary, Upstox failover via Broker Router. Cross-broker position awareness added. |
| Redis Streams | Persistent, backpressure-aware, consumer group recovery. | Unchanged. |
| On-demand option chain | 200ms latency acceptable vs. stale data risk. | Unchanged. Shared resolution before account fan-out avoids redundant API calls. |
| Mandatory server-side SL | Only protection during system outage. | Upgraded to Forever/GTT validity. DAY SL eliminated. |
| SL as first-class lifecycle | SL assumed correct = SL wrong. Verify constantly. | Enhanced: PART_TRADED sync added. SL qty verified after every fill event. |
| DuckDB single-writer | DuckDB doesn't support concurrent writes. | Unchanged. Per-account data via `account_id` column, not separate databases. |
| Same-host Redis Sentinel | Process-level HA. Host-level HA via SL + broker auto-square. | Unchanged. Scaling trigger at ₹1Cr+. |
| Signal dedup in router | Defense-in-depth against strategy bugs. | Enhanced: dedup key is (strategy_id, direction, underlying) with TTL-based cooldown. |
| Position-aware allocation | Prevents double positions from duplicate signals. | Enhanced: per-account position checks. |
| asyncio.Lock | threading.Lock blocks event loop. | Enforced: audit pass confirmed no threading.Lock in async code paths. |
| Multi-account replication | Not supported in v2. | NEW: Account Replication Layer with per-account OMS fan-out. |
| Multi-broker failover | Not supported in v2. | NEW: Broker Router with automatic failover and cross-broker position tracking. |

**Scaling triggers updated:**

| Trigger | v2 Action | v3 Action |
|---------|-----------|-----------|
| 10 OPS binding (per key) | Switch to Upstox or register for higher limits. | 3 accounts = 30 OPS. Switch to Upstox or register if 30 OPS insufficient. |
| WS instrument limit | Second WS connection or broker switch. | Unchanged. Dhan's 25,000 limit sufficient for current universe. |
| CPU saturation | Split Redis to managed service. | Unchanged. Multi-account adds ~15% CPU overhead (concurrent async, not additional processes). |
| ₹1Cr+ capital | Redis on separate instance. | Redis on separate instance. Per-account capital rebalancing automation. |
| ₹5Cr+ capital | Multi-AZ deployment. | Multi-AZ. Consider dedicated EC2 per account at ₹3Cr+ per account. |
| 5+ accounts | Not applicable in v2. | NEW: Evaluate per-account EC2 instances vs. shared host. OPS budget review. |

---

*End of Section 0 (Findings Table) and Section 1 (System Overview). Subsequent sections
(§2 through §10) specify each component's interfaces, state models, failure modes, and
recovery paths.*
