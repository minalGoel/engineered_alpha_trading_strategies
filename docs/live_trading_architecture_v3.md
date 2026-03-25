<!-- Table of Contents -->

# Live Trading System — Architecture Document v3

**Date:** 2026-03-24  
**Status:** Design (pre-implementation)  
**Scope:** v3 — single-leg orders, multi-account replication, multi-broker failover  
**Previous version:** live_trading_architecture.md (v2)

---

## Table of Contents

- [Section 0: Findings & Review Mapping](#section-0-findings--review-mapping)
- [Section 1: System Overview](#1-system-overview)
- [Section 2: Component Designs](#2-component-designs)
  - [Component 1: Data Ingestion Layer](#component-1-data-ingestion-layer)
  - [Component 2: Instrument Resolution Layer](#component-2-instrument-resolution-layer)
  - [Component 3: Strategy Orchestrator](#component-3-strategy-orchestrator)
  - [Component 4: Capital Allocation Engine](#component-4-capital-allocation-engine)
  - [Component 5: Order Management System (OMS)](#component-5-order-management-system-oms)
  - [Component 6: Multi-Broker Router](#component-6-multi-broker-router)
  - [Component 7: Risk Management Layer](#component-7-risk-management-layer)
  - [Component 8: Position & PnL Tracker](#component-8-position--pnl-tracker)
  - [Component 9: Audit & Compliance Logger](#component-9-audit--compliance-logger)
  - [Component 10: Monitoring & Alerting](#component-10-monitoring--alerting)
  - [Component 11: Account Replication Layer](#component-11-account-replication-layer)
- [Section 3: Cross-Cutting Concerns](#3-cross-cutting-concerns)
- [Section 4: Testing & Security](#4-testing--security)
- [Section 5: Open Questions & Must-Prototype](#5-open-questions--must-prototype)
- [Section 6: Implementation Phases](#6-implementation-phases)
- [Section 7: Regulatory Notes](#7-regulatory-notes)

---


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

---


## Component 1: Data Ingestion Layer

> **Multi-Account Note:** Data ingestion is SHARED across all accounts. Market data is not account-specific. A single Data Ingester instance serves all accounts. No downstream component (Signal Router, OMS, Position Tracker) affects what data is ingested or how it is stored. Adding or removing trading accounts requires zero changes to this component.

### Responsibility

- Subscribe to broker's tick-by-tick websocket for real-time market data
- Normalize ticks into a canonical `Tick` model
- Distribute ticks to strategy processes via Redis Streams (persistent, backpressure-aware)
- Record raw ticks via write-ahead log (WAL) + Parquet with 5-second flush
- **Does NOT** interpret ticks, compute indicators, or make trading decisions

### Interface Contracts

**Input:** Broker tick websocket (single connection, up to instrument limit).

**Output — Redis Streams:**

```python
class Tick(pydantic.BaseModel):
    symbol: str              # canonical: e.g. "NIFTY50-INDEX"
    ltp: float               # last traded price
    bid: float               # best bid
    ask: float               # best ask
    bid_qty: int
    ask_qty: int
    volume: int              # cumulative daily volume
    oi: int                  # open interest (0 for indices)
    exchange_ts: int         # exchange timestamp (epoch ms)
    receive_ts: int          # our receive timestamp (epoch ms)
```

Streams (one per instrument class):
- `STREAM:TICK:SPOT` — NIFTY50, BANKNIFTY, VIX indices
- `STREAM:TICK:FUT:{symbol}` — individual futures
- `STREAM:TICK:OPT:{underlying}` — option ticks by underlying
- `STREAM:TICK:EQ:{symbol}` — equities (S4)

Each strategy process creates a **consumer group** on the streams it subscribes to. Ticks persist in the stream, trimmed to N entries via `XTRIM MAXLEN`. The trim threshold is tuned after load testing — initial estimate: 60,000 entries per stream (~5 min at 200 ticks/s), validated by measuring actual peak publish rates during market open.

If a strategy process is slow or reconnects, it reads from its last acknowledged offset. No lost ticks.

Message format: orjson-serialized `Tick`. ~150 bytes per tick. Published via `XADD`.

**Output — Tick WAL + Parquet:**

```
WAL:      /var/log/trading/tick_wal/{date}.wal  (append-only mmap'd binary)
Parquet:  /var/log/trading/parquet/{date}/{symbol}/{HH}.parquet
S3:       s3://trading-data-archive/{date}/ticks/{symbol}/{HH}.parquet
```

Every tick is appended to the WAL **before** publishing to Redis. The WAL survives process crashes including SIGKILL and segfaults. On restart, the Parquet flush task replays the WAL from the last checkpoint, deduplicates by `(symbol, exchange_ts)` pair, and writes to Parquet.

WAL record format: fixed-width 60 bytes. At 1000 ticks/s peak = 60KB/s = ~180MB/day.

Parquet flush interval: **5 seconds**. At ~75KB per flush — trivial I/O. `atexit.register()` AND `signal.signal(SIGTERM/SIGINT)` handlers flush on graceful shutdown. For SIGKILL/segfault, only the WAL survives (max 5s of un-flushed ticks).

**Idempotent replay:** The WAL contains a monotonic sequence number per tick. The Parquet writer tracks the last written sequence. On replay, ticks with sequence <= last_written are skipped. Partial Parquet writes (crash mid-flush) are detected by checking the footer — incomplete files are discarded and rewritten from WAL.

### State

| What | Where | Lifecycle |
|------|-------|-----------|
| WS connection handle | In-memory | Process lifetime |
| Subscription list | Config -> in-memory | Session |
| Tick WAL file | Disk (mmap'd) | Survives crashes. New file per day. |
| Tick buffer (pre-Parquet) | In-memory deque, max 5s | Flushed every 5s |
| Last tick per symbol | Redis `LASTTICK:{symbol}` (hash) | Overwritten per tick |
| Connection health | Redis `HEALTH:data_ingester` | Heartbeat every 5s with TTL 15s |

**No per-account state in this component — all state is shared.** The subscription list, WAL, Parquet archive, Redis Streams, and `LASTTICK` hashes are account-agnostic. Whether the system trades one account or ten, the Data Ingester maintains the exact same state footprint.

### Concurrency Model

Single async process: `asyncio` + `websockets`.

- `asyncio.Task`: WS reader (receives ticks, writes WAL, publishes to Redis Stream, updates LASTTICK)
- `asyncio.Task`: Parquet flush loop (5s)
- `asyncio.Task`: Heartbeat (5s)

I/O-bound — a single async loop handles thousands of ticks/sec.

### Failure Modes

| Failure | Detection | Recovery |
|---------|-----------|----------|
| WS disconnect | `ConnectionClosed` exception | Exponential backoff: 1s, 2s, 4s, 8s, max 30s. Re-subscribe. |
| WS stale (connected, no ticks) | Watchdog: no tick on ANY instrument for 10s | Force close + reconnect |
| **All instruments silent simultaneously** | Zero ticks on all streams for 15s | Exchange halt/closure. Telegram CRITICAL. Suppress new entries. Existing positions protected by server-side SL. |
| Redis down | `aioredis` ConnectionError | Retry via Sentinel. WAL continues locally. Strategies paused. OMS enters degraded mode. |
| Process crash | systemd detects exit | Restart. WAL survives. Replay WAL -> Parquet. |

### Multi-Account Impact

Data ingestion is completely account-agnostic. This is a deliberate architectural property, not an accident.

**Why data ingestion has no account awareness:**

1. **Market data is universal.** The NIFTY50 index tick at 10:00:00.123 is the same number regardless of which account is trading on it. There is no concept of "Account A's NIFTY tick" versus "Account B's NIFTY tick." The exchange publishes one price stream per instrument, and we receive one copy.

2. **Subscriptions are instrument-driven, not account-driven.** The websocket subscription list is determined by which instruments the strategies need — NIFTY50 spot, BANKNIFTY spot, relevant futures and option chains. Adding a second or third trading account does not change which instruments we need to observe.

3. **Redis Streams are broadcast by design.** Each strategy process has its own consumer group on the relevant streams. Consumer groups are identified by strategy, not by account. Strategy S1 reads from `STREAM:TICK:SPOT` whether it is generating signals for one account or five. The tick fan-out to multiple accounts happens downstream in the Signal Router, not here.

4. **Storage is shared.** The WAL and Parquet archive record raw market data for audit and replay. These files serve all accounts equally. Replaying a day's ticks for backtesting or reconciliation does not require knowing which accounts were active.

**Scaling implications:** Adding accounts increases load on downstream components (Signal Router must route to more OMS instances, Position Tracker must maintain more account ledgers). It does NOT increase load on Data Ingestion. The tick rate, WAL write rate, Parquet flush rate, and Redis publish rate are all functions of instrument count and market activity, never of account count.

**Configuration boundary:** The Data Ingester's only configuration inputs are the broker API credentials (for the websocket connection) and the instrument subscription list. Neither of these is per-account. The broker credentials are for the data feed, which is a single connection regardless of how many trading accounts exist downstream.

---


## Component 2: Instrument Resolution Layer

> **v3 shared resolution:** Instrument resolution is SHARED. The same `security_id` is used for all accounts. A signal produces one resolved instrument, which is then replicated to N accounts. The resolution pipeline runs ONCE per signal, not once per account.

---

### Responsibility

- Translate abstract strategy signals ("go long NIFTY", "sell PE") into a concrete broker instrument ID with a limit price
- Maintain broker instrument master (daily CSV) as canonical mapping
- **Always make an on-demand option chain API call when a signal fires** for order pricing
- Maintain background 5s polling for monitoring/Greeks/display only
- Validate liquidity and spread
- **Does NOT** place orders, manage positions, or size trades

---

### Instrument Master

Downloaded at 08:30 IST daily. Parsed into O(1) lookup indices:

```python
class InstrumentRecord(pydantic.BaseModel):
    instrument_id: str            # broker's unique ID
    trading_symbol: str           # human-readable
    underlying: str               # "NIFTY" | "BANKNIFTY" | equity ticker
    exchange_segment: str         # "NSE_FNO" | "NSE_EQ"
    instrument_type: str          # "OPTIDX" | "FUTIDX" | "EQUITY"
    option_type: str | None       # "CE" | "PE" | None
    strike_price: float | None
    expiry_date: date | None
    lot_size: int
    tick_size: float              # e.g. 0.05 for options
    freeze_qty: int               # max qty before slicing needed

class InstrumentMaster:
    _option_index: dict[tuple[str, date, float, str], InstrumentRecord]
    _futures_index: dict[tuple[str, date], InstrumentRecord]
    _equity_index: dict[str, InstrumentRecord]

    def lookup_option(self, underlying, expiry, strike, option_type) -> InstrumentRecord | None
    def lookup_future(self, underlying, expiry) -> InstrumentRecord | None
    def lookup_equity(self, symbol) -> InstrumentRecord | None
```

**Fallback on download failure:** Retry 3x (08:30, 08:35, 08:40). If all fail, use yesterday's CSV with WARNING. **On Tuesdays (weekly expiry day): if fallback triggered, disable S5 for the day** — 0-DTE contracts may not be in yesterday's file. Telegram WARNING.

The InstrumentMaster is loaded once and shared across all accounts. Since instrument IDs are exchange-level identifiers (not broker-account-level), a single master serves the entire system regardless of how many accounts are active.

---

### Option Chain — Dual Mode

| Mode | Purpose | Frequency | Used For |
|------|---------|-----------|----------|
| Background poll | Monitoring, Greeks, dashboards | Every 5s | Grafana, risk monitor, Position Tracker Greeks |
| **On-demand at signal time** | **Order pricing** | Per signal (~10-20/day) | Limit price, spread check, liquidity validation |

The on-demand call adds 200-400ms to the signal-to-order path. This is the correct tradeoff: 200ms latency is far cheaper than placing orders with 5s-stale pricing data that would require multiple reprice cycles.

```python
class OptionChainEntry(pydantic.BaseModel):
    strike: float
    option_type: str              # "CE" | "PE"
    ltp: float
    bid: float
    ask: float
    bid_qty: int
    ask_qty: int
    oi: int
    volume: int
    iv: float | None              # may not be provided by all brokers
    delta: float | None           # may not be provided — compute locally if missing
    gamma: float | None
    theta: float | None
    vega: float | None
    last_trade_ts: int            # epoch ms

class OptionChainSnapshot:
    underlying: str
    expiry: date
    spot_price: float
    snapshot_ts: int
    entries: dict[tuple[float, str], OptionChainEntry]
```

**Greeks availability:** If the broker's option chain API does not return Greeks, compute locally using Black-76 with IV derived from bid/ask midpoint and the BSM inversion. This must be validated during paper trading.

---

### Resolution Pipeline

#### Signal and Resolved Order Models

```python
class StrategySignal(pydantic.BaseModel):
    strategy_id: str
    signal_id: str                # UUID, unique per signal
    signal_ts: int                # epoch ms
    direction: Literal["LONG", "SHORT"]
    underlying: str
    instrument_hint: Literal["CE", "PE", "FUT", "EQ"]
    expiry_preference: Literal["WEEKLY", "MONTHLY", "NEAREST"] | None
    urgency: Literal["NORMAL", "URGENT"]
    metadata: dict                # indicator values, bar data for audit
    legs: list[LegSpec] | None    # v2 extensibility: multi-leg orders (see below)

class ResolvedOrder(pydantic.BaseModel):
    signal: StrategySignal
    instrument_id: str            # broker instrument ID
    trading_symbol: str
    exchange_segment: str
    transaction_type: Literal["BUY", "SELL"]
    product_type: Literal["INTRADAY", "CNC", "MARGIN"]  # v1 restriction: only these 3 product types. Dhan also supports MTF, CO, BO.
    quantity: int                  # in units (lot_size x lots)
    limit_price: float
    tick_size: float
    lot_size: int
    freeze_qty: int
    spread_bps: float
    cost_estimate: float          # estimated RT cost in INR
    sl_trigger_price: float       # for mandatory server-side SL
    sl_limit_price: float         # SL limit = trigger +/- 2 ticks
    sl_validity: Literal["DAY", "GTT"]  # "DAY" for intraday, "GTT" for overnight (Forever Order)
```

#### Algorithm

```
1. RECEIVE StrategySignal

2. DETERMINE expiry:
   S1: monthly (last Thu)    S2: nearest futures    S3: monthly
   S4: equity (no expiry)    S5: weekly Tuesday     S6: monthly
   S7: nearest stock futures

3. FOR options (CE/PE):
   a. GET spot price from Redis LASTTICK (real-time)
   b. COMPUTE ATM strike = floor(spot / 50 + 0.5) * 50
      Tie-break on exact midpoint: lower strike (better liquidity)
   c. MAKE on-demand option chain API call (200-400ms)
      - On failure: retry ONCE. If still fails, fall back to cached
        snapshot if <2s old. If cache also stale: REJECT signal.
   d. GET bid/ask for (ATM_strike, CE/PE) from fresh snapshot
   e. LIQUIDITY check: if bid==0 or ask==0, try +/-1 strike ITM.
      Still zero -> REJECT.
   f. SPREAD check:
      spread_bps = (ask - bid) / mid * 10000
      Normal: reject if > 500 bps
      0-DTE (S5 Tuesday): reject if > 1000 bps
   g. LIMIT PRICE:
      BUY + NORMAL:  best_ask
      BUY + URGENT:  best_ask + 1 tick
      SELL + NORMAL: best_bid
      SELL + URGENT: best_bid - 1 tick
   h. ROUND to tick_size
   i. COMPUTE SL prices:
      BUY: sl_trigger = entry - strategy_stop_points
      SELL: sl_trigger = entry + strategy_stop_points
      sl_limit = trigger +/- 2 ticks
   j. DETERMINE SL validity:
      Intraday strategies (S1,S3,S5): "DAY"
      Overnight/multi-day (S2,S6,S7): "GTT" (Good Till Triggered)
   k. LOOKUP instrument_id from InstrumentMaster
   l. ESTIMATE cost via cost_model
   m. RETURN ResolvedOrder

4. FOR futures: same flow, no strike resolution
5. FOR equities: direct lookup, no chain call needed
```

#### 0-DTE Guards (S5 on Tuesday)

Reject `bid_qty < 100` lots. Wider spread threshold (1000 bps vs 500 bps). Force `urgency=URGENT`.

**Tuesday CSV fallback interaction:** If the instrument master fell back to yesterday's CSV (download failure), S5 is disabled for the entire day. 0-DTE weekly contracts that expired or were listed overnight will not be in yesterday's file, making resolution unreliable. The system logs a Telegram WARNING at 08:40 when this fallback activates and S5 disablement takes effect.

---

### Multi-Leg Extensibility

**v2 extensibility note:** Signal can carry `legs: list[LegSpec]` for multi-leg orders. v1: always single leg.

When `legs` is populated, each `LegSpec` defines an independent strike/option_type/direction, and the resolution pipeline runs steps 3a-3m for each leg, producing a `list[ResolvedOrder]` instead of a single `ResolvedOrder`. Multi-leg resolution still counts as a single signal event — the on-demand option chain call in step 3c is made once and reused across all legs.

```python
class LegSpec(pydantic.BaseModel):
    leg_id: int                    # sequential: 0, 1, 2, ...
    instrument_hint: Literal["CE", "PE", "FUT", "EQ"]
    direction: Literal["LONG", "SHORT"]
    strike_offset: int             # ATM offset in strike steps (0 = ATM, +1 = 1 strike OTM, etc.)
    ratio: int                     # lot multiplier (1 for standard, 2 for ratio spreads)
```

---

### Multi-Account Impact

Instrument resolution is a **shared, account-agnostic** operation. The key insight is that instrument IDs, option chains, spot prices, and spread calculations are exchange-level properties — they do not vary by trading account.

**Execution flow for N accounts:**

```
StrategySignal
    |
    v
[Instrument Resolution]  <-- runs ONCE
    |
    v
ResolvedOrder (single instance)
    |
    +---> Account 1: size, place order (OMS)
    +---> Account 2: size, place order (OMS)
    +---> Account N: size, place order (OMS)
```

What varies per account is downstream of resolution:

| Concern | Shared (resolution) | Per-account (OMS/sizing) |
|---------|---------------------|--------------------------|
| Instrument ID | Yes | -- |
| Limit price | Yes | -- |
| Spread validation | Yes | -- |
| SL trigger/limit prices | Yes | -- |
| Position quantity | -- | Yes (capital-based sizing) |
| Order placement | -- | Yes (per-account API call) |
| Margin check | -- | Yes (per-account balance) |
| Freeze qty slicing | -- | Yes (per-account quantity may differ) |

This design avoids redundant API calls. A single on-demand option chain call (200-400ms) serves all accounts. If resolution ran per-account, N accounts would require N identical API calls adding N * 200-400ms of latency and N times the rate limit consumption — wasteful since the exchange state is identical for all of them.

---

### State

| What | Where | Lifecycle |
|------|-------|-----------|
| InstrumentMaster | In-memory | Loaded 08:30, immutable for session |
| Background chain snapshots | In-memory + Redis `CHAIN:*` | Updated 5s (monitoring only) |
| Spot prices | Redis `LASTTICK:*` | Updated per tick |

All state entries are shared across accounts. There is no per-account state within the resolution layer.

---

### Concurrency

Async tasks within the Signal Router process. `resolve()` is `async def` — the on-demand API call is awaited, not blocking.

If two signals arrive simultaneously (e.g., S1 and S3 both fire on the same bar), each gets its own `resolve()` coroutine. The option chain API calls are independent and can execute concurrently via `asyncio.gather`. The InstrumentMaster is read-only after 08:30, so no locking is required for instrument lookups.

In multi-account mode, the resolved result is passed by reference to the per-account OMS dispatch. Resolution does not need to be aware of how many accounts exist or how they are configured.

---

### Failure Modes

| Failure | Detection | Impact |
|---------|-----------|--------|
| Instrument CSV download fails 3x | Retry exhaustion at 08:40 | Use yesterday's CSV. Disable S5 on Tuesdays. Telegram WARNING. |
| On-demand chain API fails + retry fails | Two consecutive HTTP errors | Fall back to cached snapshot if <2s old. If stale: REJECT signal. Signal not sent to any account. |
| Instrument not found in master | `lookup_option` returns None | REJECT signal. Log with full signal details. |
| Spread exceeds threshold | `spread_bps > 500` (or 1000 for 0-DTE) | REJECT signal. Log spread value. |
| Zero liquidity (bid=0 or ask=0) | Liquidity check step 3e | Try adjacent strike. If still zero: REJECT. |
| Spot price missing from Redis | `LASTTICK` key absent | Cannot compute ATM strike. REJECT signal. |

Signal rejection at the resolution layer is absolute — if resolution rejects, no account receives the order. This is correct behavior: if the instrument cannot be resolved reliably, it should not be traded on any account.

---


## Component 3: Strategy Orchestrator

### Responsibility

- Launch each strategy as an isolated OS process with resource limits
- Manage strategy lifecycle from discovery through shutdown via a formal state machine
- Provide the `LiveStrategy` plugin interface — the sole contract a strategy author must implement
- Aggregate ticks into configurable-interval OHLCV bars via per-strategy `BarBuilder` instances
- Route market data to strategies via Redis Streams consumer groups
- Collect signals from strategies and forward to the Signal Router (deduplication is handled downstream by the Capital Allocator's SignalDeduplicator)
- Enforce strategy isolation: crash containment, resource caps, execution timeouts
- Validate strategy configurations at startup before any process is spawned
- **Does NOT** place orders, manage positions, allocate capital, or interact with the broker

---

### Strategy Manager

The Strategy Manager is a sub-component of the Orchestrator responsible for discovering, validating, and lifecycle-managing all strategies. It runs inside the Orchestrator's main process (not in a strategy child process).

#### Discovery

Strategies are discovered via explicit registration in `config/strategies.yaml`. There is no filesystem auto-discovery — every deployable strategy must be listed in config with its fully-qualified class path.

```yaml
# config/strategies.yaml — strategy registry section
strategy_registry:
  S1:
    class_path: "live.strategies.s1_orb.S1OrbStrategy"
    enabled: true
  S2:
    class_path: "live.strategies.s2_overnight.S2OvernightStrategy"
    enabled: true
  S3:
    class_path: "live.strategies.s3_vwap_mr.S3VwapMrStrategy"
    enabled: true
  S4:
    class_path: "live.strategies.s4_momentum.S4MomentumStrategy"
    enabled: true
  S5:
    class_path: "live.strategies.s5_expiry_day.S5ExpiryDayStrategy"
    enabled: true
  S6:
    class_path: "live.strategies.s6_vol_premium.S6VolPremiumStrategy"
    enabled: true
  S7:
    class_path: "live.strategies.s7_pairs.S7PairsStrategy"
    enabled: true
  # Adding S8 requires only: write the class, add this entry, add strategy config block
```

At startup, the Strategy Manager:

1. Reads `strategy_registry` from config
2. Imports each `class_path` via `importlib.import_module` + `getattr`
3. Verifies each class is a subclass of `LiveStrategy` (raises `StrategyRegistrationError` if not)
4. Loads the per-strategy configuration block (see Strategy Configuration Schema below)
5. Runs validation (see Startup Validation below)
6. Instantiates each enabled strategy with its config
7. Transitions each to READY state

**Why not auto-discovery:** Auto-discovery (scanning a directory for `LiveStrategy` subclasses) is fragile — a half-written file, a syntax error, or a test fixture can be picked up. Explicit registration in config is one line per strategy and eliminates ambiguity. The interface does not preclude adding auto-discovery in v2, but v1 uses explicit registration.

#### Lifecycle State Machine

Every strategy moves through a formal state machine. The Strategy Manager is the sole authority for state transitions.

```
                    ┌────────────┐
                    │ DISCOVERED │  (class imported, config loaded)
                    └─────┬──────┘
                          │ validate()
                          ▼
                    ┌────────────┐
              ┌────│   READY    │  (validated, not yet running)
              │    └─────┬──────┘
              │          │ start()
              │          ▼
              │    ┌────────────┐
              │    │  RUNNING   │  (process alive, accepting ticks)
              │    └──┬───┬─────┘
              │       │   │
              │       │   │ suppress()           kill()
              │       │   ▼                        │
              │       │ ┌────────────┐             │
              │       │ │ SUPPRESSED │  (alive,    │
              │       │ │            │   no new    │
              │       │ │            │   signals)  │
              │       │ └──┬─────────┘             │
              │       │    │ unsuppress()           │
              │       │    │                        │
              │       │    ▼                        │
              │       │  RUNNING ◄─────────────────┘ (if resumable)
              │       │                             │
              │       │ kill() (non-resumable)      │
              │       ▼                             ▼
              │    ┌────────────┐
              │    │  KILLED    │  (process stopped, no new entries)
              │    └─────┬──────┘
              │          │ disable()
              │          ▼
              │    ┌────────────┐
              └───►│ DISABLED   │  (not running, skipped on next startup)
                   └────────────┘
```

**State definitions:**

| State | Process Alive | Accepts Ticks | Emits Signals | New Entries Allowed |
|-------|--------------|---------------|---------------|---------------------|
| DISCOVERED | No | No | No | No |
| READY | No | No | No | No |
| RUNNING | Yes | Yes | Yes | Yes |
| SUPPRESSED | Yes | Yes | No (suppressed) | No |
| KILLED | No | No | No | No (existing positions exit via SL/normal) |
| DISABLED | No | No | No | No |

**Transition triggers:**

| Transition | Trigger | Authority |
|-----------|---------|-----------|
| DISCOVERED → READY | Startup validation passes | Strategy Manager |
| READY → RUNNING | `start()` called after consumer groups created | Strategy Manager |
| RUNNING → SUPPRESSED | VIX filter, schedule (e.g., S5 not on Tuesday), manual CLI | Risk Manager or Strategy Manager |
| SUPPRESSED → RUNNING | Suppression condition clears | Risk Manager or Strategy Manager |
| RUNNING → KILLED | Kill condition fires (Sharpe, drawdown, consecutive losses) | Risk Manager |
| SUPPRESSED → KILLED | Kill condition fires while suppressed | Risk Manager |
| KILLED → DISABLED | Manual operator action or config change | Operator via CLI |
| RUNNING → DISABLED | Manual disable via CLI | Operator |
| DISABLED → READY | Config change + restart (v1: requires system restart) | Operator |

```python
import enum

class StrategyState(enum.Enum):
    DISCOVERED = "DISCOVERED"
    READY = "READY"
    RUNNING = "RUNNING"
    SUPPRESSED = "SUPPRESSED"
    KILLED = "KILLED"
    DISABLED = "DISABLED"

VALID_TRANSITIONS: dict[StrategyState, set[StrategyState]] = {
    StrategyState.DISCOVERED: {StrategyState.READY, StrategyState.DISABLED},
    StrategyState.READY:      {StrategyState.RUNNING, StrategyState.DISABLED},
    StrategyState.RUNNING:    {StrategyState.SUPPRESSED, StrategyState.KILLED, StrategyState.DISABLED},
    StrategyState.SUPPRESSED: {StrategyState.RUNNING, StrategyState.KILLED, StrategyState.DISABLED},
    StrategyState.KILLED:     {StrategyState.DISABLED},
    StrategyState.DISABLED:   {StrategyState.READY},  # only via restart
}

class StrategyEntry:
    """Tracks a single strategy's lifecycle inside the Strategy Manager."""
    strategy_id: str
    strategy_class: type          # the LiveStrategy subclass
    config: "StrategyConfig"      # parsed from strategies.yaml
    state: StrategyState
    process: multiprocessing.Process | None
    pid: int | None
    started_at: int | None        # epoch ms
    last_heartbeat: int | None    # epoch ms
    kill_reason: str | None
    suppression_reasons: set[str] # e.g. {"vix_filter", "schedule"}

    def transition(self, target: StrategyState) -> None:
        if target not in VALID_TRANSITIONS[self.state]:
            raise InvalidTransitionError(
                f"Cannot transition {self.strategy_id} from {self.state} to {target}. "
                f"Valid targets: {VALID_TRANSITIONS[self.state]}"
            )
        logger.info("strategy_state_transition",
                    strategy_id=self.strategy_id,
                    from_state=self.state.value,
                    to_state=target.value)
        self.state = target
```

#### Startup Validation

Before any strategy process is spawned, the Strategy Manager validates every enabled strategy. Validation failures are FATAL — the system will not start with an invalid strategy config.

```python
class StrategyManager:
    _strategies: dict[str, StrategyEntry]

    def validate_all(self) -> list[str]:
        """Validate all enabled strategies. Returns list of errors (empty = success)."""
        errors: list[str] = []

        for sid, entry in self._strategies.items():
            if entry.state == StrategyState.DISABLED:
                continue

            cfg = entry.config

            # 1. Class implements LiveStrategy ABC
            if not issubclass(entry.strategy_class, LiveStrategy):
                errors.append(f"{sid}: class {entry.strategy_class} is not a LiveStrategy subclass")

            # 2. Required config fields present
            required = ["bar_interval_s", "subscriptions", "instrument_type",
                       "direction_hint", "fill_params", "risk_params"]
            for field in required:
                if getattr(cfg, field, None) is None:
                    errors.append(f"{sid}: missing required config field '{field}'")

            # 3. Subscriptions are satisfiable
            for stream in cfg.subscriptions:
                if not self._stream_exists_or_will_exist(stream):
                    errors.append(f"{sid}: subscription '{stream}' is not a known stream")

            # 4. Bar interval is positive and reasonable
            if cfg.bar_interval_s is not None:
                if cfg.bar_interval_s < 1:
                    errors.append(f"{sid}: bar_interval_s must be >= 1, got {cfg.bar_interval_s}")
                if cfg.bar_interval_s > 86400:
                    errors.append(f"{sid}: bar_interval_s must be <= 86400, got {cfg.bar_interval_s}")

            # 5. Risk params within system bounds
            if cfg.risk_params.max_position_lots < 1:
                errors.append(f"{sid}: max_position_lots must be >= 1")
            if cfg.risk_params.stop_points <= 0:
                errors.append(f"{sid}: stop_points must be > 0")

            # 6. Fill params valid
            if cfg.fill_params.max_patience_s < cfg.fill_params.reprice_interval_s:
                errors.append(f"{sid}: max_patience_s must be >= reprice_interval_s")

            # 7. Schedule is parseable (if present)
            if cfg.schedule is not None:
                try:
                    parse_schedule(cfg.schedule)
                except ScheduleParseError as e:
                    errors.append(f"{sid}: invalid schedule: {e}")

            # 8. Kill condition is parseable
            if cfg.kill_condition is not None:
                try:
                    parse_kill_condition(cfg.kill_condition)
                except KillConditionParseError as e:
                    errors.append(f"{sid}: invalid kill_condition: {e}")

            if not errors:
                entry.transition(StrategyState.READY)

        return errors
```

**Hot-add (v1 decision):** Adding a new strategy requires a system restart. The interface (explicit config registration + class import) does not preclude hot-add in v2, but v1 does not support it. Rationale: hot-add requires dynamic consumer group creation, capital reallocation, and risk limit adjustment — all of which are safer to do during a controlled startup sequence.

---

### LiveStrategy Protocol (Plugin Interface)

This is the sole contract a strategy author must implement to deploy a new strategy. A developer building strategy S8 reads this interface, writes one class, adds one config block, and deploys.

```python
from abc import ABC, abstractmethod
from typing import Literal
import pydantic


class StrategyRequirements(pydantic.BaseModel):
    """Declared by each strategy — tells the system what resources it needs."""
    subscriptions: list[str]
    """Redis stream names this strategy consumes.
    Examples: ["STREAM:TICK:SPOT"], ["STREAM:TICK:FUT:NIFTY", "STREAM:TICK:FUT:BANKNIFTY"]
    The orchestrator creates consumer groups for these streams before the strategy starts."""

    bar_interval_s: int | None
    """Bar duration in seconds. None = no bar aggregation (event-driven only, e.g., S2).
    The orchestrator creates a BarBuilder with this interval for the strategy.
    Examples: 60 (S5), 300 (S3, S6, S7), 1800 (S1), 86400 (S4), None (S2)."""

    instrument_type: Literal["CE", "PE", "FUT", "EQ"]
    """What the strategy trades. Used by Instrument Resolver to know what to resolve."""

    expiry_preference: Literal["WEEKLY", "MONTHLY", "NEAREST"] | None
    """For options/futures. None for equities."""

    direction_hint: Literal["LONG_ONLY", "SHORT_ONLY", "BOTH"]
    """Constrains which directions this strategy can signal.
    Enforced at the Signal Router — a LONG_ONLY strategy emitting a SHORT signal is rejected."""


class StrategyConfig(pydantic.BaseModel):
    """Full configuration for one strategy. Loaded from strategies.yaml.
    Passed to the strategy at on_init(). Strategy code reads config, never writes it."""

    strategy_id: str
    enabled: bool
    class_path: str

    # Requirements (also declared in code, config overrides for flexibility)
    subscriptions: list[str]
    bar_interval_s: int | None
    instrument_type: Literal["CE", "PE", "FUT", "EQ"]
    expiry_preference: Literal["WEEKLY", "MONTHLY", "NEAREST"] | None
    direction_hint: Literal["LONG_ONLY", "SHORT_ONLY", "BOTH"]

    # Capital
    capital_weight: float            # static allocation weight (used days 1-40)
    kelly_override: float | None     # if set, overrides computed Kelly

    # Fill parameters
    fill_params: "FillParams"

    # Risk parameters
    risk_params: "RiskParams"

    # Kill condition
    kill_condition: "KillCondition"

    # Schedule (optional — restricts which days/times the strategy runs)
    schedule: "ScheduleSpec | None"

    # VIX filter
    vix_suppress_above: float | None   # suppress if VIX > this. None = no filter.
    vix_suppress_below: float | None   # suppress if VIX < this. None = no filter.

    # Strategy-specific parameters (opaque to the system, passed through)
    params: dict


class FillParams(pydantic.BaseModel):
    reprice_interval_s: float        # seconds between reprice attempts
    max_patience_s: float            # total patience before abandon
    pricing_mode: Literal["PASSIVE", "MIDPOINT", "AGGRESSIVE"]


class RiskParams(pydantic.BaseModel):
    stop_points: float               # SL distance in price points
    max_position_lots: int           # max lots for this strategy
    max_daily_trades: int            # max round-trips per day
    max_daily_loss: float            # max daily loss in ₹ before auto-suppress


class KillCondition(pydantic.BaseModel):
    metric: str                      # "sharpe_60d", "consecutive_losses", "drawdown_pct", etc.
    threshold: float
    lookback_days: int
    action: Literal["stop_new_entries", "flatten_immediately", "halt_all"]


class ScheduleSpec(pydantic.BaseModel):
    """Restricts when a strategy is active. Outside this schedule, the strategy
    is SUPPRESSED (process alive, ticks flowing, signals suppressed)."""
    active_days: list[Literal["MON", "TUE", "WED", "THU", "FRI"]] | None
    """If set, strategy only runs on these days. S5 example: ["TUE"]."""
    active_from_ist: str | None       # "09:20" — strategy signals suppressed before this
    active_until_ist: str | None      # "15:15" — strategy signals suppressed after this


class Tick(pydantic.BaseModel):
    symbol: str                       # canonical: e.g. "NIFTY50-INDEX"
    ltp: float                        # last traded price
    bid: float                        # best bid
    ask: float                        # best ask
    bid_qty: int
    ask_qty: int
    volume: int                       # cumulative daily volume
    oi: int                           # open interest (0 for indices)
    exchange_ts: int                  # exchange timestamp (epoch ms)
    receive_ts: int                   # our receive timestamp (epoch ms)


class Bar(pydantic.BaseModel):
    symbol: str
    interval_s: int                   # bar duration that produced this bar
    open: float
    high: float
    low: float
    close: float
    volume: int                       # volume in this bar (not cumulative)
    vwap: float                       # volume-weighted average price for this bar
    bar_start_ts: int                 # epoch ms — start of bar window
    bar_end_ts: int                   # epoch ms — end of bar window (= start + interval)
    tick_count: int                   # number of ticks aggregated into this bar
    has_gap: bool                     # True if suspicious tick gap detected within bar
    max_tick_gap_ms: int              # longest gap between consecutive ticks in this bar


class FillNotification(pydantic.BaseModel):
    strategy_id: str
    signal_id: str
    order_id: str
    instrument_id: str
    trading_symbol: str
    transaction_type: Literal["BUY", "SELL"]
    ordered_qty: int
    filled_qty: int
    remaining_qty: int
    avg_fill_price: float
    fill_status: Literal["FULL_FILL", "PARTIAL_FILL", "REJECTED", "CANCELLED", "TIMEOUT"]
    fill_ts: int                      # epoch ms
    sl_order_id: str | None           # paired SL order, if placed
    account_id: str                   # Account that received this fill.


class PositionUpdate(pydantic.BaseModel):
    strategy_id: str
    instrument_id: str
    trading_symbol: str
    direction: Literal["LONG", "SHORT", "FLAT"]
    quantity: int                     # absolute (0 when FLAT)
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    sl_order_id: str | None
    sl_trigger_price: float | None
    account_id: str                   # Account this position belongs to. In multi-account
                                      # mode, strategies receive one update per account.


class StrategySignal(pydantic.BaseModel):
    strategy_id: str
    signal_id: str                    # UUID, unique per signal
    signal_ts: int                    # epoch ms
    direction: Literal["LONG", "SHORT"]
    underlying: str
    instrument_hint: Literal["CE", "PE", "FUT", "EQ"]
    expiry_preference: Literal["WEEKLY", "MONTHLY", "NEAREST"] | None
    urgency: Literal["NORMAL", "URGENT"]
    legs: list["LegSpec"] | None = None  # v2 extensibility: multi-leg orders. v1: always None.
    metadata: dict                    # indicator values, bar data for audit trail


class LiveStrategy(ABC):
    """
    Abstract base class for all live trading strategies.

    A strategy is a self-contained signal generator. It receives market data
    (ticks and bars), maintains internal state, and emits StrategySignal objects
    when it wants to enter or exit a position.

    A strategy is ALLOWED to:
    - Read its own config (self.config)
    - Read ticks and bars delivered to its lifecycle hooks
    - Read its own position via on_position_update notifications
    - Read its own fill notifications via on_fill
    - Emit StrategySignal objects by returning them from on_tick / on_bar_close
    - Maintain arbitrary internal state (indicators, counters, flags)
    - Access read-only market data from the tick stream

    A strategy MUST NOT:
    - Access other strategies' state, config, or positions
    - Interact with Redis directly (the orchestrator mediates all data flow)
    - Call OMS, broker, or any execution API
    - Modify shared system state (config, risk limits, kill conditions)
    - Perform blocking I/O in on_tick (must be <10ms)
    - Spawn threads, processes, or async tasks (the orchestrator manages concurrency)
    - Import or use the broker adapter, OMS, or position tracker modules

    These restrictions are enforced by:
    1. Process isolation (strategy runs in its own OS process)
    2. The strategy receives only the objects listed in the hook signatures
    3. Redis credentials are not passed to strategy processes
    4. Code review at registration time
    """

    strategy_id: str
    config: StrategyConfig

    @abstractmethod
    def on_init(self, config: StrategyConfig) -> None:
        """
        Called once when the strategy process starts, before any ticks are delivered.

        Use this to:
        - Store config reference
        - Initialize indicators, moving averages, counters
        - Set up internal data structures
        - Log strategy parameters for audit

        This is called in the main thread of the strategy process. It may take
        up to 5 seconds. If on_init takes longer than 5 seconds, the Strategy
        Manager logs a WARNING. If it takes longer than 30 seconds, the strategy
        process is killed and marked DISABLED.

        Args:
            config: The full StrategyConfig for this strategy, loaded from
                    strategies.yaml. Includes strategy-specific params in
                    config.params dict.

        Returns:
            None

        Raises:
            Any exception from on_init is caught by the orchestrator. The strategy
            is marked DISABLED and a Telegram CRITICAL is sent.
        """
        ...

    @abstractmethod
    def on_tick(self, tick: Tick) -> StrategySignal | None:
        """
        Called on every tick from the strategy's subscribed streams.

        PERFORMANCE CONTRACT: This method MUST complete in <10ms. It runs in the
        async event loop — blocking here blocks tick ingestion for THIS strategy.
        Other strategies are unaffected (separate processes).

        If on_tick takes >10ms, a WARNING is logged. If it takes >100ms, the tick
        is still processed but the strategy is flagged for performance review.
        Sustained >100ms (10 consecutive ticks): strategy is SUPPRESSED with
        reason "performance_degraded".

        Common use cases:
        - Update last price / VWAP accumulator
        - Check threshold crossings that don't need bar aggregation
        - S2: check 15:15 entry trigger (time-based, not bar-based)

        Most strategies return None from on_tick and do their work in on_bar_close.

        Args:
            tick: A single tick from one of the subscribed streams. The symbol
                  field identifies which instrument. The strategy may receive
                  ticks for multiple symbols if it subscribes to multiple streams.

        Returns:
            StrategySignal if the strategy wants to enter or exit a position.
            None if no action needed.
            A strategy returning a signal does NOT guarantee an order will be placed —
            the signal still passes through deduplication, allocation, risk gate.
        """
        ...

    @abstractmethod
    def on_bar_close(self, bar: Bar) -> StrategySignal | None:
        """
        Called when a bar closes (bar_interval_s has elapsed since bar open).

        PERFORMANCE CONTRACT: This method runs in a ThreadPoolExecutor, NOT in the
        async event loop. It does NOT block tick ingestion. It may take up to 500ms.

        Timeouts:
        - >500ms: WARNING logged
        - >2000ms: the call is killed (concurrent.futures.Future cancelled),
          WARNING logged, the bar's signal is lost. The strategy continues with
          the next bar.
        - >2000ms on 3 consecutive bars: strategy is SUPPRESSED with reason
          "on_bar_close_timeout".

        This is where most strategies do their work: compute indicators, check
        entry/exit conditions, generate signals.

        The bar includes gap detection metadata. The strategy decides whether to
        suppress signals when has_gap=True. Some strategies (momentum) are robust
        to gaps; others (mean reversion on VWAP) should suppress.

        Args:
            bar: Completed OHLCV bar with metadata. Includes:
                 - bar.has_gap: True if any consecutive tick gap > 2x expected interval
                 - bar.max_tick_gap_ms: longest gap in the bar
                 - bar.vwap: volume-weighted average price for the bar
                 - bar.tick_count: number of ticks in the bar

        Returns:
            StrategySignal if the strategy wants to enter or exit a position.
            None if no action needed.
        """
        ...

    @abstractmethod
    def on_fill(self, fill: FillNotification) -> None:
        """
        Called when an order placed by this strategy is filled, partially filled,
        rejected, cancelled, or times out.

        Use this to update internal state: position tracking, entry price memory,
        PnL counters, trade count for the day.

        This runs in the async event loop. MUST complete in <10ms. It is a
        notification — the strategy cannot reject or modify the fill.

        Args:
            fill: Complete fill information including:
                  - fill.fill_status: one of FULL_FILL, PARTIAL_FILL, REJECTED,
                    CANCELLED, TIMEOUT
                  - fill.filled_qty: how many units were filled (0 for REJECTED)
                  - fill.avg_fill_price: weighted average fill price
                  - fill.sl_order_id: the paired SL order ID (for reference only —
                    the strategy cannot modify SL orders)

        Returns:
            None. The strategy cannot reject or modify fills.

        Note:
            PARTIAL_FILL means the entry order timed out with only partial quantity
            filled. The SL has already been adjusted to match filled_qty by the OMS.
            The strategy should update its internal position to reflect the partial.
        """
        ...

    @abstractmethod
    def on_position_update(self, position: PositionUpdate) -> None:
        """
        Called periodically (every 5 seconds) and on every fill with the strategy's
        current position state.

        This is the strategy's view of its own position. The Position Tracker is
        the source of truth — this is a read-only notification.

        Use this to:
        - Track whether you're FLAT, LONG, or SHORT
        - Read unrealized PnL for dynamic exit decisions
        - Verify position matches expectations (defensive check)

        Runs in the async event loop. MUST complete in <5ms (it's a frequent call).

        Args:
            position: Current position state including direction, quantity,
                      unrealized PnL, and SL info. When FLAT, quantity=0 and
                      direction="FLAT".

        Returns:
            None. Read-only notification.
        """
        ...

    @abstractmethod
    def on_kill(self) -> None:
        """
        Called when the Risk Manager kills this strategy due to a kill condition
        firing (e.g., Sharpe below threshold, consecutive losses exceeded, drawdown
        limit hit).

        After on_kill():
        - The strategy will receive NO more ticks or bars
        - The strategy CANNOT emit signals
        - Existing positions are NOT closed by on_kill — they exit via their
          existing server-side SL orders or normal EOD flatten
        - The strategy process is terminated after on_kill returns

        Use this to:
        - Log the kill event with current state for post-mortem
        - Clean up any internal resources
        - Save diagnostic state

        This runs in the main thread. May take up to 5 seconds.

        Args:
            None. The kill reason is logged by the Risk Manager, not passed to
            the strategy. The strategy should not attempt to override or delay
            the kill.

        Returns:
            None.
        """
        ...

    @abstractmethod
    def on_shutdown(self) -> None:
        """
        Called during graceful system shutdown (EOD or operator-initiated).

        Unlike on_kill (which is punitive), on_shutdown is orderly. The strategy
        has already been suppressed (no new signals). Existing positions are being
        flattened by the OMS.

        Use this to:
        - Flush any internal buffers
        - Log final state
        - Return cleanly

        This runs in the main thread. May take up to 5 seconds. After 5 seconds,
        SIGTERM is sent. After 10 seconds, SIGKILL.

        Args:
            None.

        Returns:
            None.
        """
        ...

    def get_state(self) -> dict:
        """
        Serialize the strategy's internal state for crash recovery.

        Called by the orchestrator:
        - Every 60 seconds (periodic snapshot)
        - On graceful shutdown
        - On parent process death detection (via parent_watchdog)

        The returned dict is serialized to JSON and stored in Redis
        under key STATE:strategy:{strategy_id}.

        The dict must be JSON-serializable (no numpy arrays, no datetime objects —
        convert to epoch ms). Include everything needed to resume: indicator state,
        bar accumulation buffers, position memory, trade counts.

        Default implementation returns empty dict. Override if your strategy has
        meaningful state to preserve across restarts.

        Returns:
            dict: JSON-serializable state. Empty dict if no state to save.
        """
        return {}

    def restore_state(self, state: dict) -> None:
        """
        Restore internal state after a crash or restart.

        Called by the orchestrator after on_init() if saved state exists in Redis.
        The state dict is exactly what get_state() returned in the previous session.

        If restore_state raises an exception, the strategy starts fresh (as if no
        saved state existed). A WARNING is logged.

        Args:
            state: The dict previously returned by get_state().

        Returns:
            None.
        """
        pass
```

---

### Strategy Configuration Schema

Every strategy is fully configured via a single YAML block in `config/strategies.yaml`. No code changes are needed to adjust parameters — only YAML edits.

**Complete example for S1 (ORB):**

```yaml
strategies:
  S1:
    # ---- Identity & Registration ----
    strategy_id: "S1"
    class_path: "live.strategies.s1_orb.S1OrbStrategy"
    enabled: true

    # ---- Data Requirements ----
    subscriptions:
      - "STREAM:TICK:SPOT"           # NIFTY50 index ticks
    bar_interval_s: 1800              # 30-minute bars for ORB range computation

    # ---- Instrument Preferences ----
    instrument_type: "CE"             # trades NIFTY call options
    expiry_preference: "MONTHLY"      # monthly expiry
    direction_hint: "LONG_ONLY"       # ORB is a long-only breakout

    # ---- Capital ----
    capital_weight: 0.20              # 20% static weight (used days 1-40)
    kelly_override: null              # use system-computed Kelly

    # ---- Fill Parameters ----
    fill_params:
      reprice_interval_s: 5           # reprice every 5 seconds if not filled
      max_patience_s: 15              # abandon after 15 seconds
      pricing_mode: "AGGRESSIVE"      # ask+1tick for buy, bid-1tick for sell

    # ---- Risk Parameters ----
    risk_params:
      stop_points: 50                 # SL at entry - 50 points
      max_position_lots: 10           # maximum 10 lots (650 qty at 65/lot)
      max_daily_trades: 2             # ORB fires at most once, but allow 2 for re-entry
      max_daily_loss: 50000           # ₹50K daily loss auto-suppress

    # ---- Kill Condition ----
    kill_condition:
      metric: "sharpe_60d"            # after-cost Sharpe over rolling 60 days
      threshold: 0.5                  # kill if < 0.5
      lookback_days: 60
      action: "stop_new_entries"      # don't flatten, just stop new entries

    # ---- Schedule ----
    schedule:
      active_days: ["MON", "TUE", "WED", "THU", "FRI"]  # all trading days
      active_from_ist: "09:20"        # skip first 5 minutes
      active_until_ist: "14:30"       # no new entries after 14:30

    # ---- VIX Filter ----
    vix_suppress_above: 25.0          # suppress if VIX > 25 (too volatile for breakout)
    vix_suppress_below: null          # no lower bound

    # ---- Strategy-Specific Parameters (opaque to system) ----
    params:
      orb_range_bars: 1               # first N 30-min bars define the ORB range
      breakout_buffer_pct: 0.5        # require 0.5% above high to trigger
      target_multiple: 2.0            # target = entry + 2 * ORB range
      use_volume_confirmation: true   # require volume > 1.5x average
      min_orb_range_points: 30        # skip if ORB range < 30 points (choppy)
      max_orb_range_points: 200       # skip if ORB range > 200 points (gap day)
```

**Complete example for S5 (Expiry Day) — showing schedule restriction:**

```yaml
  S5:
    strategy_id: "S5"
    class_path: "live.strategies.s5_expiry_day.S5ExpiryDayStrategy"
    enabled: true

    subscriptions:
      - "STREAM:TICK:SPOT"
    bar_interval_s: 60                # 1-minute bars for fast 0-DTE

    instrument_type: "CE"             # trades weekly NIFTY options
    expiry_preference: "WEEKLY"       # weekly expiry (Tuesday)
    direction_hint: "BOTH"            # can go long or short

    capital_weight: 0.10
    kelly_override: null

    fill_params:
      reprice_interval_s: 3
      max_patience_s: 10
      pricing_mode: "AGGRESSIVE"

    risk_params:
      stop_points: 30
      max_position_lots: 5
      max_daily_trades: 6
      max_daily_loss: 30000

    kill_condition:
      metric: "consecutive_expiry_losses"
      threshold: 3
      lookback_days: 30
      action: "stop_new_entries"

    schedule:
      active_days: ["TUE"]            # ONLY on Tuesdays (weekly expiry)
      active_from_ist: "09:20"
      active_until_ist: "15:00"       # no new entries in last 30 min of expiry

    vix_suppress_above: 30.0
    vix_suppress_below: null

    params:
      min_premium: 5.0                # don't trade options < ₹5 (illiquid)
      max_iv_percentile: 90           # skip if IV > 90th percentile
      gamma_scalp_mode: false
```

**What the system reads vs what the strategy reads:**

| Config Field | Read By | Purpose |
|-------------|---------|---------|
| `strategy_id`, `class_path`, `enabled` | Strategy Manager | Discovery and lifecycle |
| `subscriptions`, `bar_interval_s` | Orchestrator | Consumer group creation, BarBuilder init |
| `instrument_type`, `expiry_preference`, `direction_hint` | Signal Router, Instrument Resolver | Resolution and validation |
| `capital_weight`, `kelly_override` | Capital Allocator | Sizing |
| `fill_params` | OMS | Fill management loop |
| `risk_params` | Risk Manager | Pre-trade checks, SL computation |
| `kill_condition` | Risk Manager | Post-trade monitoring |
| `schedule` | Strategy Manager | Suppress/unsuppress transitions |
| `vix_suppress_above/below` | Strategy Manager + Risk Manager | VIX-based suppression |
| `params` | Strategy (on_init) | Strategy-specific logic (opaque to system) |

---

### Strategy Isolation Guarantees

Each strategy runs in its own OS process. The following isolation properties are guaranteed by the architecture.

#### Crash Containment

A strategy crash (unhandled exception, segfault, OOM kill) does NOT affect other strategies or the system.

**Mechanism:**
- Each strategy is a `multiprocessing.Process` forked by the Strategy Manager
- The Strategy Manager monitors each child via heartbeat (Redis key `HEALTH:strategy:{sid}`, TTL 15s, updated every 5s)
- If a child process dies, `multiprocessing.Process.exitcode` is checked:
  - Exit code 0: normal shutdown (expected during system shutdown)
  - Exit code 1: controlled error exit (parent died, fatal config error)
  - Exit code < 0: killed by signal (e.g., -9 = SIGKILL, -11 = SIGSEGV)
  - None: still running (shouldn't happen if detected via heartbeat)

**On crash detection:**
1. Log the crash with exit code and last known state
2. Save any state from Redis `STATE:strategy:{sid}` (may be up to 60s stale)
3. Restart the strategy process (up to 3 restarts per session)
4. On restart: `on_init()` → `restore_state()` → resume tick consumption
5. After 3 restarts: mark strategy KILLED, Telegram CRITICAL, do not restart
6. Existing positions protected by server-side SL orders on the broker

```python
class StrategyManager:
    MAX_RESTARTS_PER_SESSION = 3
    _restart_counts: dict[str, int]   # strategy_id → restart count

    async def monitor_strategy_health(self) -> None:
        """Runs every 5 seconds in the orchestrator's event loop."""
        for sid, entry in self._strategies.items():
            if entry.state not in (StrategyState.RUNNING, StrategyState.SUPPRESSED):
                continue

            # Check heartbeat
            heartbeat = await redis.get(f"HEALTH:strategy:{sid}")
            if heartbeat is None:
                age_ms = None
            else:
                age_ms = now_ms() - int(heartbeat)

            # Check process alive
            if entry.process is not None and not entry.process.is_alive():
                exit_code = entry.process.exitcode
                logger.error("strategy_process_died",
                           strategy_id=sid,
                           exit_code=exit_code,
                           restarts=self._restart_counts.get(sid, 0))

                if self._restart_counts.get(sid, 0) >= self.MAX_RESTARTS_PER_SESSION:
                    logger.critical("strategy_max_restarts_exceeded", strategy_id=sid)
                    entry.transition(StrategyState.KILLED)
                    entry.kill_reason = f"max_restarts_exceeded (exit_code={exit_code})"
                    await telegram.send(CRITICAL,
                        f"Strategy {sid} killed: {self.MAX_RESTARTS_PER_SESSION} restarts exceeded")
                else:
                    self._restart_counts[sid] = self._restart_counts.get(sid, 0) + 1
                    await self._restart_strategy(sid)

            # Check stale heartbeat (process alive but not responding)
            elif age_ms is not None and age_ms > 30_000:
                logger.warning("strategy_heartbeat_stale",
                             strategy_id=sid, age_ms=age_ms)
                # Process is alive but hung — SIGTERM, wait 5s, SIGKILL
                entry.process.terminate()
                await asyncio.sleep(5)
                if entry.process.is_alive():
                    entry.process.kill()
                # Will be detected as dead on next monitor cycle
```

#### Resource Limits

A strategy cannot exhaust shared resources (CPU, memory, OPS budget).

| Resource | Limit | Enforcement |
|----------|-------|-------------|
| CPU time per `on_tick` | 10ms | Measured via `time.perf_counter()`. WARNING at >10ms, performance flag at >100ms. |
| CPU time per `on_bar_close` | 2000ms | `concurrent.futures.Future` with 2s timeout. Cancelled on timeout. |
| Memory per strategy process | 512MB | `resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, ...))` set in child process |
| Redis writes | 0 | Strategy processes do not have Redis write credentials. All writes go through the orchestrator via IPC. |
| OPS budget | 0 direct | Strategies never call the broker API. Only the OMS (in the main process) makes broker calls. |
| File system writes | Logging only | Strategy processes write to their own log file. No other file I/O. |
| Network access | None | Strategy processes do not open sockets. Market data arrives via Redis Streams (read by orchestrator, forwarded via pipe). |
| Signal rate | 1 per cooldown window | Signal deduplicator in Signal Router (see below) |

```python
import resource

def _apply_resource_limits():
    """Called in child process after fork, before on_init."""
    # Memory limit: 512MB virtual
    mem_limit = 512 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (mem_limit, mem_limit))

    # CPU time: no hard limit (monitored via heartbeat instead)
    # File descriptors: 64 (stdin, stdout, stderr, log file, Redis connection, pipe)
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
```

#### Execution Timeouts

| Hook | Timeout | On Timeout |
|------|---------|------------|
| `on_init` | 30s | Kill process, mark DISABLED, Telegram CRITICAL |
| `on_tick` | 100ms (hard), 10ms (warning) | WARNING at 10ms. 10 consecutive >100ms → SUPPRESS with "performance_degraded" |
| `on_bar_close` | 2000ms | Cancel Future, WARNING. 3 consecutive timeouts → SUPPRESS with "on_bar_close_timeout" |
| `on_fill` | 10ms (warning only) | WARNING. Fill notifications are non-critical — strategy can't block fill processing |
| `on_position_update` | 5ms (warning only) | WARNING. Position updates are non-critical |
| `on_kill` | 5s | SIGTERM after 5s. SIGKILL after 10s. |
| `on_shutdown` | 5s | SIGTERM after 5s. SIGKILL after 10s. |
| `get_state` | 1s | WARNING. If >1s, state snapshot is skipped for this cycle |

---

### BarBuilder

Each strategy has its own `BarBuilder` instance, initialized by the orchestrator with the strategy's `bar_interval_s`. The strategy never instantiates or configures the BarBuilder — it just receives completed `Bar` objects in `on_bar_close`.

```python
import itertools

class BarBuilder:
    """
    Aggregates ticks into OHLCV bars with gap detection and VWAP computation.

    One BarBuilder per strategy. Created by the orchestrator based on the
    strategy's bar_interval_s config.

    Bar boundaries are aligned to clock time, not to the first tick. A 300s
    (5-minute) bar covers 09:15:00-09:19:59, 09:20:00-09:24:59, etc.
    This ensures bars are consistent across restarts and across strategies
    with the same interval.

    Strategies with bar_interval_s=None (e.g., S2) do not get a BarBuilder.
    They receive ticks via on_tick only.
    """

    def __init__(self, interval_s: int, symbol: str):
        """
        Args:
            interval_s: Bar duration in seconds. Must be > 0.
            symbol: The primary symbol this bar builder tracks. Used in the
                    output Bar.symbol field.
        """
        if interval_s <= 0:
            raise ValueError(f"bar interval must be > 0, got {interval_s}")

        self.interval_s = interval_s
        self.symbol = symbol

        # Current bar accumulation state
        self._ticks: list[Tick] = []
        self._bar_open_ts: int = 0       # epoch ms — aligned to clock boundary
        self._bar_close_ts: int = 0      # epoch ms — bar_open + interval_s * 1000

        # OHLCV running state (avoids re-scanning _ticks list)
        self._open: float = 0.0
        self._high: float = float('-inf')
        self._low: float = float('inf')
        self._close: float = 0.0
        self._volume_start: int = 0      # cumulative volume at bar open
        self._last_volume: int = 0       # last seen cumulative volume
        self._vwap_numerator: float = 0.0  # Σ(price × tick_volume_delta)
        self._vwap_denominator: int = 0    # Σ(tick_volume_delta)
        self._tick_count: int = 0
        self._last_tick_ts: int = 0      # for gap detection
        self._max_tick_gap_ms: int = 0

        # Expected tick interval for gap detection (estimated from first N ticks)
        self._expected_tick_interval_ms: int = 0
        self._tick_intervals: list[int] = []  # first 100 inter-tick intervals
        self._calibrated: bool = False

    def add_tick(self, tick: Tick) -> Bar | None:
        """
        Add a tick. Returns a completed Bar when the bar interval closes, else None.

        Ticks are assumed to arrive in exchange_ts order (enforced by Redis Streams
        ordering). Out-of-order ticks (exchange_ts < last seen) are logged and
        discarded.

        Args:
            tick: The incoming tick.

        Returns:
            Bar if the tick closes the current bar window. None otherwise.
            When a bar is returned, the tick that triggered the close is included
            in the NEXT bar (it arrived after the close boundary).
        """
        completed_bar: Bar | None = None

        # Out-of-order check
        if tick.exchange_ts < self._last_tick_ts:
            logger.debug("tick_out_of_order",
                        symbol=tick.symbol,
                        tick_ts=tick.exchange_ts,
                        last_ts=self._last_tick_ts)
            return None

        # Determine which bar window this tick belongs to
        tick_bar_open = self._align_to_bar_boundary(tick.exchange_ts)

        # If this tick belongs to a new bar window, close the current bar first
        if self._tick_count > 0 and tick_bar_open > self._bar_open_ts:
            completed_bar = self._build_bar()
            self._reset()

        # If this is the first tick (or first tick of new bar), set bar boundaries
        if self._tick_count == 0:
            self._bar_open_ts = tick_bar_open
            self._bar_close_ts = tick_bar_open + self.interval_s * 1000
            self._open = tick.ltp
            self._high = tick.ltp
            self._low = tick.ltp
            self._volume_start = tick.volume

        # Update running OHLCV state
        self._high = max(self._high, tick.ltp)
        self._low = min(self._low, tick.ltp)
        self._close = tick.ltp

        # Volume delta (cumulative volume increases)
        volume_delta = max(0, tick.volume - self._last_volume) if self._tick_count > 0 else 0
        self._last_volume = tick.volume

        # VWAP accumulation
        if volume_delta > 0:
            self._vwap_numerator += tick.ltp * volume_delta
            self._vwap_denominator += volume_delta

        # Gap detection
        if self._tick_count > 0:
            gap_ms = tick.exchange_ts - self._last_tick_ts
            self._max_tick_gap_ms = max(self._max_tick_gap_ms, gap_ms)
            # Calibrate expected tick interval from first 100 inter-tick intervals
            if not self._calibrated:
                self._tick_intervals.append(gap_ms)
                if len(self._tick_intervals) >= 100:
                    self._expected_tick_interval_ms = int(
                        sorted(self._tick_intervals)[50]  # median
                    )
                    self._calibrated = True

        self._last_tick_ts = tick.exchange_ts
        self._tick_count += 1
        self._ticks.append(tick)

        return completed_bar

    def _align_to_bar_boundary(self, ts_ms: int) -> int:
        """Align a timestamp to the nearest bar boundary (floor).

        For a 300s (5-min) bar, ts 09:17:32 → 09:15:00.
        For a 1800s (30-min) bar, ts 09:42:15 → 09:30:00.

        Uses the start of the trading day (09:15:00 IST) as epoch to avoid
        bars spanning pre-open/continuous boundaries.
        """
        # Trading day epoch: 09:15:00 IST = 03:45:00 UTC
        day_start_utc_ms = self._get_trading_day_start_ms(ts_ms)
        elapsed_ms = ts_ms - day_start_utc_ms
        interval_ms = self.interval_s * 1000
        bar_index = elapsed_ms // interval_ms
        return day_start_utc_ms + bar_index * interval_ms

    def _get_trading_day_start_ms(self, ts_ms: int) -> int:
        """Get 09:15:00 IST of the trading day containing ts_ms."""
        # IST = UTC + 5:30
        ist_offset_ms = 5 * 3600_000 + 30 * 60_000
        ist_ms = ts_ms + ist_offset_ms
        # Floor to day start (00:00 IST)
        day_start_ist = (ist_ms // 86400_000) * 86400_000
        # 09:15 IST = day_start + 9h15m
        trading_start_ist = day_start_ist + 9 * 3600_000 + 15 * 60_000
        # Convert back to UTC
        return trading_start_ist - ist_offset_ms

    def _build_bar(self) -> Bar:
        """Build the completed bar from accumulated state."""
        has_gap = False
        if self._calibrated and self._expected_tick_interval_ms > 0:
            has_gap = self._max_tick_gap_ms > 2 * self._expected_tick_interval_ms

        vwap = (self._vwap_numerator / self._vwap_denominator
                if self._vwap_denominator > 0
                else (self._open + self._high + self._low + self._close) / 4)

        bar_volume = max(0, self._last_volume - self._volume_start)

        return Bar(
            symbol=self.symbol,
            interval_s=self.interval_s,
            open=self._open,
            high=self._high,
            low=self._low,
            close=self._close,
            volume=bar_volume,
            vwap=round(vwap, 2),
            bar_start_ts=self._bar_open_ts,
            bar_end_ts=self._bar_close_ts,
            tick_count=self._tick_count,
            has_gap=has_gap,
            max_tick_gap_ms=self._max_tick_gap_ms,
        )

    def _reset(self) -> None:
        """Reset accumulation state for the next bar."""
        self._ticks.clear()
        self._tick_count = 0
        self._high = float('-inf')
        self._low = float('inf')
        self._vwap_numerator = 0.0
        self._vwap_denominator = 0
        self._max_tick_gap_ms = 0
        # NOTE: _last_tick_ts, _last_volume, _calibrated are NOT reset —
        # they carry across bars for continuity

    def force_close(self) -> Bar | None:
        """Force-close the current bar (used at EOD or on strategy shutdown).

        Returns the in-progress bar if there are accumulated ticks, else None.
        """
        if self._tick_count == 0:
            return None
        bar = self._build_bar()
        self._reset()
        return bar
```

**Bar intervals per strategy (complete table):**

| Strategy | `bar_interval_s` | Rationale |
|----------|-----------------|-----------|
| S1 (ORB) | 1800 (30 min) | ORB range = first 30-minute bar's high-low |
| S2 (Overnight Futures) | `None` | Event-driven: 15:15 IST entry, 09:20 next-day exit. No bars needed. |
| S3 (VWAP Mean Reversion) | 300 (5 min) | 5-minute bars for VWAP deviation computation |
| S4 (Equity Momentum) | 86400 (daily) | Monthly rebalance. Only active on rebalance days. Bar = full trading day. |
| S5 (Expiry Day 0-DTE) | 60 (1 min) | 1-minute bars for fast 0-DTE scalping |
| S6 (Volatility Premium) | 300 (5 min) | 5-minute bars for VIX regime monitoring and entry timing |
| S7 (Statistical Pairs) | 300 (5 min) | 5-minute bars for spread monitoring and z-score computation |

**Tick gap handling policy:**

When `bar.has_gap = True`, the strategy receives the bar normally with the flag set. The architecture does NOT force signal suppression — each strategy decides:

| Strategy | Gap Policy | Rationale |
|----------|-----------|-----------|
| S1 (ORB) | Suppress if gap in first bar | ORB range distorted by missing ticks |
| S2 (Overnight) | Ignore gaps | Time-based trigger, not price-action dependent |
| S3 (VWAP MR) | Suppress | VWAP computation corrupted by gaps |
| S4 (Momentum) | Ignore gaps | Daily bars, momentum robust to intraday gaps |
| S5 (Expiry Day) | Suppress if gap > 5s | 0-DTE is time-sensitive, 5s gap = stale signal |
| S6 (Vol Premium) | Ignore gaps | VIX-based, not tick-sensitive |
| S7 (Pairs) | Suppress if gap > 10s | Spread computation needs synchronized prices |

---

### Signal Deduplication

The Signal Deduplicator sits in the Signal Router process (not in the strategy process). It prevents duplicate orders when a strategy emits multiple signals for the same trade within a short window.

**Why this happens:** Two ticks crossing the same threshold 50ms apart can each trigger `on_tick` → signal. The BarBuilder's `on_bar_close` is deterministic (one bar = one call), but `on_tick` can fire on every tick.

```python
class SignalDeduplicator:
    """
    Suppress duplicate signals from the same strategy for the same underlying
    and direction within a configurable cooldown window.

    Keyed on (strategy_id, direction, underlying). A signal is a duplicate if
    another signal with the same key was accepted within the cooldown window.

    The cooldown window is per-strategy because different strategies have
    different signal frequencies:
    - S1 fires once per day (ORB breakout) → long cooldown
    - S5 fires multiple times per expiry day → short cooldown
    - S2 fires once per day (15:15 entry) → very long cooldown

    ALSO: Each signal carries a UUID signal_id. The OMS tracks placed signal_ids
    and rejects any signal_id it has already processed. This is a second layer
    of dedup (idempotent placement) independent of the time-based dedup here.
    """

    def __init__(self):
        self._last_accepted: dict[tuple[str, str, str], int] = {}
        # Key: (strategy_id, direction, underlying) → epoch ms of last accepted signal

    # Per-strategy cooldown windows in milliseconds
    COOLDOWN_MS: dict[str, int] = {
        "S1": 300_000,    # 5 min — ORB fires once per day, but allow re-entry after
                          #          300s if first order was rejected/timeout
        "S2": 86_400_000, # 24 hours — overnight fires exactly once per day
        "S3": 60_000,     # 1 min — VWAP MR can fire multiple times per day,
                          #          but not within the same minute
        "S4": 2_592_000_000, # 30 days — monthly rebalance, one signal per stock per month
        "S5": 21_600_000,    # 6 hours — 0-DTE, wide enough to avoid re-entry same half-day
        "S6": 604_800_000,   # 7 days — patient vol selling, weekly cooldown
        "S7": 86_400_000,    # 24 hours — pairs spread, daily cooldown
    }

    # Fallback for strategies not in the map (e.g., new S8)
    DEFAULT_COOLDOWN_MS = 60_000  # 1 minute default

    def is_duplicate(self, signal: StrategySignal) -> bool:
        """
        Check if this signal is a duplicate of a recently accepted signal.

        Args:
            signal: The incoming signal to check.

        Returns:
            True if this signal should be suppressed (duplicate).
            False if this signal should be forwarded (not a duplicate).
        """
        key = (signal.strategy_id, signal.direction, signal.underlying)
        last_accepted_ts = self._last_accepted.get(key, 0)
        cooldown = self.COOLDOWN_MS.get(signal.strategy_id, self.DEFAULT_COOLDOWN_MS)

        if signal.signal_ts - last_accepted_ts < cooldown:
            logger.info("signal_deduplicated",
                       strategy_id=signal.strategy_id,
                       direction=signal.direction,
                       underlying=signal.underlying,
                       cooldown_remaining_ms=cooldown - (signal.signal_ts - last_accepted_ts))
            return True

        # Accept and record
        self._last_accepted[key] = signal.signal_ts
        return False

    def reset_for_strategy(self, strategy_id: str) -> None:
        """Clear dedup state for a strategy (called on strategy restart)."""
        keys_to_remove = [k for k in self._last_accepted if k[0] == strategy_id]
        for k in keys_to_remove:
            del self._last_accepted[k]

    def get_cooldown_for(self, strategy_id: str) -> int:
        """Return cooldown in ms for a given strategy. Used by config validation."""
        return self.COOLDOWN_MS.get(strategy_id, self.DEFAULT_COOLDOWN_MS)
```

**Second dedup layer (OMS-side):** The OMS maintains an in-memory `set[str]` of placed `signal_id`s. Before placing any order, it checks `signal.signal_id not in placed_signal_ids`. This catches any duplicate that slips past the time-based dedup (e.g., if the Signal Router restarts and loses in-memory state).

---

### Consumer Group Creation and Management

Redis Streams consumer groups are the mechanism by which strategy processes read market data. Each strategy has its own consumer group on each stream it subscribes to. Consumer groups track read offsets — if a strategy process restarts, it resumes from its last acknowledged message.

#### Creation

Consumer groups are created by the Orchestrator in **Phase 8 of startup**, BEFORE any strategy process is spawned. This eliminates a race condition where a strategy process tries to read from a non-existent group.

```python
class ConsumerGroupManager:
    """
    Manages Redis Streams consumer groups for all strategy processes.

    Created groups:
    - One group per (strategy_id, stream) pair
    - Group name format: "strategy_{strategy_id}"
    - Consumer name within group: same as strategy_id
    """

    def __init__(self, redis_client):
        self._redis = redis_client

    async def create_all_groups(self, strategies: dict[str, StrategyEntry]) -> None:
        """
        Create consumer groups for all enabled strategies.

        Called once during startup Phase 8. Idempotent — if a group already
        exists (from a previous session), this is a no-op for that group.

        Uses "$" as the starting ID, meaning strategies only see messages
        published AFTER the group was created. This prevents strategies from
        processing stale ticks from a previous session on restart.

        Args:
            strategies: All enabled strategy entries with their configs.
        """
        for sid, entry in strategies.items():
            if entry.state == StrategyState.DISABLED:
                continue

            group_name = f"strategy_{sid}"

            for stream in entry.config.subscriptions:
                try:
                    await self._redis.xgroup_create(
                        name=stream,
                        groupname=group_name,
                        id="$",          # only new messages from this point forward
                        mkstream=True    # create stream if it doesn't exist yet
                    )
                    logger.info("consumer_group_created",
                              stream=stream, group=group_name, start_id="$")
                except Exception as e:
                    if "BUSYGROUP" in str(e):
                        # Group already exists (previous session). This is fine.
                        # The group will resume from its last ack'd offset.
                        logger.info("consumer_group_exists",
                                  stream=stream, group=group_name)
                    else:
                        raise

    async def destroy_group(self, strategy_id: str, streams: list[str]) -> None:
        """
        Destroy consumer groups for a strategy. Called when a strategy is
        permanently disabled (not on restart — restart preserves offset).

        Args:
            strategy_id: The strategy whose groups to destroy.
            streams: The streams the strategy was subscribed to.
        """
        group_name = f"strategy_{strategy_id}"
        for stream in streams:
            try:
                await self._redis.xgroup_destroy(stream, group_name)
                logger.info("consumer_group_destroyed",
                          stream=stream, group=group_name)
            except Exception:
                pass  # group may not exist, that's fine

    async def get_group_lag(self, strategy_id: str, stream: str) -> int:
        """
        Get the number of unread messages in a consumer group.
        Used by monitoring to detect a slow strategy falling behind.

        Returns:
            Number of unprocessed messages. 0 = caught up.
        """
        group_name = f"strategy_{strategy_id}"
        info = await self._redis.xinfo_groups(stream)
        for group in info:
            if group["name"] == group_name:
                return group.get("lag", 0)
        return -1  # group not found

    async def ack_message(self, stream: str, group_name: str, message_id: str) -> None:
        """Acknowledge a message as processed."""
        await self._redis.xack(stream, group_name, message_id)
```

#### Stream Trimming

Tick streams are trimmed to prevent unbounded memory growth:

```python
STREAM_TRIM_CONFIG = {
    "STREAM:TICK:SPOT": 60_000,           # ~5 min at 200 ticks/s
    "STREAM:TICK:FUT:*": 60_000,
    "STREAM:TICK:OPT:*": 120_000,         # higher — more instruments
    "STREAM:TICK:EQ:*": 30_000,           # fewer ticks for equities
}
```

Trimming is done by the Data Ingester via `XTRIM MAXLEN ~{threshold}` (approximate trim for performance). The `~` prefix allows Redis to trim in blocks rather than per-message.

---

### Strategy Process Event Loop

Each strategy runs in its own OS process. The event loop is the core of the strategy runtime — it reads ticks, feeds them to the BarBuilder, calls strategy hooks, publishes signals, and maintains health.

```python
import asyncio
import concurrent.futures
import multiprocessing
import os
import signal
import sys
import time

async def strategy_main(strategy: LiveStrategy, config: StrategyConfig):
    """
    Main entry point for a strategy process.

    This function runs in a child process forked by the Strategy Manager.
    It creates the async event loop, sets up all tasks, and runs until
    shutdown or crash.

    Architecture:
    - tick_reader: reads from Redis Streams, calls on_tick, feeds BarBuilder
    - bar_executor: runs on_bar_close in a ThreadPoolExecutor (non-blocking)
    - heartbeat: publishes health to Redis every 5s
    - parent_watchdog: detects orchestrator death
    - position_poller: reads own position from Redis every 5s, calls on_position_update
    - fill_listener: reads fill notifications from Redis, calls on_fill
    """

    # Apply resource limits BEFORE any strategy code runs
    _apply_resource_limits()

    # Initialize strategy
    try:
        strategy.on_init(config)
    except Exception:
        logger.exception("strategy_on_init_failed", strategy_id=config.strategy_id)
        sys.exit(1)

    # Restore state if available
    saved_state = await redis.get(f"STATE:strategy:{config.strategy_id}")
    if saved_state:
        try:
            strategy.restore_state(orjson.loads(saved_state))
            logger.info("strategy_state_restored", strategy_id=config.strategy_id)
        except Exception:
            logger.warning("strategy_state_restore_failed", strategy_id=config.strategy_id)
            # Continue without restored state — strategy starts fresh

    # Set up components
    group_name = f"strategy_{config.strategy_id}"
    bar_builder: BarBuilder | None = None
    if config.bar_interval_s is not None:
        primary_symbol = _extract_primary_symbol(config.subscriptions)
        bar_builder = BarBuilder(config.bar_interval_s, primary_symbol)

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix=f"bar_{config.strategy_id}"
    )

    # Track on_tick performance
    tick_slow_count = 0
    TICK_SLOW_THRESHOLD_MS = 100
    TICK_SLOW_SUPPRESS_COUNT = 10

    # Track on_bar_close timeouts
    bar_timeout_count = 0
    BAR_TIMEOUT_SUPPRESS_COUNT = 3

    async def tick_reader():
        """
        Read ticks from Redis Streams. This is the hot path.

        NEVER blocks on strategy computation — on_bar_close runs in a
        thread executor. on_tick runs inline but must be <10ms.

        Uses XREADGROUP with block=50ms. This means the event loop yields
        every 50ms even when no ticks arrive, allowing other tasks (heartbeat,
        watchdog) to run.
        """
        nonlocal tick_slow_count, bar_timeout_count

        while True:
            entries = await redis.xreadgroup(
                groupname=group_name,
                consumername=config.strategy_id,
                streams={s: ">" for s in config.subscriptions},
                count=100,      # batch up to 100 messages per read
                block=50        # block max 50ms (yields event loop)
            )

            if entries is None:
                continue  # timeout, no new messages

            for stream_name, messages in entries:
                for msg_id, data in messages:
                    tick = Tick.model_validate_json(data[b"tick"])

                    # ---- on_tick (fast path) ----
                    tick_start = time.perf_counter_ns()

                    signal = strategy.on_tick(tick)

                    tick_elapsed_ms = (time.perf_counter_ns() - tick_start) / 1_000_000
                    if tick_elapsed_ms > TICK_SLOW_THRESHOLD_MS:
                        tick_slow_count += 1
                        logger.warning("on_tick_slow",
                                     strategy_id=config.strategy_id,
                                     elapsed_ms=round(tick_elapsed_ms, 1),
                                     consecutive_slow=tick_slow_count)
                        if tick_slow_count >= TICK_SLOW_SUPPRESS_COUNT:
                            await redis.set(
                                f"SUPPRESS:strategy:{config.strategy_id}",
                                "performance_degraded")
                            logger.error("strategy_suppressed_performance",
                                       strategy_id=config.strategy_id)
                            tick_slow_count = 0
                    elif tick_elapsed_ms > 10:
                        logger.debug("on_tick_warning",
                                   strategy_id=config.strategy_id,
                                   elapsed_ms=round(tick_elapsed_ms, 1))
                    else:
                        tick_slow_count = 0  # reset on good tick

                    if signal is not None:
                        # Check suppression
                        suppressed = await redis.exists(
                            f"SUPPRESS:strategy:{config.strategy_id}")
                        if not suppressed:
                            await publish_signal(signal)
                        else:
                            logger.debug("signal_suppressed",
                                       strategy_id=config.strategy_id)

                    # ---- Bar aggregation ----
                    if bar_builder is not None:
                        bar = bar_builder.add_tick(tick)
                        if bar is not None:
                            # Run on_bar_close in thread executor with timeout
                            try:
                                loop = asyncio.get_event_loop()
                                future = loop.run_in_executor(
                                    executor, strategy.on_bar_close, bar)
                                bar_signal = await asyncio.wait_for(
                                    future, timeout=2.0)
                                bar_timeout_count = 0  # reset on success

                                if bar_signal is not None:
                                    suppressed = await redis.exists(
                                        f"SUPPRESS:strategy:{config.strategy_id}")
                                    if not suppressed:
                                        await publish_signal(bar_signal)
                            except asyncio.TimeoutError:
                                bar_timeout_count += 1
                                logger.warning("on_bar_close_timeout",
                                             strategy_id=config.strategy_id,
                                             timeout_s=2.0,
                                             consecutive_timeouts=bar_timeout_count)
                                if bar_timeout_count >= BAR_TIMEOUT_SUPPRESS_COUNT:
                                    await redis.set(
                                        f"SUPPRESS:strategy:{config.strategy_id}",
                                        "on_bar_close_timeout")
                                    logger.error(
                                        "strategy_suppressed_bar_timeout",
                                        strategy_id=config.strategy_id)
                                    bar_timeout_count = 0
                            except Exception:
                                logger.exception("on_bar_close_error",
                                               strategy_id=config.strategy_id)

                    # Acknowledge message AFTER processing
                    await redis.xack(stream_name, group_name, msg_id)

    async def heartbeat():
        """Publish health status to Redis every 5 seconds."""
        while True:
            await redis.set(
                f"HEALTH:strategy:{config.strategy_id}",
                str(int(time.time() * 1000)),
                ex=15  # TTL 15s — if not refreshed, considered dead
            )
            await asyncio.sleep(5)

    async def parent_watchdog():
        """
        Detect orchestrator (parent process) death.

        If the parent dies, save state and exit. Without the parent:
        - No more fill notifications
        - No more position updates
        - No one to restart us if we crash
        - Server-side SL orders protect positions
        """
        parent_pid = os.getppid()
        while True:
            await asyncio.sleep(5)
            try:
                os.kill(parent_pid, 0)  # signal 0 = existence check
            except OSError:
                logger.critical("parent_process_died",
                              strategy_id=config.strategy_id,
                              parent_pid=parent_pid)
                # Save state before exit
                try:
                    state = strategy.get_state()
                    await redis.set(
                        f"STATE:strategy:{config.strategy_id}",
                        orjson.dumps(state),
                        ex=3600  # expire in 1 hour (stale state is worse than no state)
                    )
                except Exception:
                    pass
                sys.exit(1)

    async def state_saver():
        """Periodically save strategy state to Redis for crash recovery."""
        while True:
            await asyncio.sleep(60)
            try:
                state_start = time.perf_counter_ns()
                state = strategy.get_state()
                state_elapsed_ms = (time.perf_counter_ns() - state_start) / 1_000_000
                if state_elapsed_ms > 1000:
                    logger.warning("get_state_slow",
                                 strategy_id=config.strategy_id,
                                 elapsed_ms=round(state_elapsed_ms, 1))
                else:
                    await redis.set(
                        f"STATE:strategy:{config.strategy_id}",
                        orjson.dumps(state),
                        ex=3600
                    )
            except Exception:
                logger.exception("state_save_failed",
                               strategy_id=config.strategy_id)

    async def position_poller():
        """
        Read own position from Redis every 5 seconds, call on_position_update.

        The Position Tracker (in the main process) is the sole writer of
        POSITION:strategy:{sid} keys. This poller reads them.
        """
        while True:
            await asyncio.sleep(5)
            try:
                pos_data = await redis.get(
                    f"POSITION:strategy:{config.strategy_id}")
                if pos_data:
                    position = PositionUpdate.model_validate_json(pos_data)
                    strategy.on_position_update(position)
            except Exception:
                logger.debug("position_poll_error",
                           strategy_id=config.strategy_id)

    async def fill_listener():
        """
        Listen for fill notifications on a Redis pubsub channel.

        The OMS publishes fill notifications to channel FILL:{strategy_id}.
        This listener receives them and calls strategy.on_fill().
        """
        pubsub = redis.pubsub()
        await pubsub.subscribe(f"FILL:{config.strategy_id}")

        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=1.0)
            if message and message["type"] == "message":
                try:
                    fill = FillNotification.model_validate_json(message["data"])
                    strategy.on_fill(fill)
                except Exception:
                    logger.exception("on_fill_error",
                                   strategy_id=config.strategy_id)

    # ---- Signal publication ----

    async def publish_signal(sig: StrategySignal) -> None:
        """Publish a strategy signal to the signal stream."""
        await redis.xadd(
            "STREAM:SIGNAL",
            {"signal": sig.model_dump_json()},
        )
        logger.info("signal_published",
                   strategy_id=sig.strategy_id,
                   signal_id=sig.signal_id,
                   direction=sig.direction,
                   underlying=sig.underlying)

    # ---- Task lifecycle with restart-on-error ----

    async def _restart_on_error(coro_fn, name: str):
        """
        Wrap an async coroutine so unhandled exceptions restart it instead
        of killing the entire strategy process.

        This replaces bare asyncio.gather which would cancel all tasks if one
        throws. Each task is independent — tick_reader crashing should not
        kill the heartbeat, and vice versa.

        Restart backoff: 1s, 2s, 4s, 8s, max 30s. Resets to 1s after 60s
        of successful running.
        """
        backoff_s = 1
        max_backoff_s = 30
        last_success_ts = time.time()

        while True:
            try:
                await coro_fn()
            except asyncio.CancelledError:
                logger.info(f"task_{name}_cancelled",
                          strategy_id=config.strategy_id)
                raise  # propagate cancellation (shutdown)
            except Exception:
                logger.exception(f"task_{name}_crashed_restarting",
                               strategy_id=config.strategy_id,
                               backoff_s=backoff_s)

                # Reset backoff if task ran successfully for >60s
                if time.time() - last_success_ts > 60:
                    backoff_s = 1

                await asyncio.sleep(backoff_s)
                backoff_s = min(backoff_s * 2, max_backoff_s)
                last_success_ts = time.time()

    # ---- Register shutdown handler ----

    loop = asyncio.get_event_loop()
    shutdown_event = asyncio.Event()

    def _handle_sigterm(signum, frame):
        logger.info("strategy_sigterm_received", strategy_id=config.strategy_id)
        strategy.on_shutdown()
        # Save final state
        try:
            state = strategy.get_state()
            # Sync Redis call (we're in signal handler, can't await)
            # Use a thread to save state
        except Exception:
            pass
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # ---- Launch all tasks ----

    tasks = [
        asyncio.create_task(
            _restart_on_error(tick_reader, "tick_reader"),
            name=f"{config.strategy_id}_tick_reader"
        ),
        asyncio.create_task(
            _restart_on_error(heartbeat, "heartbeat"),
            name=f"{config.strategy_id}_heartbeat"
        ),
        asyncio.create_task(
            _restart_on_error(parent_watchdog, "parent_watchdog"),
            name=f"{config.strategy_id}_parent_watchdog"
        ),
        asyncio.create_task(
            _restart_on_error(state_saver, "state_saver"),
            name=f"{config.strategy_id}_state_saver"
        ),
        asyncio.create_task(
            _restart_on_error(position_poller, "position_poller"),
            name=f"{config.strategy_id}_position_poller"
        ),
        asyncio.create_task(
            _restart_on_error(fill_listener, "fill_listener"),
            name=f"{config.strategy_id}_fill_listener"
        ),
    ]

    # Wait for shutdown signal or all tasks to complete
    done, pending = await asyncio.wait(
        tasks + [asyncio.create_task(shutdown_event.wait())],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # If shutdown_event was set, cancel all other tasks gracefully
    for task in pending:
        task.cancel()

    # Wait for cancellation to propagate
    await asyncio.gather(*pending, return_exceptions=True)

    # Final state save
    try:
        state = strategy.get_state()
        await redis.set(
            f"STATE:strategy:{config.strategy_id}",
            orjson.dumps(state),
            ex=3600
        )
    except Exception:
        pass

    logger.info("strategy_process_exiting", strategy_id=config.strategy_id)


def _extract_primary_symbol(subscriptions: list[str]) -> str:
    """
    Extract the primary symbol from subscription list for BarBuilder.

    STREAM:TICK:SPOT → "NIFTY50-INDEX"
    STREAM:TICK:FUT:NIFTY → "NIFTY-FUT"
    STREAM:TICK:OPT:NIFTY → "NIFTY-OPT"
    STREAM:TICK:EQ:RELIANCE → "RELIANCE"
    """
    stream = subscriptions[0]
    parts = stream.split(":")
    if parts[2] == "SPOT":
        return "NIFTY50-INDEX"
    elif len(parts) >= 4:
        return f"{parts[3]}-{parts[2]}"
    return "UNKNOWN"
```

---

### Cross-Strategy Awareness

Strategies are isolated by default. However, two specific cross-strategy interactions are architecturally supported:

#### S1 ↔ S3 Suppression

S1 (ORB breakout) and S3 (VWAP mean reversion) trade opposite regimes on the same underlying (NIFTY). When S1 has an active position (breakout in progress), S3 should not counter-trade the breakout.

**Mechanism:**

The Position Tracker writes per-strategy position state to Redis:
```
POSITION:strategy:S1 → {"direction": "LONG", "quantity": 650, ...}
POSITION:strategy:S3 → {"direction": "FLAT", "quantity": 0, ...}
```

S3 reads `POSITION:strategy:S1` in its `on_bar_close`:
```python
# Inside S3's on_bar_close:
s1_position = self._s1_position  # populated by on_position_update poller

if s1_position and s1_position.direction != "FLAT":
    # S1 has an active breakout position — suppress mean reversion signals
    # that would counter-trade the breakout
    if self._would_counter_trade(s1_position.direction, my_signal_direction):
        logger.info("s3_suppressed_by_s1_position",
                   s1_direction=s1_position.direction,
                   my_direction=my_signal_direction)
        return None
```

**How S3 gets S1's position:** The `position_poller` task in the strategy process (see event loop above) reads `POSITION:strategy:S1` for S3's process specifically. This is configured in S3's strategy config:

```yaml
  S3:
    params:
      cross_strategy_reads:
        - "S1"   # S3 reads S1's position for suppression logic
```

The orchestrator, when setting up S3's position_poller, subscribes it to both its own position AND S1's position.

**Access control:** S3 can READ S1's position. S3 CANNOT read S1's config, internal state, or signals. S3 CANNOT write to S1's position or any shared state. The Position Tracker (sole writer) is the only entity that updates position keys.

#### Risk Manager Cross-Strategy Checks

The Risk Manager (Component 6) performs portfolio-level checks that span strategies:
- S1 active → block S3 (same as above, enforced at Risk Gate)
- Portfolio delta exposure > ±50 NIFTY lot equivalents → WARNING
- Total portfolio drawdown > 10% → halt all strategies

These checks run in the Risk Manager process, not in strategy processes. Strategies are unaware of portfolio-level risk — they just see their signals being rejected at the Risk Gate.

---

### Data Routing Table

Each strategy subscribes to specific Redis Streams based on what instruments it trades. The orchestrator creates consumer groups for exactly these streams.

| Strategy | Strategy ID | Subscriptions | Primary Symbol | Notes |
|----------|------------|---------------|----------------|-------|
| ORB Breakout | S1 | `STREAM:TICK:SPOT` | NIFTY50-INDEX | Trades NIFTY CE options, needs spot for breakout detection |
| Overnight Futures | S2 | `STREAM:TICK:FUT:NIFTY` | NIFTY-FUT | Trades NIFTY futures, needs futures price for entry/exit |
| VWAP Mean Reversion | S3 | `STREAM:TICK:SPOT`, `STREAM:TICK:FUT:NIFTY` | NIFTY50-INDEX | Spot for VWAP, futures for execution reference |
| Equity Momentum | S4 | `STREAM:TICK:EQ:RELIANCE`, `STREAM:TICK:EQ:HDFCBANK`, `STREAM:TICK:EQ:INFY`, ... | Multiple | Basket of Nifty50 stocks. Exact list from config. |
| Expiry Day 0-DTE | S5 | `STREAM:TICK:SPOT` | NIFTY50-INDEX | Trades weekly NIFTY options, needs spot for strike selection |
| Volatility Premium | S6 | `STREAM:TICK:SPOT` | NIFTY50-INDEX | Sells NIFTY options, needs spot + VIX for regime |
| Statistical Pairs | S7 | `STREAM:TICK:FUT:{sym1}`, `STREAM:TICK:FUT:{sym2}` | {sym1}-FUT | Pairs on two correlated futures. Symbols from config. |

**VIX access:** S1, S5, and S6 need VIX for suppression logic. VIX is published to `STREAM:TICK:SPOT` as a regular tick (symbol="INDIAVIX"). Strategies that need VIX subscribe to `STREAM:TICK:SPOT` and filter by symbol in their `on_tick`.

**Stream fan-out:** A single stream can have multiple consumer groups. `STREAM:TICK:SPOT` has consumer groups for S1, S3, S5, and S6. Each group reads independently — S1 reading slowly does not affect S5's read speed.

---

### Failure Modes

| Failure | Detection | Impact | Recovery |
|---------|-----------|--------|----------|
| **Strategy process crashes (unhandled exception)** | `multiprocessing.Process.exitcode` checked by Strategy Manager every 5s | Single strategy down. Other strategies, OMS, and risk manager unaffected. Existing positions protected by server-side SL. | Auto-restart up to 3 times per session. `on_init()` → `restore_state()` → resume. After 3 restarts: mark KILLED, Telegram CRITICAL. |
| **Strategy process hangs (infinite loop, deadlock)** | Heartbeat stale >30s (Redis key HEALTH:strategy:{sid} TTL expires) | Strategy stops processing ticks. Bar signals lost. No new signals. | SIGTERM → wait 5s → SIGKILL → restart. |
| **Strategy on_tick takes >100ms consistently** | `time.perf_counter_ns()` measurement in tick_reader | Tick processing latency increases for this strategy. Other strategies unaffected (separate process). | 10 consecutive slow ticks → SUPPRESS with "performance_degraded". Operator investigates. |
| **Strategy on_bar_close takes >2s** | `asyncio.wait_for` timeout in tick_reader | Bar signal is lost. Tick ingestion continues (bar_close runs in thread executor). | 3 consecutive timeouts → SUPPRESS with "on_bar_close_timeout". Operator investigates. |
| **Strategy on_init fails** | Exception caught in `strategy_main` before event loop starts | Strategy never starts. | Mark DISABLED. Telegram CRITICAL. Operator fixes and restarts system. |
| **Strategy OOM (512MB limit)** | Process killed by OS with SIGKILL (exit code -9) | Strategy dies instantly. State may be up to 60s stale. | Auto-restart via Strategy Manager (counts toward 3-restart limit). |
| **Redis connection lost in strategy process** | `aioredis.ConnectionError` in tick_reader | Strategy cannot read ticks or publish signals. | `_restart_on_error` retries with exponential backoff (1s, 2s, 4s, ..., 30s). Consumer group preserves read offset — no lost ticks on reconnect. |
| **Parent process (orchestrator) dies** | `parent_watchdog` detects via `os.kill(parent_pid, 0)` | Strategy is orphaned. No fill notifications, no position updates, no restart on crash. | Save state to Redis, exit with code 1. Server-side SLs protect positions. systemd restarts the entire system. |
| **Redis Streams backlog (strategy too slow)** | `xinfo_groups` lag metric > threshold (e.g., 10,000 messages) | Strategy falls behind real-time. Signals based on stale data. | WARNING at 5,000 lag. SUPPRESS at 10,000 lag. Strategy continues consuming to catch up. |
| **Consumer group doesn't exist** | `NOGROUP` error from XREADGROUP | Strategy cannot read ticks. | Should not happen — groups created in Phase 8. If it does: log ERROR, exit, Strategy Manager recreates group on restart. |
| **Duplicate signals from strategy** | Signal deduplicator in Signal Router | Duplicate order would be placed. | Deduplicator suppresses. OMS signal_id idempotency provides second layer. |
| **BarBuilder receives out-of-order ticks** | `tick.exchange_ts < last_tick_ts` check in `add_tick` | Bar OHLCV would be corrupted. | Out-of-order ticks are logged and discarded. This is rare — Redis Streams preserve insertion order. |
| **BarBuilder gap in ticks (exchange halt, WS drop)** | `has_gap` flag computed from inter-tick intervals | Bar may not reflect true price action during gap. | `has_gap=True` set in Bar. Strategy decides whether to suppress signals. |
| **Strategy state restore fails** | Exception in `restore_state()` | Strategy starts from scratch, no memory of previous session. | WARNING logged. Strategy continues without restored state. |
| **All strategies die simultaneously** | All heartbeats stale, all processes dead | No signal generation. Existing positions protected by server-side SLs. | Strategy Manager restarts each. If persistent (e.g., Redis down), system enters degraded mode (OMS manages existing positions only). |
| **Hot-reload config change** | Config reloaded via CLI → Redis CONFIG:strategy:{sid} | Strategy picks up new params on next bar close. | No process restart needed. If config is invalid, old config persists. |
| **Strategy emits signal for wrong instrument type** | Instrument Resolver rejects signal that doesn't match config.instrument_type | Signal lost. | Log WARNING. Strategy bug — needs code fix. |

---

### State Table

| State Item | Storage | Scope | Lifecycle | Writer | Readers |
|-----------|---------|-------|-----------|--------|---------|
| Strategy lifecycle state (FSM) | In-memory (Strategy Manager) | Per-strategy | Session | Strategy Manager | Risk Manager, Monitoring, CLI |
| Strategy process PID | In-memory (Strategy Manager) | Per-strategy | Process lifetime | Strategy Manager | Monitoring |
| Strategy heartbeat | Redis `HEALTH:strategy:{sid}` (TTL 15s) | Per-strategy | Refreshed every 5s | Strategy process | Strategy Manager, Monitoring |
| Strategy internal state (indicators, counters) | In-memory (strategy process) | Per-strategy | Process lifetime | Strategy code | Strategy code only |
| Strategy state snapshot | Redis `STATE:strategy:{sid}` (TTL 1hr) | Per-strategy | Updated every 60s + on shutdown | Strategy process | Strategy Manager (on restart) |
| Suppression flag | Redis `SUPPRESS:strategy:{sid}` | Per-strategy | Set by Manager/Risk, cleared by Manager | Strategy Manager, Risk Manager | Strategy process (tick_reader) |
| BarBuilder state (current bar ticks) | In-memory (strategy process) | Per-strategy | Bar lifetime | BarBuilder | Strategy process |
| BarBuilder calibration (tick intervals) | In-memory (strategy process) | Per-strategy | Session (carries across bars) | BarBuilder | BarBuilder |
| Consumer group offsets | Redis (managed by Redis Streams) | Per-strategy per-stream | Persistent across restarts | Redis (on XACK) | Strategy process (on XREADGROUP) |
| Signal dedup state | In-memory (Signal Router process) | Per-strategy | Session | SignalDeduplicator | SignalDeduplicator |
| Placed signal IDs (OMS dedup) | In-memory (OMS) | Global | Session | OMS | OMS |
| Strategy config | Redis `CONFIG:strategy:{sid}` | Per-strategy | Session (hot-reloadable) | CLI (reload-config) | All components |
| Strategy config (source) | Disk `config/strategies.yaml` | All strategies | Persistent | Operator | CLI (reload-config) |
| Strategy class registry | Config `strategy_registry` in strategies.yaml | All strategies | Persistent | Operator | Strategy Manager (at startup) |
| Position (read by strategy) | Redis `POSITION:strategy:{sid}` | Per-strategy | Updated on every fill | Position Tracker | Strategy process (position_poller) |
| Cross-strategy position (S1→S3) | Redis `POSITION:strategy:S1` | S1 | Updated on every fill | Position Tracker | S3 process (position_poller) |
| Fill notifications | Redis pubsub `FILL:{sid}` | Per-strategy | Transient (pubsub) | OMS | Strategy process (fill_listener) |
| Restart count | In-memory (Strategy Manager) | Per-strategy | Session | Strategy Manager | Strategy Manager |
| Kill reason | In-memory (Strategy Manager) | Per-strategy | Session | Risk Manager → Strategy Manager | Monitoring, CLI, Telegram |

---

### Concurrency Model

The Strategy Orchestrator uses **multiprocessing for inter-strategy isolation** and **asyncio for intra-strategy concurrency**.

#### Inter-Strategy: OS Processes

```
Orchestrator (main process)
├── Strategy Manager (in main process event loop)
│   ├── Health monitor task (5s)
│   ├── Schedule checker task (60s)
│   └── State management
│
├── S1 Process (multiprocessing.Process)
│   └── asyncio event loop
│       ├── tick_reader (async)
│       ├── heartbeat (async)
│       ├── parent_watchdog (async)
│       ├── state_saver (async)
│       ├── position_poller (async)
│       ├── fill_listener (async)
│       └── ThreadPoolExecutor (1 thread: on_bar_close)
│
├── S2 Process
│   └── (same structure, no BarBuilder)
│
├── S3 Process
│   └── (same structure + reads S1 position)
│
├── S4 Process
│   └── (same structure, daily bars)
│
├── S5 Process
│   └── (same structure, 1-min bars)
│
├── S6 Process
│   └── (same structure, 5-min bars)
│
└── S7 Process
    └── (same structure, 5-min bars, two stream subscriptions)
```

**Why multiprocessing, not threading:** Python's GIL prevents true parallelism with threads. With 7 strategies, CPU-bound `on_bar_close` calls would serialize. Separate processes get true parallel execution on multi-core EC2 instances (recommended: 4+ vCPU).

**Why asyncio within each process:** Each strategy process is I/O-bound (Redis reads, heartbeat writes). asyncio handles thousands of concurrent I/O operations efficiently in a single thread. The only CPU-bound work (`on_bar_close`) is offloaded to a single-thread executor.

#### Intra-Strategy: asyncio + ThreadPoolExecutor

Within each strategy process:

| Task | Runs In | Blocking? | Notes |
|------|---------|-----------|-------|
| tick_reader | asyncio event loop | No (async Redis read) | Hot path. Must not block. |
| on_tick call | asyncio event loop (inline) | Yes, but <10ms | Must be fast. |
| on_bar_close call | ThreadPoolExecutor (1 thread) | Yes, up to 2s | Offloaded to thread. Event loop continues reading ticks. |
| heartbeat | asyncio event loop | No (async Redis write) | 5s interval. |
| parent_watchdog | asyncio event loop | No (os.kill is fast) | 5s interval. |
| state_saver | asyncio event loop | No (async Redis write) | 60s interval. |
| position_poller | asyncio event loop | No (async Redis read) | 5s interval. |
| fill_listener | asyncio event loop | No (async Redis pubsub) | Continuous. |

**Key invariant:** The event loop is NEVER blocked by strategy computation. `on_tick` must be <10ms (enforced by monitoring). `on_bar_close` runs in the thread executor — the event loop continues reading ticks, publishing heartbeats, and receiving fills while the strategy computes.

**ThreadPoolExecutor sizing:** 1 thread per strategy process. This means `on_bar_close` calls are serialized within a single strategy (you can't have two bars computing simultaneously for the same strategy). This is correct — bars are sequential by definition.

#### IPC: Redis as Message Bus

All inter-process communication goes through Redis:

| From | To | Channel | Message Type |
|------|----|---------|-------------|
| Data Ingester | Strategy processes | Redis Streams `STREAM:TICK:*` | Tick (orjson) |
| Strategy processes | Signal Router | Redis Stream `STREAM:SIGNAL` | StrategySignal (orjson) |
| OMS | Strategy processes | Redis pubsub `FILL:{sid}` | FillNotification (orjson) |
| Position Tracker | Strategy processes | Redis key `POSITION:strategy:{sid}` | PositionUpdate (orjson) |
| Strategy processes | Strategy Manager | Redis key `HEALTH:strategy:{sid}` | Epoch ms (string) |
| Strategy processes | Redis | Redis key `STATE:strategy:{sid}` | Strategy state (orjson) |
| Strategy Manager / Risk Manager | Strategy processes | Redis key `SUPPRESS:strategy:{sid}` | Reason string |

**No pipes, no shared memory, no sockets.** Redis provides persistence (consumer groups survive restart), pub/sub (fill notifications), and key-value (position reads). The overhead of Redis serialization (~150 bytes/tick, ~2KB/signal) is negligible compared to the isolation benefits.

---

### Edge Cases

#### 1. Strategy emits signal during SUPPRESSED state

**Scenario:** S1 is SUPPRESSED (VIX > 25). `on_bar_close` still runs (strategy process is alive, ticks flow). Strategy computes a breakout signal and returns it.

**Handling:** The `tick_reader` checks `SUPPRESS:strategy:S1` before publishing. Signal is logged but NOT published to `STREAM:SIGNAL`. The strategy is unaware its signal was suppressed — it doesn't need to know.

#### 2. Bar closes exactly at EOD (15:30)

**Scenario:** A 5-minute bar for S3 spans 15:25:00-15:29:59. The last tick arrives at 15:29:58. No tick arrives at 15:30:00 to trigger the close.

**Handling:** The `BarBuilder.force_close()` method is called during shutdown Phase 1. The in-progress bar is completed with whatever ticks have been accumulated. The strategy's `on_bar_close` is called one final time. If it emits a signal, the signal is suppressed (shutdown has blocked new entries via `HALT:no_new_entries`).

#### 3. Strategy restart during active position

**Scenario:** S1 crashes while holding a LONG position. Strategy Manager restarts it.

**Handling:**
1. Server-side SL order remains active on broker (protection)
2. Strategy process restarts: `on_init()` → `restore_state()` (from Redis snapshot)
3. `position_poller` immediately reads `POSITION:strategy:S1` → calls `on_position_update` with current position
4. Strategy now knows it's positioned and adjusts behavior accordingly
5. BarBuilder starts fresh (no tick history). First partial bar may have few ticks — `has_gap` will be True

#### 4. Two strategies fire signals simultaneously

**Scenario:** S1 and S5 both emit signals at exactly the same millisecond.

**Handling:** Both signals are published to `STREAM:SIGNAL` (Redis Streams are append-only, concurrent writes are serialized by Redis). The Signal Router processes them sequentially. Capital Allocator checks available capital for each — if insufficient for both, the higher-priority signal (S5 > S1, see priority table) gets filled first.

#### 5. Config hot-reload while bar is computing

**Scenario:** Operator runs `live.cli reload-config` while S3's `on_bar_close` is running in the thread executor.

**Handling:** The config change updates Redis `CONFIG:strategy:S3`. The strategy picks up new config on the NEXT `on_bar_close` call (or the next time it reads config). The in-progress computation uses the old config. This is correct — mid-computation config changes would cause inconsistent state.

#### 6. Redis Streams message loss (XTRIM races)

**Scenario:** Data Ingester calls `XTRIM MAXLEN ~60000`. A slow strategy hasn't read message 59,999 yet.

**Handling:** The consumer group tracks the last ACK'd offset. If the group's offset points to a trimmed message, Redis returns the next available message. The strategy loses ticks between its last ACK and the trim point. This manifests as a gap in `BarBuilder` (detected via `has_gap`). The trim threshold (60,000) is sized to provide ~5 minutes of buffer at peak tick rates — more than enough for any strategy that isn't completely stalled.

**Mitigation:** If a strategy's consumer group lag exceeds 50% of the trim threshold (30,000 messages), Monitoring fires a WARNING. At 80% (48,000), the strategy is SUPPRESSED until it catches up.

#### 7. Instrument master CSV not available on Tuesday (S5 impact)

**Scenario:** It's Tuesday (weekly expiry). The 08:30 instrument CSV download fails all 3 retries (08:30, 08:35, 08:40). Yesterday's CSV is loaded as fallback.

**Handling:** The Strategy Manager checks `is_tuesday AND csv_is_fallback`. If true: S5 is set to DISABLED for the day (not SUPPRESSED — it never starts). Telegram WARNING: "S5 disabled: instrument CSV fallback on expiry day". All other strategies continue normally (they don't trade 0-DTE contracts affected by the stale CSV).

#### 8. Strategy returns signal with wrong signal_id format

**Scenario:** Strategy S8 (new plugin) returns a StrategySignal where `signal_id` is not a valid UUID.

**Handling:** `StrategySignal` is a Pydantic model. Validation runs on construction. If `signal_id` doesn't match UUID format, Pydantic raises `ValidationError`. The `tick_reader` catches this in the signal publication path, logs ERROR, and discards the signal. The strategy is not crashed — just the bad signal is dropped.

#### 9. Strategy process fork and Redis connection

**Scenario:** `multiprocessing.Process` forks. The child inherits the parent's Redis connection object.

**Handling:** The child process must NOT reuse the parent's Redis connection (forked file descriptors cause data corruption). The strategy process creates its own Redis connection in `strategy_main()` before any Redis operations. The `multiprocessing.Process` is started with `start_method="spawn"` (not "fork") on Linux to avoid inheriting any connection state.

```python
multiprocessing.set_start_method("spawn")  # set once at import time
```

---


## Component 4: Capital Allocation Engine

### Responsibility

- Receive strategy signals from the Signal Router and allocate capital to each (account, signal) pair
- Signal deduplication: suppress duplicate signals within per-strategy cooldown windows before account fan-out
- Position-aware allocation: check existing positions per (account, strategy) before sizing
- ERC weight computation: daily rebalance of strategy weights using Equal Risk Contribution
- Per-account Kelly fraction sizing with ramp schedule and Sharpe override
- Margin monitoring: poll Dhan fund/margin API per account, throttle allocations when utilization is high
- Priority ordering: when multiple strategies fire simultaneously, process in defined priority order
- **Does NOT** place orders, resolve instruments, or manage positions (those are downstream components)
- **Does NOT** make decisions based on aggregate cross-account state (each account is sized independently)

---

### Signal Deduplication

Signal deduplication runs BEFORE the account fan-out loop. A signal is checked for duplicates exactly once. If it passes, it fans out to N accounts. If it fails, no account sees it. This prevents N redundant dedup checks and ensures consistent behavior: either all accounts process a signal or none do.

#### Why asyncio.Lock, Not threading.Lock

The Capital Allocation Engine runs inside an `asyncio` event loop on the main process. All signal processing is coroutine-based. Using `threading.Lock` inside an async context causes a deadlock:

```
1. Coroutine A acquires threading.Lock (blocks the OS thread)
2. Coroutine A awaits an async operation (e.g., Redis read)
3. The event loop cannot schedule Coroutine B because the OS thread is blocked
4. Coroutine A's await never completes because the event loop is frozen
5. Deadlock: the OS thread is blocked waiting for an await that requires the OS thread
```

`asyncio.Lock` suspends the coroutine (not the OS thread), allowing the event loop to continue scheduling other coroutines while the lock holder is awaiting.

This is the Finding 12 fix. Every lock in the Capital Allocation Engine is `asyncio.Lock`.

#### Dedup Key and Cooldown Windows

The dedup key is a 3-tuple: `(strategy_id, direction, underlying)`. This means:

- S1 LONG NIFTY and S1 SHORT NIFTY are different keys (direction differs)
- S1 LONG NIFTY and S3 LONG NIFTY are different keys (strategy differs)
- S1 LONG NIFTY fired twice within cooldown: second is suppressed

Per-strategy cooldowns are calibrated to each strategy's signal frequency:

| Strategy | Cooldown (seconds) | Cooldown (human) | Rationale |
|----------|-------------------|-------------------|-----------|
| S1 (ORB) | 300 | 5 minutes | ORB fires once per day at open. 5 min covers re-trigger on noisy first candle. |
| S2 (Overnight) | 86400 | 24 hours | One signal per day at 15:15. Full-day cooldown. |
| S3 (VWAP MR) | 60 | 1 minute | Intraday mean-reversion fires frequently. 60s prevents stacking on oscillation. |
| S4 (Momentum) | 2592000 | 30 days | Monthly rebalance. Cooldown covers full rebalance window. |
| S5 (Expiry Day) | 21600 | 6 hours | 0-DTE signals. 6h prevents re-entry on same expiry day. |
| S6 (Vol Premium) | 604800 | 7 days | Weekly vol selling cadence. |
| S7 (Pairs) | 86400 | 24 hours | Daily pairs rebalance frequency. |

#### Full Implementation

```python
import asyncio
import time
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class SignalDeduplicator:
    """
    Suppress duplicate signals within a per-strategy cooldown window.

    Runs BEFORE account fan-out. A signal that passes dedup is guaranteed
    to be the only live signal for its (strategy_id, direction, underlying)
    key within the cooldown window.

    Thread-safety: uses asyncio.Lock (NOT threading.Lock). See Finding 12.

    Lifecycle: created once at session startup. State is in-memory only.
    On restart, all cooldowns reset — this is acceptable because a restart
    also resets strategy state, so re-firing is expected and correct.
    """

    COOLDOWN_MS: dict[str, int] = field(default_factory=lambda: {
        "S1": 300000,
        "S2": 86400000,
        "S3": 60000,
        "S4": 2592000000,
        "S5": 21600000,
        "S6": 604800000,
        "S7": 86400000,
    })

    _last_signal: dict[tuple[str, str, str], int] = field(default_factory=dict)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _suppressed_count: int = field(default=0)
    _passed_count: int = field(default=0)

    async def check(self, signal: "StrategySignal") -> bool:
        """
        Check whether a signal should be processed or suppressed.

        Args:
            signal: The incoming strategy signal. Must have attributes:
                    strategy_id (str), direction (str), underlying (str),
                    signal_ts (int — epoch milliseconds).

        Returns:
            True if the signal should be processed (not a duplicate).
            False if the signal is suppressed (within cooldown of a prior
            signal with the same key).
        """
        key = (signal.strategy_id, signal.direction, signal.underlying)
        cooldown = self.COOLDOWN_MS.get(signal.strategy_id)

        if cooldown is None:
            logger.error("dedup_unknown_strategy",
                        strategy_id=signal.strategy_id)
            return False  # unknown strategy → reject for safety

        async with self._lock:
            last_ts = self._last_signal.get(key, 0)
            elapsed = signal.signal_ts - last_ts

            if elapsed < cooldown:
                self._suppressed_count += 1
                logger.info("signal_deduplicated",
                           strategy_id=signal.strategy_id,
                           direction=signal.direction,
                           underlying=signal.underlying,
                           elapsed_ms=elapsed,
                           cooldown_ms=cooldown,
                           total_suppressed=self._suppressed_count)
                return False

            self._last_signal[key] = signal.signal_ts
            self._passed_count += 1
            return True

    def stats(self) -> dict[str, int]:
        """Return dedup statistics for monitoring."""
        return {
            "passed": self._passed_count,
            "suppressed": self._suppressed_count,
            "active_keys": len(self._last_signal),
        }
```

---

### Pydantic Models

#### AllocationRequest

```python
import pydantic
from decimal import Decimal
from typing import Literal


class AllocationRequest(pydantic.BaseModel):
    """
    Request for capital allocation for one (account, signal) pair.

    Created by the multi-account fan-out loop in the Signal Router.
    The existing_position_qty field is populated by querying the
    PositionTracker for this specific (account_id, strategy_id) pair.

    Fields:
        account_id: The Dhan account ID receiving this allocation.
        strategy_id: Which strategy generated the signal (S1-S7).
        direction: Signal direction — LONG or SHORT.
        underlying: The underlying instrument (e.g., "NIFTY", "BANKNIFTY").
        instrument_type: What the strategy wants to trade.
        estimated_premium: Per-unit price estimate for sizing (₹ per share/unit).
                          For options: option premium. For futures: margin required.
        lot_size: Exchange-defined lot size for the instrument.
        existing_position_qty: Current position quantity held by THIS account
                              for THIS strategy. Signed: positive = long, negative = short,
                              zero = flat. Read from PositionTracker API.
        signal_id: UUID of the originating signal (for audit trail).
        signal_ts: Epoch milliseconds when the strategy generated the signal.
    """
    account_id: str
    strategy_id: str
    direction: Literal["LONG", "SHORT"]
    underlying: str
    instrument_type: Literal["OPT_BUY", "OPT_SELL", "FUT", "EQ"]
    estimated_premium: float = pydantic.Field(gt=0)
    lot_size: int = pydantic.Field(gt=0)
    existing_position_qty: int
    signal_id: str
    signal_ts: int

    model_config = pydantic.ConfigDict(frozen=True)
```

#### AllocationResponse

```python
class AllocationResponse(pydantic.BaseModel):
    """
    Result of a capital allocation request for one (account, signal) pair.

    Fields:
        approved: Whether the allocation was granted.
        allocated_capital: The ₹ amount allocated to this trade for this account.
                          Zero if rejected.
        max_lots: Number of lots this account should trade. Floor-rounded.
                  Zero if rejected.
        kelly_fraction: The effective Kelly fraction used for this allocation
                       (after ramp, Sharpe override, and margin clamping).
        erc_weight: The ERC weight for this strategy at the time of allocation.
        is_exit: True if this signal is an exit (opposite direction to existing
                position). Exits do not consume capital — they release it.
        rejection_reason: Human-readable reason if approved=False. None if approved.
        margin_util_at_allocation: The account's margin utilization at the time
                                   of this allocation decision (0.0 to 1.0).
    """
    approved: bool
    allocated_capital: Decimal = Decimal("0")
    max_lots: int = 0
    kelly_fraction: float = 0.0
    erc_weight: float = 0.0
    is_exit: bool = False
    rejection_reason: str | None = None
    margin_util_at_allocation: float = 0.0

    model_config = pydantic.ConfigDict(frozen=True)
```

#### AccountConfig

```python
class AccountConfig(pydantic.BaseModel):
    """
    Per-account configuration projection used by the Capital Allocator.

    This is a READ-ONLY view of the canonical Account model defined in
    Component 11 (Account Replication Layer). The Capital Allocator receives
    AccountConfig instances from the AccountManager — it never constructs
    them directly.

    Fields:
        account_id: Unique identifier (e.g., "prop", "client_001").
        dhan_client_id: Dhan's internal client ID string.
        dhan_access_token: OAuth2 access token (refreshed daily). Never logged.
        capital: Total capital in ₹. Decimal for exact arithmetic at ₹10Cr scale.
        kelly_fraction: Maximum Kelly fraction for this account (0.05 to 1.0).
        enabled_strategies: Which strategies this account participates in.
        strategy_weights: Per-strategy capital allocation weights.
    """
    account_id: str
    dhan_client_id: str
    dhan_access_token: str = pydantic.Field(repr=False)  # never log this
    capital: Decimal = pydantic.Field(gt=0)
    kelly_fraction: float = pydantic.Field(ge=0.05, le=1.0)
    enabled_strategies: list[str] = pydantic.Field(
        default_factory=lambda: ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    )
    strategy_weights: dict[str, float] = pydantic.Field(default_factory=dict)

    model_config = pydantic.ConfigDict(frozen=True)
```

#### MarginState

```python
class MarginState(pydantic.BaseModel):
    """
    Per-account margin snapshot from Dhan's GET /v2/fund-limit endpoint.

    Polled every 30 seconds per account AND on every fill event.

    Fields:
        account_id: The account this margin state belongs to.
        available_margin: Unrealized available margin in ₹.
        used_margin: Currently consumed margin in ₹.
        total_margin: available_margin + used_margin.
        utilization: used_margin / total_margin (0.0 to 1.0).
                    When utilization > 0.80, new allocations for this account
                    are throttled.
        last_polled_ts: Epoch seconds of the last successful poll.
        last_polled_source: Whether the last update came from periodic polling
                           or a fill-triggered refresh.
        consecutive_poll_failures: Count of consecutive poll failures. If > 3,
                                  the account enters degraded margin mode.
    """
    account_id: str
    available_margin: float = 0.0
    used_margin: float = 0.0
    total_margin: float = 0.0
    utilization: float = 0.0
    last_polled_ts: float = 0.0
    last_polled_source: Literal["periodic", "fill", "startup"] = "startup"
    consecutive_poll_failures: int = 0

    @property
    def is_throttled(self) -> bool:
        """True if margin utilization exceeds 80% threshold."""
        return self.utilization > 0.80

    @property
    def is_degraded(self) -> bool:
        """True if margin data is stale (3+ consecutive poll failures)."""
        return self.consecutive_poll_failures > 3

    @property
    def staleness_s(self) -> float:
        """Seconds since last successful margin poll."""
        return time.time() - self.last_polled_ts
```

---

### Existing Position Check (HIGH-5 Fix)

The HIGH-5 bug: S3 (VWAP mean reversion) fires multiple LONG signals during a sustained dip. Without position-aware allocation, each signal allocates fresh capital, stacking 5 concurrent long positions in the same underlying. This is unintended — S3 should hold at most one position at a time per account.

The fix: every `AllocationRequest` carries `existing_position_qty`, read from the PositionTracker for the specific `(account_id, strategy_id)` pair. The allocator checks this before computing any sizing.

#### Position Check Logic

There are three cases:

| existing_position_qty | Signal direction | Result | Rationale |
|----------------------|-----------------|--------|-----------|
| 0 (flat) | LONG or SHORT | Normal allocation | No existing position. Proceed with ERC + Kelly sizing. |
| > 0 (long) | LONG | REJECT | Same-direction stacking. Strategy already holds a long. HIGH-5 fix. |
| > 0 (long) | SHORT | APPROVE as exit | Opposite direction = exit signal. No capital allocation needed. Flag `is_exit=True`. |
| < 0 (short) | SHORT | REJECT | Same-direction stacking. Strategy already holds a short. |
| < 0 (short) | LONG | APPROVE as exit | Opposite direction = exit signal. Flag `is_exit=True`. |

#### Per-Account Position State

Position state is per-account. Account A may have an S3 LONG position while Account B is flat for S3. This happens when:

1. Account B rejected the original S3 signal (insufficient margin)
2. Account B was added after Account A took the S3 position
3. Account A got a full fill, Account B got zero fill (liquidity divergence)

The position check queries the PositionTracker with `(account_id, strategy_id)`, not just `strategy_id`. This is why `AllocationRequest` includes `account_id`.

#### Implementation

```python
def _check_existing_position(
    self,
    req: AllocationRequest,
) -> AllocationResponse | None:
    """
    Check existing position for this (account, strategy) pair.

    Returns:
        AllocationResponse if a decision can be made immediately
        (reject same-direction, approve exit). Returns None if the
        position is flat and normal allocation should proceed.
    """
    qty = req.existing_position_qty

    if qty == 0:
        # Flat — proceed to normal allocation
        return None

    is_long = qty > 0
    signal_is_long = req.direction == "LONG"
    same_direction = (is_long and signal_is_long) or (not is_long and not signal_is_long)

    if same_direction:
        # HIGH-5 fix: reject same-direction stacking
        logger.warning("allocation_rejected_position_exists",
                      account_id=req.account_id,
                      strategy_id=req.strategy_id,
                      direction=req.direction,
                      existing_qty=qty)
        return AllocationResponse(
            approved=False,
            rejection_reason=(
                f"strategy_already_positioned_same_direction: "
                f"{req.strategy_id} holds {qty} units on account {req.account_id}"
            ),
        )

    # Opposite direction = exit signal
    logger.info("allocation_exit_signal",
               account_id=req.account_id,
               strategy_id=req.strategy_id,
               direction=req.direction,
               existing_qty=qty)
    return AllocationResponse(
        approved=True,
        allocated_capital=0.0,  # exits release capital, they don't consume it
        max_lots=abs(qty) // req.lot_size,  # exit the full position
        kelly_fraction=0.0,
        erc_weight=0.0,
        is_exit=True,
        rejection_reason=None,
    )
```

---

### ERC Allocation

Equal Risk Contribution (ERC) weights ensure each strategy contributes equally to portfolio risk. A strategy with higher volatility gets a smaller capital allocation.

#### Weight Computation Schedule

| Day range | Method | Rationale |
|-----------|--------|-----------|
| Day 1-20 | Static weights only | Insufficient return history for reliable vol estimates. |
| Day 20-40 | 50% static + 50% ERC | Blending period. Vol estimates stabilize but remain noisy. |
| Day 40+ | Pure ERC | Sufficient 20-day rolling vol history. |

#### Static Weights

These are the baseline weights used when ERC history is insufficient:

| Strategy | Static Weight | Rationale |
|----------|--------------|-----------|
| S1 (ORB) | 0.20 (20%) | High-frequency intraday, consistent edge |
| S2 (Overnight) | 0.15 (15%) | Overnight carry, moderate sizing |
| S3 (VWAP MR) | 0.15 (15%) | Intraday mean reversion, moderate sizing |
| S4 (Momentum) | 0.20 (20%) | Monthly rebalance, large positions |
| S5 (Expiry Day) | 0.10 (10%) | 0-DTE, high risk per trade, small allocation |
| S6 (Vol Premium) | 0.15 (15%) | Weekly vol selling, moderate sizing |
| S7 (Pairs) | 0.05 (5%) | Smallest allocation, lowest conviction |

Static weights sum to 1.00.

#### ERC Formula

```
For each strategy i with active return history:
    σ_i = 20-day rolling standard deviation of daily strategy returns
    inverse_vol_i = 1 / σ_i
    w_i = inverse_vol_i / Σ(inverse_vol_j) for all j

This gives: w_i × σ_i ≈ constant for all i
(Each strategy contributes approximately equal risk to the portfolio)
```

#### Recomputation Timing

ERC weights are recomputed daily at 08:45 IST (15 minutes before market open at 09:00). This timing ensures:

1. Previous day's returns are finalized (EOD P&L computed by Position Tracker overnight)
2. Weights are ready before the first signal can fire
3. No mid-session weight changes (allocation is deterministic within a trading day)

#### Full Implementation

```python
import numpy as np
import asyncio
from dataclasses import dataclass, field

import structlog
import redis.asyncio as aioredis

logger = structlog.get_logger(__name__)


STATIC_WEIGHTS: dict[str, float] = {
    "S1": 0.20,
    "S2": 0.15,
    "S3": 0.15,
    "S4": 0.20,
    "S5": 0.10,
    "S6": 0.15,
    "S7": 0.05,
}

STRATEGY_IDS: list[str] = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]


@dataclass
class ERCComputer:
    """
    Compute Equal Risk Contribution weights for 7 strategies.

    Weights are shared across all accounts — strategy performance is
    strategy-level, not account-level. Account A and Account B both
    use the same ERC weights because the strategies generate the same
    signals for both accounts (signals are shared; only sizing differs).

    State:
        - Reads daily returns from Redis (written by Position Tracker)
        - Writes computed weights to Redis ALLOC:erc_weights (shared)
        - In-memory cache of current weights for fast access

    Called once per day at 08:45 IST by the session orchestrator.
    """

    redis: aioredis.Redis
    _current_weights: dict[str, float] = field(default_factory=lambda: dict(STATIC_WEIGHTS))
    _session_start_date: str = ""  # YYYY-MM-DD, set at init
    _trading_day_count: int = 0

    async def initialize(self, session_start_date: str) -> None:
        """
        Initialize the ERC computer at session startup.

        Args:
            session_start_date: The date the system first went live (YYYY-MM-DD).
                               Used to determine which weight regime applies
                               (static, blend, or pure ERC).
        """
        self._session_start_date = session_start_date

        # Try to load existing weights from Redis (survives restart within same day)
        cached = await self.redis.hgetall("ALLOC:erc_weights")
        if cached:
            self._current_weights = {
                k.decode(): float(v) for k, v in cached.items()
            }
            logger.info("erc_weights_loaded_from_cache",
                       weights=self._current_weights)
        else:
            self._current_weights = dict(STATIC_WEIGHTS)
            await self._write_to_redis()
            logger.info("erc_weights_initialized_static",
                       weights=self._current_weights)

    async def recompute(self, trading_day_count: int) -> dict[str, float]:
        """
        Recompute ERC weights. Called daily at 08:45 IST.

        Args:
            trading_day_count: Number of trading days since system went live.
                              Day 1 = first trading day. Not calendar days.

        Returns:
            Updated weight dict {strategy_id: weight}.
        """
        self._trading_day_count = trading_day_count

        if trading_day_count <= 20:
            # Regime 1: static weights only
            self._current_weights = dict(STATIC_WEIGHTS)
            await self._write_to_redis()
            logger.info("erc_regime_static",
                       day=trading_day_count,
                       weights=self._current_weights)
            return self._current_weights

        # Compute ERC weights from 20-day rolling vol
        erc_weights = await self._compute_erc_from_returns()

        if erc_weights is None:
            # Insufficient data or computation error — fall back to static
            logger.warning("erc_computation_failed_fallback_static",
                         day=trading_day_count)
            self._current_weights = dict(STATIC_WEIGHTS)
            await self._write_to_redis()
            return self._current_weights

        if trading_day_count <= 40:
            # Regime 2: 50% static + 50% ERC blend
            blended = {}
            for sid in STRATEGY_IDS:
                blended[sid] = 0.5 * STATIC_WEIGHTS[sid] + 0.5 * erc_weights[sid]

            # Renormalize to sum to 1.0 (blending preserves sum, but float precision)
            total = sum(blended.values())
            self._current_weights = {
                sid: w / total for sid, w in blended.items()
            }

            logger.info("erc_regime_blend",
                       day=trading_day_count,
                       static=STATIC_WEIGHTS,
                       erc_raw=erc_weights,
                       blended=self._current_weights)
        else:
            # Regime 3: pure ERC
            self._current_weights = erc_weights
            logger.info("erc_regime_pure",
                       day=trading_day_count,
                       weights=self._current_weights)

        await self._write_to_redis()
        return self._current_weights

    async def _compute_erc_from_returns(self) -> dict[str, float] | None:
        """
        Compute inverse-volatility weights from 20-day rolling returns.

        Reads daily returns from Redis keys RETURNS:{strategy_id}:daily
        (written by Position Tracker at EOD).

        Returns:
            Weight dict, or None if any strategy has fewer than 20 data points.
        """
        vols: dict[str, float] = {}

        for sid in STRATEGY_IDS:
            key = f"RETURNS:{sid}:daily"
            raw = await self.redis.lrange(key, -20, -1)  # last 20 entries

            if len(raw) < 20:
                logger.debug("erc_insufficient_data",
                           strategy_id=sid,
                           data_points=len(raw))
                return None

            returns = np.array([float(r) for r in raw], dtype=np.float64)
            vol = float(np.std(returns, ddof=1))

            if vol < 1e-10:
                # Strategy returned exactly 0 for 20 days (no trades or flat P&L)
                # Assign a small floor vol to avoid division by zero
                vol = 1e-6
                logger.warning("erc_zero_vol_floored",
                             strategy_id=sid)

            vols[sid] = vol

        # Inverse-vol weighting
        inverse_vols = {sid: 1.0 / v for sid, v in vols.items()}
        total_inverse = sum(inverse_vols.values())

        weights = {
            sid: iv / total_inverse
            for sid, iv in inverse_vols.items()
        }

        # Sanity check: no single strategy exceeds 40% weight
        for sid, w in weights.items():
            if w > 0.40:
                logger.warning("erc_weight_capped",
                             strategy_id=sid,
                             raw_weight=w,
                             capped_to=0.40)
                weights[sid] = 0.40

        # Renormalize after capping
        total = sum(weights.values())
        weights = {sid: w / total for sid, w in weights.items()}

        return weights

    async def _write_to_redis(self) -> None:
        """Write current weights to Redis for cross-component access."""
        mapping = {sid: str(w) for sid, w in self._current_weights.items()}
        await self.redis.hset("ALLOC:erc_weights", mapping=mapping)

    def get_weight(self, strategy_id: str) -> float:
        """
        Get the current ERC weight for a strategy.

        Fast path: in-memory lookup, no async, no Redis call.
        Used in the hot path during allocation (under asyncio.Lock).
        """
        return self._current_weights.get(strategy_id, 0.0)
```

---

### Per-Account Kelly Fraction

Each account has its own `kelly_fraction` in config, reflecting the account owner's risk tolerance. Conservative accounts (e.g., a client's discretionary capital) use 0.25. Aggressive accounts (e.g., the prop desk) use 0.50.

The system never uses full Kelly (1.0). Even the most aggressive account caps at 0.50 (half Kelly), which is the empirically recommended maximum for strategies with uncertain edge estimates.

#### Kelly Ramp Schedule

The ramp schedule scales relative to each account's `kelly_fraction`:

| Day range | Effective Kelly | Formula | Rationale |
|-----------|----------------|---------|-----------|
| Day 1-20 | Quarter of account Kelly | `0.25 × account.kelly_fraction` | System is new. Prove the edge before sizing up. |
| Day 21-60 | Linear ramp from quarter to full | `(0.25 + (day - 20) / 40 × 0.75) × account.kelly_fraction` | Gradually increase sizing as confidence grows. |
| Day 61+ | Full account Kelly | `account.kelly_fraction` | Ramp complete. Steady-state sizing. |

**Example for an account with `kelly_fraction = 0.40`:**

| Day | Multiplier | Effective Kelly |
|-----|-----------|----------------|
| 1 | 0.25 | 0.10 |
| 10 | 0.25 | 0.10 |
| 20 | 0.25 | 0.10 |
| 30 | 0.4375 | 0.175 |
| 40 | 0.625 | 0.25 |
| 50 | 0.8125 | 0.325 |
| 60 | 1.0 | 0.40 |
| 100 | 1.0 | 0.40 |

#### Sharpe Override

If a strategy's 20-day rolling Sharpe ratio drops below 0.5, the effective Kelly for ALL accounts is clamped to `0.25 × account.kelly_fraction` for that strategy. This is a per-strategy clamp, not a portfolio-wide clamp.

The Sharpe override persists until the strategy's 20-day Sharpe recovers above 0.5. There is no hysteresis band — the threshold is a hard cutoff at 0.5.

**Why 0.5?** A Sharpe below 0.5 annualized over 20 days indicates the strategy's edge is either weakening or in a drawdown phase. Reducing sizing limits damage during regime deterioration.

#### Full Implementation

```python
@dataclass
class KellyComputer:
    """
    Compute per-account, per-strategy Kelly fraction with ramp and override.

    The Kelly fraction determines what fraction of the allocated capital
    (after ERC weighting) is actually deployed. It is the second layer
    of position sizing:

        position_size = account.capital × erc_weight × kelly_fraction

    State:
        - Reads 20-day Sharpe from Redis (computed by Position Tracker)
        - Per-account kelly_fraction from AccountConfig (immutable)
        - Trading day count from session orchestrator
    """

    redis: aioredis.Redis

    async def compute(
        self,
        account: AccountConfig,
        strategy_id: str,
        trading_day_count: int,
    ) -> float:
        """
        Compute the effective Kelly fraction for one (account, strategy) pair.

        Args:
            account: Account configuration with base kelly_fraction.
            strategy_id: Strategy ID (S1-S7).
            trading_day_count: Trading days since system went live.

        Returns:
            Effective Kelly fraction (0.0 to account.kelly_fraction).
        """
        base_kelly = account.kelly_fraction

        # Step 1: Compute ramp multiplier
        if trading_day_count <= 20:
            ramp_multiplier = 0.25
        elif trading_day_count <= 60:
            # Linear ramp from 0.25 to 1.0 over days 21-60
            progress = (trading_day_count - 20) / 40.0
            ramp_multiplier = 0.25 + progress * 0.75
        else:
            ramp_multiplier = 1.0

        effective = ramp_multiplier * base_kelly

        # Step 2: Sharpe override check
        sharpe = await self._get_strategy_sharpe(strategy_id)

        if sharpe is not None and sharpe < 0.5:
            clamped = 0.25 * base_kelly
            if effective > clamped:
                logger.info("kelly_sharpe_override",
                           account_id=account.account_id,
                           strategy_id=strategy_id,
                           sharpe=round(sharpe, 3),
                           effective_before=round(effective, 4),
                           clamped_to=round(clamped, 4))
                effective = clamped

        return round(effective, 6)

    async def _get_strategy_sharpe(self, strategy_id: str) -> float | None:
        """
        Read the 20-day rolling Sharpe for a strategy from Redis.

        Returns None if insufficient data (fewer than 20 trading days).
        """
        raw = await self.redis.get(f"SHARPE:{strategy_id}:20d")
        if raw is None:
            return None
        return float(raw)
```

---

### Margin Management

Margin state is per-account. Each account's available margin is polled independently from the Dhan `GET /v2/fund-limit` endpoint. There is no cross-account margin netting or pooling.

#### Polling Schedule

| Trigger | Frequency | Rationale |
|---------|-----------|-----------|
| Periodic | Every 30 seconds per account | Catch margin changes from external trades or manual activity on the Dhan account. |
| On fill | Immediately after any fill event for the account | Margin changes significantly on fill. Get fresh state for the next allocation decision. |
| On startup | Once per account during Phase 4 (Position Recovery) | Establish initial margin state before any trading. |

#### Margin Throttling

When `MarginState.utilization > 0.80` for an account, new allocations for THAT account are reduced:

| Utilization | Effect | Rationale |
|-------------|--------|-----------|
| 0% - 80% | Normal allocation | Sufficient margin headroom. |
| 80% - 90% | Allocation reduced by 50% | Approaching danger zone. Halve new positions. |
| 90% - 95% | Allocation reduced by 75% | Near margin call territory. Minimal new positions. |
| 95%+ | No new allocations | Protect against margin call. Only exits allowed. |

This throttling applies to the account whose margin is high. Other accounts are unaffected. There is no aggregate margin view that drives allocation decisions.

An aggregate margin dashboard exists for monitoring only — the operator can see total margin across all accounts in Grafana, but no automated decision uses the aggregate number.

#### Polling Implementation

```python
import aiohttp

@dataclass
class MarginPoller:
    """
    Poll Dhan's GET /v2/fund-limit for each account's margin state.

    One poller per account. Runs as a long-lived asyncio task.

    Architecture:
        - Periodic poll: every 30s
        - Event-driven poll: triggered via asyncio.Event on fill
        - Writes MarginState to in-memory dict (fast access for allocator)
        - Publishes to Redis MARGIN:{account_id} for monitoring

    Failure handling:
        - HTTP timeout (10s) or error: increment consecutive_poll_failures
        - After 3 consecutive failures: account enters degraded margin mode
        - In degraded mode: allocator uses the LAST KNOWN margin state with
          a 50% safety haircut (assumes worst case)
    """

    account: AccountConfig
    redis: aioredis.Redis
    _state: MarginState = field(init=False)
    _fill_event: asyncio.Event = field(default_factory=asyncio.Event)
    _session: aiohttp.ClientSession | None = None

    def __post_init__(self) -> None:
        self._state = MarginState(account_id=self.account.account_id)

    @property
    def state(self) -> MarginState:
        """Current margin state (in-memory, fast read)."""
        return self._state

    def notify_fill(self) -> None:
        """Called by OMS on any fill for this account. Triggers immediate poll."""
        self._fill_event.set()

    async def run(self) -> None:
        """
        Main polling loop. Runs until cancelled.

        Two await paths:
        1. asyncio.sleep(30) — periodic poll
        2. self._fill_event.wait() — fill-triggered poll

        whichever fires first triggers a poll.
        """
        self._session = aiohttp.ClientSession(
            headers={
                "access-token": self.account.dhan_access_token,
                "Content-Type": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=10),
        )

        try:
            while True:
                # Wait for either 30s timeout or fill event
                fill_triggered = False
                try:
                    await asyncio.wait_for(self._fill_event.wait(), timeout=30.0)
                    fill_triggered = True
                    self._fill_event.clear()
                except asyncio.TimeoutError:
                    pass  # periodic poll

                source = "fill" if fill_triggered else "periodic"
                await self._poll(source)
        finally:
            if self._session:
                await self._session.close()

    async def _poll(self, source: str) -> None:
        """Execute one margin poll."""
        try:
            async with self._session.get(
                "https://api.dhan.co/v2/fund-limit"
            ) as resp:
                if resp.status != 200:
                    self._state.consecutive_poll_failures += 1
                    logger.warning("margin_poll_http_error",
                                 account_id=self.account.account_id,
                                 status=resp.status,
                                 failures=self._state.consecutive_poll_failures)
                    return

                data = await resp.json()

            available = float(data.get("availabelBalance", 0))
            used = float(data.get("utilizedMargin", 0))
            total = available + used

            self._state = MarginState(
                account_id=self.account.account_id,
                available_margin=available,
                used_margin=used,
                total_margin=total,
                utilization=used / total if total > 0 else 0.0,
                last_polled_ts=time.time(),
                last_polled_source=source,
                consecutive_poll_failures=0,
            )

            # Publish to Redis for monitoring dashboards
            await self.redis.hset(
                f"MARGIN:{self.account.account_id}",
                mapping={
                    "available": str(available),
                    "used": str(used),
                    "total": str(total),
                    "utilization": str(self._state.utilization),
                    "ts": str(self._state.last_polled_ts),
                    "source": source,
                },
            )

        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            self._state.consecutive_poll_failures += 1
            logger.warning("margin_poll_network_error",
                         account_id=self.account.account_id,
                         error=str(e),
                         failures=self._state.consecutive_poll_failures)

    def margin_multiplier(self) -> float:
        """
        Return the allocation multiplier based on current margin utilization.

        Called by the allocator to scale down allocations when margin is tight.

        Returns:
            1.0 for normal, 0.5/0.25/0.0 for throttled states.
        """
        if self._state.is_degraded:
            # Margin data is stale — apply 50% haircut as safety measure
            return 0.5

        util = self._state.utilization

        if util <= 0.80:
            return 1.0
        elif util <= 0.90:
            return 0.5
        elif util <= 0.95:
            return 0.25
        else:
            return 0.0  # no new allocations
```

---

### Priority Queue

When multiple strategies fire signals within the same event loop cycle (or within milliseconds of each other), the allocator processes them in a defined priority order. This ensures time-sensitive strategies (exits, expiry-day plays) get allocated first, before margin is consumed by less urgent strategies.

#### Priority Order

| Priority | Strategy | Rationale |
|----------|----------|-----------|
| 0 (highest) | EXIT (any strategy) | Exits reduce risk. Always process first. |
| 1 | S5 (Expiry Day) | 0-DTE, every second matters. Time decay is working against us. |
| 2 | S1 (ORB) | Opening range breakout is time-sensitive. Signal decays quickly. |
| 3 | S3 (VWAP MR) | Intraday mean reversion, moderate time sensitivity. |
| 4 | S2 (Overnight) | Fires at 15:15, has a few minutes before close. |
| 5 | S6 (Vol Premium) | Patient vol selling, no urgency. |
| 6 | S7 (Pairs) | Pairs trading, moderate patience. |
| 7 (lowest) | S4 (Momentum) | Monthly rebalance, large orders, most patient. |

The priority order is applied identically across all accounts. If S1 and S3 fire simultaneously, S1 is allocated first for ALL accounts, then S3 for ALL accounts.

#### Implementation

```python
import asyncio
from dataclasses import dataclass, field as dc_field


STRATEGY_PRIORITY: dict[str, int] = {
    "EXIT": 0,
    "S5": 1,
    "S1": 2,
    "S3": 3,
    "S2": 4,
    "S6": 5,
    "S7": 6,
    "S4": 7,
}


@dataclass(order=True)
class PrioritizedSignal:
    """
    Wrapper for signals in the priority queue.

    The @dataclass(order=True) decorator enables comparison by field order.
    priority is the first field, so lower priority values sort first
    (higher priority). sequence breaks ties in FIFO order.
    """
    priority: int
    sequence: int  # monotonic counter for FIFO within same priority
    signal: "StrategySignal" = dc_field(compare=False)
    is_exit: bool = dc_field(compare=False, default=False)


class SignalPriorityQueue:
    """
    Priority queue for strategy signals waiting for allocation.

    Uses asyncio.PriorityQueue (not queue.PriorityQueue) because
    the consumer is an async coroutine.

    Signals are enqueued by the Signal Router after dedup passes.
    The allocation loop dequeues in priority order and runs
    request_allocation for each (account, signal) pair.
    """

    def __init__(self, maxsize: int = 100) -> None:
        self._queue: asyncio.PriorityQueue[PrioritizedSignal] = (
            asyncio.PriorityQueue(maxsize=maxsize)
        )
        self._sequence: int = 0

    async def enqueue(self, signal: "StrategySignal", is_exit: bool = False) -> None:
        """
        Add a signal to the priority queue.

        Args:
            signal: The strategy signal to enqueue.
            is_exit: Whether this signal closes an existing position.
                    Exit signals always get priority 0.
        """
        if is_exit:
            priority = STRATEGY_PRIORITY["EXIT"]
        else:
            priority = STRATEGY_PRIORITY.get(signal.strategy_id, 99)

        item = PrioritizedSignal(
            priority=priority,
            sequence=self._sequence,
            signal=signal,
            is_exit=is_exit,
        )
        self._sequence += 1

        await self._queue.put(item)

        logger.debug("signal_enqueued",
                    strategy_id=signal.strategy_id,
                    priority=priority,
                    queue_size=self._queue.qsize())

    async def dequeue(self) -> PrioritizedSignal:
        """
        Get the highest-priority signal from the queue.

        Blocks until a signal is available.
        """
        return await self._queue.get()

    def qsize(self) -> int:
        """Current queue depth."""
        return self._queue.qsize()

    def empty(self) -> bool:
        """True if the queue is empty."""
        return self._queue.empty()
```

---

### Multi-Account Sizing: request_allocation

The `request_allocation` method is the main entry point for capital allocation. It runs once per `(account, signal)` pair. The multi-account fan-out loop in the Signal Router calls this N times for N accounts.

#### Sizing Formula

```
max_lots = floor(
    account.capital
    × erc_weight(strategy_id)
    × effective_kelly(account, strategy_id, trading_day)
    × margin_multiplier(account)
    / (estimated_premium × lot_size)
)
```

Each factor in the formula:

| Factor | Source | Scope | Update frequency |
|--------|--------|-------|-----------------|
| `account.capital` | `AccountConfig.capital` | Per-account | Immutable within session |
| `erc_weight` | `ERCComputer._current_weights` | Shared (all accounts) | Daily at 08:45 |
| `effective_kelly` | `KellyComputer.compute()` | Per-account, per-strategy | Daily (ramp) + dynamic (Sharpe override) |
| `margin_multiplier` | `MarginPoller.margin_multiplier()` | Per-account | Every 30s + on fill |
| `estimated_premium` | `AllocationRequest.estimated_premium` | Per-signal | Signal time |
| `lot_size` | `AllocationRequest.lot_size` | Per-instrument | Fixed by exchange |

#### Full CapitalAllocator Implementation

```python
@dataclass
class CapitalAllocator:
    """
    Central capital allocation engine.

    Receives AllocationRequests from the multi-account fan-out loop
    and returns AllocationResponses with sizing decisions.

    Thread-safety: all mutable state access is protected by asyncio.Lock.
    The lock is held for <1ms (dict lookups and arithmetic only).
    No I/O under lock — Redis reads for ERC/Kelly are pre-computed.

    Components:
        - SignalDeduplicator: shared, pre-fan-out
        - ERCComputer: shared weights
        - KellyComputer: per-account Kelly
        - MarginPoller: per-account margin (one per account)
        - SignalPriorityQueue: shared priority ordering

    Lifecycle:
        - Created once at session startup
        - initialize() called during Phase 3 (pre-market setup)
        - request_allocation() called on every (account, signal) pair
        - recompute_daily() called at 08:45 IST
    """

    redis: aioredis.Redis
    accounts: dict[str, AccountConfig]
    erc: ERCComputer
    kelly: KellyComputer
    margin_pollers: dict[str, MarginPoller]  # key: account_id
    deduplicator: SignalDeduplicator
    priority_queue: SignalPriorityQueue
    position_tracker: "PositionTracker"  # injected, not owned

    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _trading_day_count: int = 0
    _allocation_count: int = 0
    _rejection_count: int = 0

    async def initialize(
        self,
        session_start_date: str,
        trading_day_count: int,
    ) -> None:
        """
        Initialize all sub-components at session startup.

        Called during Phase 3 (pre-market setup), after Position Recovery
        has established current positions.

        Args:
            session_start_date: YYYY-MM-DD when the system first went live.
            trading_day_count: Trading days since session_start_date.
        """
        self._trading_day_count = trading_day_count
        await self.erc.initialize(session_start_date)
        logger.info("capital_allocator_initialized",
                   trading_day=trading_day_count,
                   accounts=list(self.accounts.keys()))

    async def recompute_daily(self, trading_day_count: int) -> None:
        """
        Recompute ERC weights and Kelly fractions. Called at 08:45 IST daily.

        This is NOT called under the allocation lock. ERC recomputation
        reads from Redis and writes to an in-memory dict atomically
        (Python dict assignment is atomic for single keys). No torn reads.
        """
        self._trading_day_count = trading_day_count
        await self.erc.recompute(trading_day_count)
        logger.info("daily_recompute_complete",
                   trading_day=trading_day_count,
                   erc_weights={sid: round(self.erc.get_weight(sid), 4)
                               for sid in STRATEGY_IDS})

    async def request_allocation(
        self,
        req: AllocationRequest,
        account: AccountConfig,
    ) -> AllocationResponse:
        """
        Allocate capital for one (account, signal) pair.

        This method is called by the multi-account fan-out loop:
            for account in accounts:
                allocation = await allocator.request_allocation(req, account)

        The asyncio.Lock serializes allocation decisions to prevent
        two concurrent allocations from double-spending margin headroom.
        The lock is held for <1ms (no I/O under lock).

        Args:
            req: The allocation request with signal details and existing position.
            account: The account configuration for this specific account.

        Returns:
            AllocationResponse with sizing decision.
        """
        async with self._lock:
            # Step 1: Check existing position (HIGH-5 fix)
            position_decision = self._check_existing_position(req)
            if position_decision is not None:
                self._allocation_count += 1
                if not position_decision.approved:
                    self._rejection_count += 1
                return position_decision

            # Step 2: Get margin state for this account
            margin_poller = self.margin_pollers.get(req.account_id)
            if margin_poller is None:
                logger.error("margin_poller_not_found",
                           account_id=req.account_id)
                return AllocationResponse(
                    approved=False,
                    rejection_reason=f"no_margin_poller_for_account:{req.account_id}",
                )

            margin_mult = margin_poller.margin_multiplier()
            margin_util = margin_poller.state.utilization

            if margin_mult == 0.0:
                self._rejection_count += 1
                self._allocation_count += 1
                return AllocationResponse(
                    approved=False,
                    rejection_reason=(
                        f"margin_utilization_too_high:"
                        f"{margin_util:.1%} for account {req.account_id}"
                    ),
                    margin_util_at_allocation=margin_util,
                )

            # Step 3: Get ERC weight (shared, in-memory, fast)
            erc_weight = self.erc.get_weight(req.strategy_id)

            if erc_weight <= 0:
                self._rejection_count += 1
                self._allocation_count += 1
                return AllocationResponse(
                    approved=False,
                    rejection_reason=f"erc_weight_zero_for:{req.strategy_id}",
                )

            # Step 4: Compute effective Kelly (may read Redis for Sharpe,
            # but Sharpe is cached — fast path)
            # NOTE: This is the one async call under lock. The Redis read
            # is a single GET, typically <0.5ms on localhost Redis.
            effective_kelly = await self.kelly.compute(
                account=account,
                strategy_id=req.strategy_id,
                trading_day_count=self._trading_day_count,
            )

            # Step 5: Compute sizing
            raw_capital = (
                account.capital
                * erc_weight
                * effective_kelly
                * margin_mult
            )

            if raw_capital <= 0:
                self._rejection_count += 1
                self._allocation_count += 1
                return AllocationResponse(
                    approved=False,
                    rejection_reason="computed_capital_zero_or_negative",
                    kelly_fraction=effective_kelly,
                    erc_weight=erc_weight,
                    margin_util_at_allocation=margin_util,
                )

            cost_per_lot = req.estimated_premium * req.lot_size
            if cost_per_lot <= 0:
                self._rejection_count += 1
                self._allocation_count += 1
                return AllocationResponse(
                    approved=False,
                    rejection_reason="estimated_premium_or_lot_size_invalid",
                )

            raw_lots = raw_capital / cost_per_lot
            max_lots = int(raw_lots)  # floor

            if max_lots < 1:
                self._rejection_count += 1
                self._allocation_count += 1
                return AllocationResponse(
                    approved=False,
                    rejection_reason=(
                        f"insufficient_capital_for_minimum_lot:"
                        f" raw_lots={raw_lots:.2f},"
                        f" capital={raw_capital:.0f},"
                        f" cost_per_lot={cost_per_lot:.0f}"
                    ),
                    kelly_fraction=effective_kelly,
                    erc_weight=erc_weight,
                    margin_util_at_allocation=margin_util,
                )

            allocated_capital = max_lots * cost_per_lot

            self._allocation_count += 1

            logger.info("allocation_approved",
                       account_id=req.account_id,
                       strategy_id=req.strategy_id,
                       direction=req.direction,
                       max_lots=max_lots,
                       allocated_capital=round(allocated_capital, 2),
                       erc_weight=round(erc_weight, 4),
                       kelly=round(effective_kelly, 4),
                       margin_mult=margin_mult,
                       margin_util=round(margin_util, 3),
                       raw_lots=round(raw_lots, 2))

            return AllocationResponse(
                approved=True,
                allocated_capital=allocated_capital,
                max_lots=max_lots,
                kelly_fraction=effective_kelly,
                erc_weight=erc_weight,
                is_exit=False,
                rejection_reason=None,
                margin_util_at_allocation=margin_util,
            )

    def _check_existing_position(
        self,
        req: AllocationRequest,
    ) -> AllocationResponse | None:
        """
        Check existing position for this (account, strategy) pair.

        Returns:
            AllocationResponse if a decision can be made immediately
            (reject same-direction, approve exit). Returns None if the
            position is flat and normal allocation should proceed.
        """
        qty = req.existing_position_qty

        if qty == 0:
            return None

        is_long = qty > 0
        signal_is_long = req.direction == "LONG"
        same_direction = (is_long and signal_is_long) or (not is_long and not signal_is_long)

        if same_direction:
            logger.warning("allocation_rejected_position_exists",
                          account_id=req.account_id,
                          strategy_id=req.strategy_id,
                          direction=req.direction,
                          existing_qty=qty)
            return AllocationResponse(
                approved=False,
                rejection_reason=(
                    f"strategy_already_positioned_same_direction: "
                    f"{req.strategy_id} holds {qty} units on account {req.account_id}"
                ),
            )

        logger.info("allocation_exit_signal",
                   account_id=req.account_id,
                   strategy_id=req.strategy_id,
                   direction=req.direction,
                   existing_qty=qty)
        return AllocationResponse(
            approved=True,
            allocated_capital=0.0,
            max_lots=abs(qty) // req.lot_size,
            kelly_fraction=0.0,
            erc_weight=0.0,
            is_exit=True,
            rejection_reason=None,
        )

    def stats(self) -> dict[str, int]:
        """Return allocator statistics for monitoring."""
        return {
            "total_allocations": self._allocation_count,
            "total_rejections": self._rejection_count,
            "dedup": self.deduplicator.stats(),
            "queue_depth": self.priority_queue.qsize(),
        }
```

---

### State Table

| State Item | Storage | Scope | Lifecycle | Writer | Readers |
|-----------|---------|-------|-----------|--------|---------|
| ERC weights | Redis `ALLOC:erc_weights` | **Shared** (all accounts use same weights) | Daily recompute at 08:45 IST. Survives restart within same day. | `ERCComputer.recompute()` | `CapitalAllocator.request_allocation()` (reads in-memory cache) |
| ERC weights (in-memory cache) | In-memory dict in `ERCComputer` | **Shared** | Session. Initialized from Redis on startup. | `ERCComputer.recompute()` | `ERCComputer.get_weight()` (hot path, no async) |
| Sharpe ratios (20d) | Redis `SHARPE:{sid}:20d` | **Shared** (per-strategy, not per-account) | Daily, written by Position Tracker at EOD | Position Tracker | `KellyComputer._get_strategy_sharpe()` |
| Daily returns (for vol) | Redis `RETURNS:{sid}:daily` (list) | **Shared** (per-strategy) | Appended daily by Position Tracker | Position Tracker | `ERCComputer._compute_erc_from_returns()` |
| Margin state (in-memory) | In-memory `MarginState` in `MarginPoller` | **Per-account** | Updated every 30s + on fill. Reset on restart. | `MarginPoller._poll()` | `CapitalAllocator.request_allocation()` |
| Margin state (Redis) | Redis `MARGIN:{account_id}` | **Per-account** | Updated every 30s + on fill | `MarginPoller._poll()` | Monitoring dashboards (Grafana) |
| Kelly fractions (effective) | Computed on-the-fly | **Per-account, per-strategy** | Computed on every allocation request | `KellyComputer.compute()` | `CapitalAllocator.request_allocation()` |
| Kelly fractions (base config) | `AccountConfig.kelly_fraction` | **Per-account** | Immutable within session | Config file (startup) | `KellyComputer.compute()` |
| Dedup state | In-memory dict in `SignalDeduplicator` | **Shared** (pre-fan-out) | Session. Reset on restart. | `SignalDeduplicator.check()` | Signal Router (pre-fan-out) |
| Signal priority queue | In-memory `asyncio.PriorityQueue` | **Shared** | Session. Drained continuously. | Signal Router (enqueue) | Allocation loop (dequeue) |
| Trading day count | In-memory int in `CapitalAllocator` | **Shared** | Set at startup, incremented daily | Session Orchestrator | `ERCComputer`, `KellyComputer` |
| Allocation counters | In-memory ints in `CapitalAllocator` | **Shared** | Session. Reset on restart. | `CapitalAllocator.request_allocation()` | Monitoring (`stats()`) |
| Account capital | `AccountConfig.capital` | **Per-account** | Immutable within session | Config file (startup) | `CapitalAllocator.request_allocation()` |
| Existing position qty | Read from PositionTracker API | **Per-account, per-strategy** | Real-time (reflects all fills) | Position Tracker (via OMS fills) | `CapitalAllocator._check_existing_position()` |

**Key distinction:** ERC weights are **shared** because they reflect strategy-level performance (the strategy generates the same signal for all accounts). Kelly fractions are **per-account** because different accounts have different risk tolerances (`kelly_fraction` config). Margin is **per-account** because each Dhan account has its own margin pool.

---

### DuckDB Read Pattern

The Capital Allocator reads existing position quantities from the PositionTracker API. It never queries DuckDB directly.

#### Why Indirect Access

1. **Single writer principle.** The PositionTracker owns the DuckDB write path. If the allocator also reads DuckDB directly, it creates a hidden coupling: schema changes in the PositionTracker's DuckDB tables would silently break the allocator. The API contract (`get_position_qty(account_id, strategy_id) -> int`) is stable even if the underlying storage changes.

2. **Consistency model.** The PositionTracker maintains an in-memory cache of positions that is updated synchronously on every fill. A direct DuckDB read could return stale data if the WAL has not flushed. The PositionTracker's in-memory state is always the most current.

3. **Locking semantics.** DuckDB uses a single-writer, multi-reader lock at the file level. The PositionTracker already holds the write lock. A concurrent read from the allocator would work (DuckDB supports it), but adds unnecessary contention on the WAL. Reading from the PositionTracker's in-memory cache is lock-free.

4. **Testability.** The allocator can be tested with a mock PositionTracker that returns arbitrary position states. No DuckDB setup required in unit tests.

#### Access Pattern

```python
# In the Signal Router's fan-out loop:
existing_qty = position_tracker.get_position_qty(
    account_id=account.account_id,
    strategy_id=signal.strategy_id,
)

# This calls into PositionTracker's in-memory dict:
# _positions: dict[tuple[str, str], int]  # (account_id, strategy_id) → signed qty

# The allocator NEVER does this:
# conn = duckdb.connect("positions.db")
# conn.execute("SELECT qty FROM positions WHERE ...")
```

---

### Failure Modes

| # | Failure | Detection | Impact | Recovery |
|---|---------|-----------|--------|----------|
| 1 | **Redis unavailable during ERC recompute** | `ConnectionError` in `ERCComputer.recompute()` | Cannot read daily returns or write new weights. | Fall back to last known in-memory weights (from previous day or static). Log CRITICAL alert. Retry on next 30s cycle. Trading continues with stale weights. |
| 2 | **Margin poll fails for one account** | HTTP error or timeout in `MarginPoller._poll()` | That account's margin state becomes stale. | Increment `consecutive_poll_failures`. After 3 failures: apply 50% safety haircut to that account's allocations. Other accounts unaffected. |
| 3 | **Margin poll fails for ALL accounts** | All `MarginPoller` instances have `consecutive_poll_failures > 3` | All accounts in degraded margin mode. System-wide 50% allocation reduction. | Telegram CRITICAL alert. Operator can either wait for Dhan API recovery or manually set margin state via Redis CLI. |
| 4 | **PositionTracker returns stale position data** | No direct detection (silent failure). | Allocator may approve an allocation for a strategy that already holds a position (HIGH-5 re-emergence). | PositionTracker updates in-memory cache synchronously on fill. Staleness only possible if PositionTracker itself has crashed. If PositionTracker is down, the orchestrator detects it via heartbeat and halts all trading. |
| 5 | **asyncio.Lock contention during burst** | Lock wait time exceeds 100ms (monitored via timing around `async with self._lock`) | Allocation latency increases for queued signals. | The lock body is <1ms (arithmetic only). Contention implies >1000 signals/second, which exceeds system design capacity. If detected: log WARNING, investigate signal source (likely a misbehaving strategy flooding signals). |
| 6 | **ERC weight becomes 0 for a strategy** | `erc_weight <= 0` check in `request_allocation()` | That strategy receives no allocations for the day. | This happens when the strategy had near-zero vol (all returns were identical). The 1e-6 vol floor in `_compute_erc_from_returns` prevents exact zero. If the floor itself produces a rounding-to-zero weight: the allocation is rejected with reason `erc_weight_zero_for:{sid}`, logged, and the operator can manually override via Redis. |
| 7 | **Account capital set to wrong value in config** | No automatic detection. | Account is over-sized or under-sized for the session. | Capital is immutable within a session. Fix requires config change + restart. Pre-market validation script checks that sum of account capitals does not exceed total portfolio capital. |
| 8 | **Dedup clock skew** | `signal_ts` in the future relative to system clock | Cooldown computation is incorrect. Signal may be suppressed for too long or not long enough. | Strategy signals use `time.time()` from the same EC2 instance (NTP-synced). Clock skew is <1ms. If detected (signal_ts > now + 5s): reject the signal and log ERROR. |
| 9 | **Priority queue full (maxsize=100)** | `asyncio.PriorityQueue.put()` blocks | New signals wait for queue space. Allocation latency increases. | Queue size of 100 is generous — 7 strategies firing simultaneously produces 7 entries. Full queue implies a downstream consumer is stuck. The allocation loop has a 5s watchdog: if no signal is dequeued for 5s while queue is non-empty, log CRITICAL and investigate. |
| 10 | **Sharpe read returns NaN or invalid float** | `float(raw)` raises `ValueError` | Kelly override logic fails. | Catch `ValueError` in `_get_strategy_sharpe()`, return `None`. `None` Sharpe means "insufficient data" — no override applied. Log WARNING. |
| 11 | **Account removed from config but margin poller still running** | `margin_pollers.get(req.account_id)` returns `None` | Allocation for orphan account fails. | Return rejection with reason `no_margin_poller_for_account`. This should not happen in practice (accounts are immutable within a session), but the check prevents a crash. |
| 12 | **Two strategies fire at exactly the same timestamp for the same underlying** | Both pass dedup (different strategy_ids) | Priority queue orders them correctly. No issue unless both consume more than available margin. | The sequential allocation under `asyncio.Lock` prevents double-spend: the first allocation reduces available margin (via the margin multiplier), and the second allocation sees the updated state. |

---

### Edge Cases

#### 1. Strategy fires for the first time on Day 1 (no return history)

On Day 1, `_compute_erc_from_returns()` returns `None` because no strategy has 20 data points. The allocator uses static weights. Kelly ramp is at 0.25x. The first trade is sized conservatively by design.

#### 2. Account has capital < cost of 1 lot

An account with ₹50,000 capital and a BANKNIFTY option premium of ₹300 at lot size 15 needs ₹4,500 per lot minimum. After ERC weight (say 15%) and Kelly (say 0.10): available = ₹50,000 × 0.15 × 0.10 = ₹750. This is less than ₹4,500 per lot.

The allocator computes `raw_lots = 750 / 4500 = 0.167`, floors to 0, and rejects with `insufficient_capital_for_minimum_lot`. The account simply does not participate in this trade. No error — this is expected behavior for small accounts.

#### 3. Exit signal arrives but PositionTracker reports qty=0

The signal says SHORT but the account is already flat for this strategy. This happens when:

- The position was stopped out by a server-side SL while the system was down
- The EOD flatten closed the position but the strategy didn't notice
- A race between fill processing and signal generation

The allocator sees `existing_position_qty=0`, treats this as a new SHORT entry (not an exit), and proceeds with normal allocation. The downstream Risk Gate validates whether a new short position is acceptable.

#### 4. Margin utilization oscillates around 80% threshold

The account's margin utilization is 79% at time T, 81% at T+30s, 78% at T+60s. An allocation request at T+35s sees 81% and applies the 50% multiplier. An identical request at T+65s sees 78% and gets full allocation.

This is correct behavior. The throttling is instantaneous and reflects the latest margin state. There is no hysteresis — adding hysteresis would increase complexity without meaningful benefit, because margin utilization changes discretely on fills, not continuously.

#### 5. All 7 strategies fire simultaneously at market open

All signals pass dedup (different strategy_ids). The priority queue orders them: S5 > S1 > S3 > S2 > S6 > S7 > S4. Each allocation is processed sequentially under the asyncio.Lock. Total processing time: ~7ms (7 signals × <1ms each).

After the first few allocations consume margin, later allocations may see higher margin utilization and get throttled. This is the intended behavior: high-priority strategies get first access to margin.

#### 6. Kelly ramp crosses the Day 20 boundary mid-session

The trading day count is set at session startup (08:00 IST) and does not change mid-session. If today is Day 20, all allocations throughout the day use Day 20's Kelly multiplier (0.25x). The transition to Day 21 (ramp begins) happens at the next session startup.

This prevents a mid-day Kelly change that would make morning and afternoon allocations inconsistent. The transition is always at session boundary.

#### 7. ERC weight cap triggers for a strategy

If a strategy's raw ERC weight exceeds 40% (because all other strategies had high volatility and this one was very stable), the weight is capped at 40% and all weights are renormalized. This prevents a single low-vol strategy from consuming nearly all capital, which would be dangerous if that strategy's low vol was caused by not trading (no returns = no vol) rather than genuinely low risk.

#### 8. Multi-account: Account A is at 92% margin, Account B is at 30%

S1 fires. For Account A: margin multiplier is 0.25 (90-95% band), so allocation is 25% of normal. For Account B: margin multiplier is 1.0, full allocation. The resulting position sizes diverge — Account A gets 2 lots, Account B gets 8 lots. This is correct: each account is sized independently based on its own margin state.

The DivergenceTracker (in the OMS) records this divergence for compliance reporting, but no corrective action is taken. The accounts are expected to diverge when their margin states differ.

---


## Component 5: Order Management System (OMS)

### Responsibility

- Place limit orders paired with mandatory server-side stop-loss orders on the Dhan broker API
- Fan out orders to N accounts (multi-account replication): one signal produces N independent order lifecycles
- Track order status via Dhan's live order update WebSocket (primary) with REST fallback
- Execute per-order fill management loops with per-order locking and sequence numbers
- Manage SL lifecycle as a first-class concern: pairing, qty sync on partial fills, overnight conversion, verification
- Rate-limit API calls via per-account priority token buckets (10 OPS per Dhan account)
- Flatten all intraday positions at EOD via batched parallel IOC-escalating exit sequence
- Operate in degraded mode when Redis is unavailable
- Enforce a global kill switch that cancels everything and flattens all positions across all accounts
- **Does NOT** generate signals, resolve instruments, allocate capital, or manage positions (those are upstream components)

---

### Dhan API Integration

All broker interaction goes through the Dhan HTTP REST API and WebSocket API. This section documents the exact endpoints, request/response schemas, and integration patterns.

**Base URL:** `https://api.dhan.co`
**Auth header:** `access-token: {dhan_access_token}` (per-account, obtained during daily login)
**Content-Type:** `application/json`
**Rate limit:** 10 orders per second per API key (hard limit, enforced server-side)

#### Order Placement: POST /v2/orders

Places a new order on the exchange.

```python
class DhanOrderRequest(pydantic.BaseModel):
    """Exact request body for POST /v2/orders."""
    dhanClientId: str                # Dhan client ID for this account
    transactionType: Literal["BUY", "SELL"]
    exchangeSegment: Literal["NSE_EQ", "NSE_FNO", "BSE_EQ", "BSE_FNO"]
    productType: Literal["INTRADAY", "CNC", "MARGIN", "MTF", "CO", "BO"]
    orderType: Literal["LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_MARKET"]
    validity: Literal["DAY", "IOC"]
    securityId: str                  # Dhan's internal security ID (from instrument master CSV)
    quantity: int                    # total order quantity in units (not lots)
    price: float                    # limit price (required for LIMIT and STOP_LOSS)
    triggerPrice: float | None      # required for STOP_LOSS and STOP_LOSS_MARKET
    disclosedQuantity: int | None   # portion visible to market (0 = full visibility)
    afterMarketOrder: bool          # True for AMO (placed after market hours, executed at open)
    amoTime: Literal["OPEN", "OPEN_30", "OPEN_60"] | None  # when AMO activates
    boProfitValue: float | None     # bracket order profit target (not used in v1)
    boStopLossValue: float | None   # bracket order SL (not used in v1)
    correlationId: str | None       # our internal reference (max 25 chars)
    # NOTE: correlationId is returned in order updates but NOT used by Dhan
    # for dedup. Two identical orders with the same correlationId = two orders.

class DhanOrderResponse(pydantic.BaseModel):
    """Response from POST /v2/orders on success (HTTP 200)."""
    orderId: str                    # Dhan's order ID (unique, used for all subsequent operations)
    orderStatus: Literal["TRANSIT", "PENDING", "REJECTED"]
    # TRANSIT = sent to exchange, awaiting acknowledgement
    # PENDING = accepted by exchange, waiting in order book
    # REJECTED = rejected by Dhan's risk checks or exchange

class DhanErrorResponse(pydantic.BaseModel):
    """Response on failure (HTTP 4xx/5xx)."""
    status: str                     # "failure"
    remarks: str                    # human-readable error
    errorCode: str | None           # e.g., "DH-906" (insufficient margin)
    errorType: str | None           # e.g., "Input_Exception", "Order_Exception"
```

**Example: Place a LIMIT BUY for 650 qty of NIFTY CE option:**

```python
request = DhanOrderRequest(
    dhanClientId="1000000001",
    transactionType="BUY",
    exchangeSegment="NSE_FNO",
    productType="INTRADAY",
    orderType="LIMIT",
    validity="DAY",
    securityId="43925",              # NIFTY 24500 CE weekly
    quantity=650,                     # 10 lots × 65 per lot
    price=245.50,
    triggerPrice=None,
    disclosedQuantity=0,
    afterMarketOrder=False,
    amoTime=None,
    boProfitValue=None,
    boStopLossValue=None,
    correlationId="S1_a3f8c1d2e4b6",
)

# POST https://api.dhan.co/v2/orders
# Headers: {"access-token": "...", "Content-Type": "application/json"}
# Body: request.model_dump_json()
```

#### Order Modification: PUT /v2/orders/{order-id}

Modifies price, quantity, order type, or validity of a pending order.

```python
class DhanModifyRequest(pydantic.BaseModel):
    """Request body for PUT /v2/orders/{orderId}."""
    dhanClientId: str
    orderId: str
    orderType: Literal["LIMIT", "MARKET", "STOP_LOSS", "STOP_LOSS_MARKET"]
    legName: str | None              # for multi-leg orders (not used in v1)
    quantity: int                    # CRITICAL: this is the ORIGINAL TOTAL quantity,
                                     # NOT the remaining quantity. If you placed 650
                                     # and 325 filled, you still send quantity=650
                                     # to keep the remaining 325 alive.
    price: float
    triggerPrice: float | None
    disclosedQuantity: int | None
    validity: Literal["DAY", "IOC"]

class DhanModifyResponse(pydantic.BaseModel):
    """Response from PUT /v2/orders/{orderId}."""
    orderId: str
    orderStatus: str                 # typically "TRANSIT" or "PENDING"
```

**The quantity=original_total gotcha:**

Dhan's modify endpoint interprets `quantity` as the total order quantity, not the remaining unfilled quantity. This is a critical integration detail:

```
Scenario:
  1. Place order: qty=650
  2. Partial fill: 325 filled, 325 remaining
  3. Want to reprice the remaining 325

  CORRECT:   PUT /v2/orders/{id}  quantity=650  price=new_price
             → Dhan keeps remaining_qty=325, updates price

  WRONG:     PUT /v2/orders/{id}  quantity=325  price=new_price
             → Dhan interprets this as "reduce total qty to 325"
             → Since 325 already filled, remaining becomes 0
             → Order cancelled! 325 shares bought at old price, no more fills.

  CRITICAL:  Always pass the ORIGINAL quantity from the initial placement.
             Store original_qty in OrderState at creation time. Never change it.
```

**25-modification limit:**

Dhan allows a maximum of 25 modifications per order. After 25 mods, further modify requests are rejected. The OMS tracks `modification_count` per order and switches to cancel-replace at modification 24 (one before the limit, as a safety margin):

```python
MAX_MODIFICATIONS = 24  # switch to cancel-replace at 24, not 25

async def reprice_order(self, state: OrderState, order: ResolvedOrder,
                        params: FillParams, account: Account) -> None:
    new_price = self.compute_new_limit_price(order, params)
    rate_limiter = self._account_rate_limiters[account.account_id]

    if state.modification_count >= MAX_MODIFICATIONS:
        # Cancel-replace: cancel the old order, place a new one
        await rate_limiter.acquire("MODIFY")
        try:
            await self._dhan_cancel_order(account, state.order_id)
        except OrderAlreadyFilled:
            return  # race: filled between check and cancel — fine

        await rate_limiter.acquire("MODIFY")
        new_resp = await self._dhan_place_order(account, DhanOrderRequest(
            dhanClientId=account.dhan_client_id,
            transactionType=order.transaction_type,
            exchangeSegment=order.exchange_segment,
            productType=self._product_type(order),
            orderType="LIMIT",
            validity="DAY",
            securityId=order.instrument_id,
            quantity=state.remaining_qty,  # NEW order: use remaining, not original
            price=new_price,
            triggerPrice=None,
            disclosedQuantity=0,
            afterMarketOrder=False,
            amoTime=None,
            boProfitValue=None,
            boStopLossValue=None,
            correlationId=f"CR_{order.signal.strategy_id}_{order.signal.signal_id[:12]}",
        ))

        if new_resp.orderStatus == "REJECTED":
            logger.error("cancel_replace_rejected",
                        account_id=account.account_id,
                        old_order=state.order_id)
            return

        # Update state to track the new order
        old_order_id = state.order_id
        state.order_id = new_resp.orderId
        state.modification_count = 0
        state.original_qty = state.remaining_qty  # new order, new original
        state.current_limit_price = new_price

        # Update SL pairing to reference new entry order ID
        await self.sl_manager.reparent_sl(old_order_id, state.order_id, account)

        logger.info("cancel_replace_completed",
                   account_id=account.account_id,
                   old_order=old_order_id,
                   new_order=state.order_id,
                   new_price=new_price)
    else:
        # Normal modify
        await rate_limiter.acquire("MODIFY")
        try:
            await self._dhan_modify_order(account, DhanModifyRequest(
                dhanClientId=account.dhan_client_id,
                orderId=state.order_id,
                orderType="LIMIT",
                legName=None,
                quantity=state.original_qty,  # ALWAYS original total qty
                price=new_price,
                triggerPrice=None,
                disclosedQuantity=0,
                validity="DAY",
            ))
            state.modification_count += 1
            state.current_limit_price = new_price

        except OrderNotModifiable:
            # Race: filled or cancelled between our check and modify
            await rate_limiter.acquire("POLL")
            fresh = await self._dhan_get_order_status(account, state.order_id)
            state.apply_update(fresh)
            # Next loop iteration handles the new status

        # Post-modify verification: confirm the modify was actually applied
        await rate_limiter.acquire("POLL")
        verified = await self._dhan_get_order_status(account, state.order_id)
        state.apply_update(verified)
```

#### Order Cancellation: DELETE /v2/orders/{order-id}

```python
# DELETE https://api.dhan.co/v2/orders/{orderId}
# Headers: {"access-token": "..."}
# No request body.

class DhanCancelResponse(pydantic.BaseModel):
    orderId: str
    orderStatus: str  # "CANCELLED" on success
```

**Cancellation edge cases:**
- Order already filled → Dhan returns error. Catch and treat as filled.
- Order already cancelled → Dhan returns error. Idempotent — ignore.
- Order in TRANSIT (not yet on exchange) → Cancellation may fail. Retry after 500ms.

#### Order Slicing: POST /v2/orders/slicing

Used when order quantity exceeds the exchange's freeze quantity limit. NSE sets per-instrument freeze limits (e.g., NIFTY options: 1800 qty = ~27 lots). Orders above this limit must be sliced into multiple child orders.

```python
class DhanSlicingRequest(pydantic.BaseModel):
    """Request body for POST /v2/orders/slicing.
    Same fields as regular order, but Dhan auto-splits into children."""
    dhanClientId: str
    transactionType: Literal["BUY", "SELL"]
    exchangeSegment: str
    productType: str
    orderType: str
    validity: str
    securityId: str
    quantity: int                    # total qty (may exceed freeze limit)
    price: float
    triggerPrice: float | None
    disclosedQuantity: int | None
    afterMarketOrder: bool
    amoTime: str | None
    boProfitValue: float | None
    boStopLossValue: float | None
    correlationId: str | None

class DhanSlicingResponse(pydantic.BaseModel):
    """Response: list of child order IDs."""
    orderId: str                    # parent order ID
    # Note: Dhan creates child orders automatically.
    # Each child appears as a separate order in order updates.
    # The parent orderId is referenced in child updates.
```

**When to use slicing:**

```python
def should_slice(self, quantity: int, freeze_qty: int) -> bool:
    """Check if order needs slicing based on exchange freeze limits."""
    return quantity > freeze_qty

# freeze_qty comes from the instrument master CSV
# Examples:
#   NIFTY options: freeze_qty = 1800 (27.7 lots at 65/lot)
#   BANKNIFTY options: freeze_qty = 900 (60 lots at 15/lot)
#   NIFTY futures: freeze_qty = 1800
```

**OMS handling of sliced orders:** When the OMS detects `quantity > freeze_qty`, it uses the slicing endpoint instead of the regular order endpoint. The fill management loop then tracks each child order independently. SL is placed for the TOTAL quantity (SL orders are not subject to freeze limits on Dhan).

#### Order Status: GET /v2/orders/{order-id}

REST endpoint for polling order status. Used as fallback when WS is unhealthy and for post-modify verification.

```python
# GET https://api.dhan.co/v2/orders/{orderId}
# Headers: {"access-token": "..."}

class DhanOrderDetail(pydantic.BaseModel):
    """Response from GET /v2/orders/{orderId}."""
    orderId: str
    correlationId: str | None
    orderStatus: Literal[
        "TRANSIT",        # sent to exchange, not yet acknowledged
        "PENDING",        # on exchange, waiting to match
        "TRADED",         # fully filled
        "PART_TRADED",    # partially filled, remaining still pending
        "CANCELLED",      # cancelled (by us or exchange)
        "REJECTED",       # rejected by Dhan risk or exchange
        "EXPIRED",        # DAY order expired at session close
    ]
    transactionType: str
    exchangeSegment: str
    productType: str
    orderType: str
    validity: str
    securityId: str
    quantity: int                    # original total quantity
    filledQty: int                   # how many units filled so far
    remainingQuantity: int           # quantity - filledQty
    price: float                     # current limit price
    triggerPrice: float | None
    averageTradedPrice: float        # volume-weighted avg fill price
    exchangeOrderId: str | None      # exchange's internal order ID
    exchangeTime: str | None         # exchange timestamp (ISO 8601)
    createTime: str                  # order creation time
    updateTime: str                  # last update time
    legName: str | None
    drvExpiryDate: str | None        # derivative expiry (YYYY-MM-DD)
    drvOptionType: str | None        # "CALL" | "PUT"
    drvStrikePrice: float | None
```

**PART_TRADED handling:**

When `orderStatus == "PART_TRADED"`, the order is partially filled with remaining quantity still pending on the exchange. Critical actions:

1. `filledQty` tells us how many units have been filled
2. The paired SL order qty MUST be updated to match `filledQty` (not the original qty)
3. The fill management loop decides whether to continue waiting (within patience) or abandon (cancel remaining, keep partial fill + adjusted SL)
4. `averageTradedPrice` gives the weighted average across all partial fills so far

#### Trades of Order: GET /v2/trades/{order-id}

Returns individual trade executions for an order. A single order may fill in multiple trades at different prices and quantities.

```python
# GET https://api.dhan.co/v2/trades/{orderId}
# Headers: {"access-token": "..."}

class DhanTradeDetail(pydantic.BaseModel):
    """One trade execution within an order."""
    orderId: str
    exchangeOrderId: str
    tradingSymbol: str
    securityId: str
    transactionType: str
    exchangeSegment: str
    productType: str
    orderType: str
    tradedQuantity: int              # qty filled in THIS trade
    tradedPrice: float               # price of THIS trade
    exchangeTime: str                # exchange timestamp for this trade
    drvExpiryDate: str | None
    drvOptionType: str | None
    drvStrikePrice: float | None

class DhanTradesResponse(pydantic.BaseModel):
    """Response: list of trades for this order."""
    trades: list[DhanTradeDetail]
```

**Why GET /v2/trades matters:** The `averageTradedPrice` in the order detail endpoint is the volume-weighted average. But for accurate cost analysis and audit trail, we need each individual trade (price × qty). The Position Tracker calls this endpoint after every order completion to record per-trade details.

#### Live Order Update WebSocket: wss://api-order-update.dhan.co

Primary channel for real-time order status updates. One WebSocket connection per account.

**Connection and authentication:**

```python
class DhanOrderWS:
    """
    WebSocket client for Dhan live order updates.

    One instance per account. Authenticates with the account's access token.
    Receives real-time updates for all orders on that account.
    """

    WS_URL = "wss://api-order-update.dhan.co"

    def __init__(self, account: "Account"):
        self.account = account
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._connected = False

    async def connect(self) -> None:
        """Connect and authenticate."""
        self._ws = await websockets.connect(
            self.WS_URL,
            extra_headers={
                "access-token": self.account.dhan_access_token,
            },
            ping_interval=30,        # send ping every 30s to keep alive
            ping_timeout=10,
            close_timeout=5,
        )

        # Send authentication message
        auth_msg = {
            "LoginReq": {
                "MsgCode": 42,
                "ClientId": self.account.dhan_client_id,
                "Token": self.account.dhan_access_token,
            }
        }
        await self._ws.send(orjson.dumps(auth_msg))

        # Wait for auth response
        resp = await asyncio.wait_for(self._ws.recv(), timeout=10.0)
        auth_resp = orjson.loads(resp)
        if auth_resp.get("type") == "order_alert":
            # Dhan sends existing pending orders as first messages after auth
            # Process them to sync state
            pass
        self._connected = True
        logger.info("dhan_order_ws_connected",
                   account_id=self.account.account_id)
```

**Order update message schema (from Dhan WS):**

```python
class DhanWSOrderUpdate(pydantic.BaseModel):
    """
    Schema of order update messages received on the Dhan order WS.

    Dhan sends these in real-time when any order status changes:
    placed, modified, partially filled, fully filled, cancelled, rejected.
    """
    type: str                        # "order_alert"
    orderId: str
    correlationId: str | None
    orderStatus: Literal[
        "TRANSIT", "PENDING", "TRADED",
        "PART_TRADED", "CANCELLED", "REJECTED", "EXPIRED"
    ]
    transactionType: str
    exchangeSegment: str
    productType: str
    orderType: str
    validity: str
    securityId: str
    quantity: int                    # original total quantity
    filledQty: int
    remainingQuantity: int
    price: float
    triggerPrice: float | None
    averageTradedPrice: float
    exchangeOrderId: str | None
    exchangeTime: str | None
    updateTime: str
```

**WS Demuxer design:**

The demuxer receives all order updates on a per-account WS connection and routes them to the correct per-order fill management loop.

```python
class OrderUpdateDemuxer:
    """
    Receives order updates from Dhan WS and routes them to the correct
    OrderState object. One demuxer per account.

    Architecture:
    1. WS reader task continuously reads from the WebSocket
    2. For each update, look up the OrderState by orderId
    3. Acquire the per-order lock
    4. Apply the update (increment sequence, update fields)
    5. Signal the fill management loop via update_event
    """

    def __init__(self, account: "Account"):
        self.account = account
        self._order_states: dict[str, "OrderState"] = {}

    def register_order(self, order_id: str, state: "OrderState") -> None:
        """Register a new order for update routing."""
        self._order_states[order_id] = state

    def unregister_order(self, order_id: str) -> None:
        """Unregister a completed/cancelled order."""
        self._order_states.pop(order_id, None)

    async def run(self, ws: DhanOrderWS) -> None:
        """
        Main demuxer loop. Reads from WS, routes to OrderState.

        This runs as an asyncio task for the lifetime of the WS connection.
        If the WS disconnects, this task exits and is restarted by
        _restart_on_error wrapper.
        """
        while True:
            raw = await ws.recv()
            try:
                update = DhanWSOrderUpdate.model_validate_json(raw)
            except pydantic.ValidationError:
                logger.warning("ws_update_parse_error",
                             account_id=self.account.account_id,
                             raw=raw[:200])
                continue

            if update.type != "order_alert":
                continue

            state = self._order_states.get(update.orderId)
            if state is None:
                # Order we're not tracking (e.g., manual order placed in Dhan app)
                logger.debug("ws_update_unknown_order",
                           account_id=self.account.account_id,
                           order_id=update.orderId)
                continue

            # Apply update under per-order lock
            async with state.lock:
                state.apply_ws_update(update)
                state.update_event.set()

            logger.debug("ws_update_routed",
                       account_id=self.account.account_id,
                       order_id=update.orderId,
                       status=update.orderStatus,
                       filled=update.filledQty)
```

**WS health monitoring:**

```python
class WSHealthMonitor:
    """Tracks WS connection health per account."""

    def __init__(self, account_id: str):
        self.account_id = account_id
        self.is_healthy: bool = False
        self.last_message_ts: int = 0
        self.reconnect_count: int = 0
        self.disconnect_ts: int | None = None

    def on_message(self) -> None:
        self.last_message_ts = now_ms()
        self.is_healthy = True

    def on_disconnect(self) -> None:
        self.is_healthy = False
        self.disconnect_ts = now_ms()
        self.reconnect_count += 1

    def on_reconnect(self) -> None:
        self.is_healthy = True
        gap_ms = now_ms() - (self.disconnect_ts or now_ms())
        logger.info("ws_reconnected",
                   account_id=self.account_id,
                   gap_ms=gap_ms,
                   reconnect_count=self.reconnect_count)
        self.disconnect_ts = None
```

#### Forever Orders: POST /v2/forever/orders

Forever Orders (also called GTT — Good Till Triggered) persist across trading sessions. They are not cancelled at EOD like DAY orders. This is the mechanism for overnight SL protection for strategies S2, S6 (multi-day), and S7.

```python
class DhanForeverOrderRequest(pydantic.BaseModel):
    """
    Request body for POST /v2/forever/orders.

    Forever Orders have TWO legs:
    - Leg 1: The trigger condition (price crosses trigger)
    - Leg 2: The order to place when triggered (LIMIT order)

    For SL purposes:
    - orderFlag = "SINGLE" (one-leg SL, not OCO)
    - triggerType = "STOP_LOSS" (trigger when price crosses DOWN for long, UP for short)
    """
    dhanClientId: str
    orderFlag: Literal["SINGLE", "OCO"]  # SINGLE for SL, OCO for target+SL
    transactionType: Literal["BUY", "SELL"]
    exchangeSegment: str
    productType: Literal["CNC", "MARGIN"]  # NOT INTRADAY — forever orders are multi-day
    orderType: Literal["LIMIT", "MARKET"]
    securityId: str
    quantity: int
    price: float                    # limit price for the order when triggered
    triggerPrice: float             # price at which the order activates
    correlationId: str | None

class DhanForeverOrderResponse(pydantic.BaseModel):
    orderId: str                    # forever order ID (different namespace from regular orders)
    orderStatus: str
```

**Forever Order for overnight SL (S2 example):**

```python
# S2 holds a NIFTY futures LONG overnight.
# Entry at 24500, SL at 24400 (100 points stop)

forever_sl = DhanForeverOrderRequest(
    dhanClientId="1000000001",
    orderFlag="SINGLE",
    transactionType="SELL",          # exit a LONG position
    exchangeSegment="NSE_FNO",
    productType="MARGIN",            # overnight position = MARGIN, not INTRADAY
    orderType="LIMIT",
    securityId="13",                 # NIFTY futures security ID
    quantity=75,                     # 1 lot NIFTY futures = 75
    price=24390.0,                   # limit price = trigger - 2 ticks (₹0.05 × 2)
    triggerPrice=24400.0,            # SL trigger
    correlationId="SL_S2_forever_abc123",
)

# POST https://api.dhan.co/v2/forever/orders
```

**Forever Order lifecycle:**
1. Created via API → stored on Dhan servers
2. Persists across market sessions (survives EOD, holidays, weekends)
3. When market price crosses triggerPrice → Dhan auto-places a regular LIMIT order
4. The triggered regular order then fills at the exchange
5. Modification: PUT /v2/forever/orders/{orderId}
6. Cancellation: DELETE /v2/forever/orders/{orderId}
7. Max validity: 1 year (Dhan's limit)

**Differences from DAY SL orders:**

| Aspect | DAY SL | Forever Order SL |
|--------|--------|-----------------|
| Validity | Expires at 15:30 IST session close | Persists until triggered, cancelled, or 1-year expiry |
| Product type | INTRADAY | CNC or MARGIN |
| Order type | STOP_LOSS (regular order) | Forever order (separate API) |
| WS updates | Yes, on regular order WS | Limited — trigger notifications only |
| Modification | PUT /v2/orders/{id} | PUT /v2/forever/orders/{id} |
| Use case | Intraday strategies (S1, S3, S5) | Overnight/multi-day (S2, S6, S7) |
| OPS cost | Counts against 10 OPS | Counts against 10 OPS |

---

### Mandatory Server-Side Stop-Loss

Every entry order placed by the OMS is paired with a server-side stop-loss order. This is the sole protection during system outage (process crash, EC2 reboot, Redis failure, network partition). It is not optional. There is no code path that places an entry without an SL.

#### Pairing Logic

```python
class SLPairing(pydantic.BaseModel):
    """Tracks the entry-SL pair for one position leg, one account."""
    entry_order_id: str
    sl_order_id: str
    account_id: str
    strategy_id: str
    instrument_id: str
    sl_type: Literal["DAY", "FOREVER"]
    sl_qty: int                      # MUST match entry filled_qty at all times
    sl_trigger: float
    sl_limit: float
    sl_status: Literal["PENDING", "TRIGGERED", "CANCELLED", "REJECTED", "EXPIRED", "UNKNOWN"]
    last_verified_ts: int            # epoch ms of last REST verification
    created_ts: int

async def place_entry_with_sl(
    self,
    order: "ResolvedOrder",
    account: "Account",
) -> tuple[str | None, str | None]:
    """
    Place entry order + paired SL order for one account.

    Returns (entry_order_id, sl_order_id) on success.
    Returns (None, None) if either fails.

    The SL is placed IMMEDIATELY after the entry. If SL placement fails,
    the entry is cancelled. A position without SL is NEVER acceptable.
    """
    rate_limiter = self._account_rate_limiters[account.account_id]

    # 1. Place entry order
    await rate_limiter.acquire("NEW")
    entry_resp = await self._dhan_place_order(account, DhanOrderRequest(
        dhanClientId=account.dhan_client_id,
        transactionType=order.transaction_type,
        exchangeSegment=order.exchange_segment,
        productType=self._product_type(order),
        orderType="LIMIT",
        validity="DAY",
        securityId=order.instrument_id,
        quantity=order.quantity,
        price=order.limit_price,
        triggerPrice=None,
        disclosedQuantity=0,
        afterMarketOrder=False,
        amoTime=None,
        boProfitValue=None,
        boStopLossValue=None,
        correlationId=self._make_correlation_id(order, account),
    ))

    if entry_resp.orderStatus == "REJECTED":
        logger.warning("entry_rejected",
                      account_id=account.account_id,
                      strategy_id=order.signal.strategy_id,
                      reason="broker_rejection")
        return (None, None)

    entry_order_id = entry_resp.orderId

    # 2. Register entry in demuxer for WS updates
    entry_state = OrderState(
        order_id=entry_order_id,
        account_id=account.account_id,
        original_qty=order.quantity,
        current_limit_price=order.limit_price,
    )
    self._demuxers[account.account_id].register_order(entry_order_id, entry_state)
    self._order_states[(account.account_id, entry_order_id)] = entry_state

    # 3. Place paired SL
    sl_txn = "SELL" if order.transaction_type == "BUY" else "BUY"

    if order.sl_type == "FOREVER":
        # Overnight/multi-day: use Forever Order API
        await rate_limiter.acquire("SL")
        sl_resp = await self._dhan_place_forever_order(account, DhanForeverOrderRequest(
            dhanClientId=account.dhan_client_id,
            orderFlag="SINGLE",
            transactionType=sl_txn,
            exchangeSegment=order.exchange_segment,
            productType="MARGIN",     # overnight = MARGIN
            orderType="LIMIT",
            securityId=order.instrument_id,
            quantity=order.quantity,
            price=order.sl_limit_price,
            triggerPrice=order.sl_trigger_price,
            correlationId=f"SL_{self._make_correlation_id(order, account)}",
        ))
    else:
        # Intraday: use regular SL order with DAY validity
        await rate_limiter.acquire("SL")
        sl_resp = await self._dhan_place_order(account, DhanOrderRequest(
            dhanClientId=account.dhan_client_id,
            transactionType=sl_txn,
            exchangeSegment=order.exchange_segment,
            productType="INTRADAY",
            orderType="STOP_LOSS",
            validity="DAY",
            securityId=order.instrument_id,
            quantity=order.quantity,
            price=order.sl_limit_price,
            triggerPrice=order.sl_trigger_price,
            disclosedQuantity=0,
            afterMarketOrder=False,
            amoTime=None,
            boProfitValue=None,
            boStopLossValue=None,
            correlationId=f"SL_{self._make_correlation_id(order, account)}",
        ))

    if sl_resp.orderStatus == "REJECTED":
        # SL failed — cancel entry immediately.
        # Position without SL is NEVER acceptable.
        logger.critical("sl_placement_failed_cancelling_entry",
                       account_id=account.account_id,
                       entry_id=entry_order_id,
                       strategy_id=order.signal.strategy_id)
        await rate_limiter.acquire("EXIT")
        try:
            await self._dhan_cancel_order(account, entry_order_id)
        except Exception:
            # Entry may have filled while we tried to cancel.
            # Query status to find out.
            await rate_limiter.acquire("POLL")
            status = await self._dhan_get_order_status(account, entry_order_id)
            if status.orderStatus in ("TRADED", "PART_TRADED"):
                # Entry filled but SL failed — EMERGENCY
                # Place an immediate market exit
                logger.critical("entry_filled_sl_failed_emergency_exit",
                              account_id=account.account_id,
                              entry_id=entry_order_id,
                              filled_qty=status.filledQty)
                await rate_limiter.acquire("EXIT")
                await self._dhan_place_order(account, DhanOrderRequest(
                    dhanClientId=account.dhan_client_id,
                    transactionType=sl_txn,
                    exchangeSegment=order.exchange_segment,
                    productType=self._product_type(order),
                    orderType="MARKET",
                    validity="IOC",
                    securityId=order.instrument_id,
                    quantity=status.filledQty,
                    price=0,
                    triggerPrice=None,
                    disclosedQuantity=0,
                    afterMarketOrder=False,
                    amoTime=None,
                    boProfitValue=None,
                    boStopLossValue=None,
                    correlationId=f"EMRG_{order.signal.strategy_id}",
                ))
                await telegram.send(CRITICAL,
                    f"EMERGENCY EXIT: {account.account_id} entry filled but SL rejected. "
                    f"Market exit placed for {status.filledQty} qty.")
        return (None, None)

    sl_order_id = sl_resp.orderId

    # 4. Register SL pairing
    pairing = SLPairing(
        entry_order_id=entry_order_id,
        sl_order_id=sl_order_id,
        account_id=account.account_id,
        strategy_id=order.signal.strategy_id,
        instrument_id=order.instrument_id,
        sl_type=order.sl_type,
        sl_qty=order.quantity,
        sl_trigger=order.sl_trigger_price,
        sl_limit=order.sl_limit_price,
        sl_status="PENDING",
        last_verified_ts=now_ms(),
        created_ts=now_ms(),
    )
    self.sl_manager.register(pairing)

    # 5. Register SL in demuxer if it's a regular (DAY) order
    if order.sl_type != "FOREVER":
        sl_state = OrderState(
            order_id=sl_order_id,
            account_id=account.account_id,
            original_qty=order.quantity,
            current_limit_price=order.sl_limit_price,
            is_sl=True,
        )
        self._demuxers[account.account_id].register_order(sl_order_id, sl_state)
        self._order_states[(account.account_id, sl_order_id)] = sl_state

    logger.info("entry_sl_paired",
               account_id=account.account_id,
               strategy_id=order.signal.strategy_id,
               entry_id=entry_order_id,
               sl_id=sl_order_id,
               sl_type=order.sl_type)

    return (entry_order_id, sl_order_id)
```

#### SL Validity by Strategy

| Strategy | Position Duration | SL Type | SL Validity | Rationale |
|----------|------------------|---------|-------------|-----------|
| S1 (ORB) | Intraday | Regular SL | DAY | Position closed by EOD flatten |
| S2 (Overnight) | Multi-day (overnight) | **Forever Order** | Until triggered/cancelled | **CRITICAL-1 fix**: DAY SL expires at 15:30, leaving overnight position unprotected |
| S3 (VWAP MR) | Intraday | Regular SL | DAY | Position closed by EOD flatten |
| S4 (Momentum) | Multi-day (monthly hold) | **Forever Order** | Until triggered/cancelled | Equity positions held across sessions |
| S5 (Expiry Day) | Intraday (0-DTE) | Regular SL | DAY | Expires worthless at EOD anyway |
| S6 (Vol Premium) | Variable: intraday or multi-day | **Forever Order** if multi-day, Regular SL if intraday | Depends on config | Short options may be held overnight |
| S7 (Pairs) | Multi-day | **Forever Order** | Until triggered/cancelled | Pairs positions held across sessions |

#### SL Quantity Adjustment on Every Partial Fill (CRITICAL-2 Fix)

When an entry order is partially filled, the SL order quantity MUST be immediately adjusted to match the filled quantity. If not adjusted, the SL covers more quantity than the actual position, creating a naked short/long if it triggers.

**The problem:**
```
1. Place entry: BUY 650 qty at 245.50
2. Place SL: SELL 650 qty trigger at 240.00
3. Partial fill: 325 filled, 325 remaining
4. SL still set at 650 qty
5. If SL triggers: SELL 650 — but we only hold 325
6. We're now SHORT 325 that we don't own → catastrophic
```

**The fix:** On EVERY `PART_TRADED` update, immediately modify the SL order quantity.

```python
class SLLifecycleManager:
    """
    First-class SL lifecycle management. Every SL state transition is tracked,
    verified, and acted upon.
    """

    _pairings: dict[tuple[str, str], SLPairing]
    # Key: (account_id, entry_order_id) → SLPairing

    def register(self, pairing: SLPairing) -> None:
        key = (pairing.account_id, pairing.entry_order_id)
        self._pairings[key] = pairing

    async def on_entry_partial_fill(
        self,
        account: "Account",
        entry_order_id: str,
        filled_qty: int,
    ) -> None:
        """
        IMMEDIATELY modify SL qty to match filled qty.

        Called on EVERY PART_TRADED update — not just on the timeout/abandon path.
        This is the CRITICAL-2 fix: the SL must always reflect the actual position.

        Args:
            account: The account this order belongs to
            entry_order_id: The entry order that was partially filled
            filled_qty: How many units have been filled so far (cumulative)
        """
        key = (account.account_id, entry_order_id)
        pairing = self._pairings.get(key)
        if pairing is None:
            logger.error("sl_pairing_not_found", entry_id=entry_order_id,
                        account_id=account.account_id)
            return

        if pairing.sl_qty == filled_qty:
            return  # already synced

        rate_limiter = self._oms._account_rate_limiters[account.account_id]

        if pairing.sl_type == "FOREVER":
            # Modify Forever Order
            await rate_limiter.acquire("SL")
            await self._oms._dhan_modify_forever_order(account, pairing.sl_order_id,
                quantity=filled_qty,
                price=pairing.sl_limit,
                trigger_price=pairing.sl_trigger)
        else:
            # Modify regular SL order
            # NOTE: For SL modify, quantity = NEW total (not original total like entry modify)
            # This is because the SL is being REDUCED, not repriced.
            # Dhan interprets SL modify qty as the new desired qty.
            await rate_limiter.acquire("SL")
            await self._oms._dhan_modify_order(account, DhanModifyRequest(
                dhanClientId=account.dhan_client_id,
                orderId=pairing.sl_order_id,
                orderType="STOP_LOSS",
                legName=None,
                quantity=filled_qty,    # new SL qty = filled qty
                price=pairing.sl_limit,
                triggerPrice=pairing.sl_trigger,
                disclosedQuantity=0,
                validity="DAY",
            ))

        old_qty = pairing.sl_qty
        pairing.sl_qty = filled_qty
        logger.info("sl_qty_synced_on_partial",
                   account_id=account.account_id,
                   entry_id=entry_order_id,
                   sl_id=pairing.sl_order_id,
                   old_qty=old_qty,
                   new_qty=filled_qty)

    async def on_entry_fully_filled(
        self,
        account: "Account",
        entry_order_id: str,
        filled_qty: int,
    ) -> None:
        """
        Entry fully filled. SL qty should already match (from partial fill updates).
        Verify via REST — don't trust WS alone for SL state.
        """
        key = (account.account_id, entry_order_id)
        pairing = self._pairings.get(key)
        if pairing is None:
            return

        # Verify SL is still active
        await self._verify_sl_active(account, pairing)

        # Ensure qty matches
        if pairing.sl_qty != filled_qty:
            logger.warning("sl_qty_mismatch_on_full_fill",
                         account_id=account.account_id,
                         sl_qty=pairing.sl_qty,
                         filled_qty=filled_qty)
            await self.on_entry_partial_fill(account, entry_order_id, filled_qty)

    async def on_normal_exit(
        self,
        account: "Account",
        entry_order_id: str,
    ) -> None:
        """
        Strategy exits normally (target hit, signal reversal, EOD flatten).
        Cancel the SL order — we're managing the exit ourselves.

        Race condition handled: SL may trigger between our exit decision and
        the cancel attempt. If cancel fails because SL already triggered,
        the position is already closed by the SL — skip the exit order.
        """
        key = (account.account_id, entry_order_id)
        pairing = self._pairings.get(key)
        if pairing is None:
            return

        rate_limiter = self._oms._account_rate_limiters[account.account_id]

        try:
            if pairing.sl_type == "FOREVER":
                await rate_limiter.acquire("SL")
                await self._oms._dhan_cancel_forever_order(account, pairing.sl_order_id)
            else:
                await rate_limiter.acquire("SL")
                await self._oms._dhan_cancel_order(account, pairing.sl_order_id)

            pairing.sl_status = "CANCELLED"

        except OrderNotCancellable:
            # SL may have already triggered — check
            await self._verify_sl_active(account, pairing)
            if pairing.sl_status == "TRIGGERED":
                logger.warning("sl_triggered_during_exit",
                             account_id=account.account_id,
                             entry_id=entry_order_id)
                # Position already closed by SL. Caller must skip exit order.
                return
            if pairing.sl_status == "CANCELLED":
                # Already cancelled (race with EOD flatten)
                return
            # Unknown state — log and continue
            logger.error("sl_cancel_failed_unknown_state",
                        account_id=account.account_id,
                        sl_status=pairing.sl_status)

    async def on_eod_for_overnight(
        self,
        account: "Account",
        entry_order_id: str,
    ) -> None:
        """
        EOD processing for an overnight position.

        DAY SL orders expire at 15:30 (session close). For overnight positions
        (S2, S6 multi-day, S7), we MUST place a Forever Order SL before the
        DAY SL expires. This is the CRITICAL-1 fix.

        Sequence:
        1. Place Forever Order SL (survives overnight)
        2. If Forever Order succeeds: cancel the DAY SL (or let it expire)
        3. If Forever Order fails: place AMO SL (After Market Order, active at next open)
        4. If AMO also fails: Telegram CRITICAL — position is unprotected overnight
        """
        key = (account.account_id, entry_order_id)
        pairing = self._pairings.get(key)
        if pairing is None:
            return

        if pairing.sl_type == "FOREVER":
            # Already a Forever Order — nothing to convert
            return

        rate_limiter = self._oms._account_rate_limiters[account.account_id]
        sl_txn = "SELL" if pairing.sl_trigger < self._get_current_price(pairing.instrument_id) else "BUY"

        # 1. Place Forever Order SL
        await rate_limiter.acquire("SL")
        try:
            forever_resp = await self._oms._dhan_place_forever_order(
                account,
                DhanForeverOrderRequest(
                    dhanClientId=account.dhan_client_id,
                    orderFlag="SINGLE",
                    transactionType=sl_txn,
                    exchangeSegment=self._get_exchange_segment(pairing.instrument_id),
                    productType="MARGIN",
                    orderType="LIMIT",
                    securityId=pairing.instrument_id,
                    quantity=pairing.sl_qty,
                    price=pairing.sl_limit,
                    triggerPrice=pairing.sl_trigger,
                    correlationId=f"FRVR_SL_{pairing.strategy_id}_{entry_order_id[:8]}",
                ))

            if forever_resp.orderStatus != "REJECTED":
                # Success — update pairing to reference Forever Order
                old_sl_id = pairing.sl_order_id
                pairing.sl_order_id = forever_resp.orderId
                pairing.sl_type = "FOREVER"
                pairing.last_verified_ts = now_ms()

                # Cancel old DAY SL (it will expire anyway, but cancel to be clean)
                try:
                    await rate_limiter.acquire("SL")
                    await self._oms._dhan_cancel_order(account, old_sl_id)
                except Exception:
                    pass  # old SL will expire at EOD anyway

                logger.info("sl_converted_to_forever",
                           account_id=account.account_id,
                           entry_id=entry_order_id,
                           old_sl=old_sl_id,
                           new_sl=forever_resp.orderId)
                return

        except Exception as e:
            logger.error("forever_order_placement_failed",
                        account_id=account.account_id,
                        error=str(e))

        # 2. Forever Order failed — try AMO SL
        await rate_limiter.acquire("SL")
        try:
            amo_resp = await self._oms._dhan_place_order(account, DhanOrderRequest(
                dhanClientId=account.dhan_client_id,
                transactionType=sl_txn,
                exchangeSegment=self._get_exchange_segment(pairing.instrument_id),
                productType="MARGIN",
                orderType="STOP_LOSS",
                validity="DAY",
                securityId=pairing.instrument_id,
                quantity=pairing.sl_qty,
                price=pairing.sl_limit,
                triggerPrice=pairing.sl_trigger,
                disclosedQuantity=0,
                afterMarketOrder=True,
                amoTime="OPEN",       # active immediately at next market open
                boProfitValue=None,
                boStopLossValue=None,
                correlationId=f"AMO_SL_{pairing.strategy_id}_{entry_order_id[:8]}",
            ))

            if amo_resp.orderStatus != "REJECTED":
                old_sl_id = pairing.sl_order_id
                pairing.sl_order_id = amo_resp.orderId
                pairing.sl_type = "DAY"  # AMO becomes DAY order at open
                pairing.last_verified_ts = now_ms()
                logger.info("sl_converted_to_amo",
                           account_id=account.account_id,
                           entry_id=entry_order_id)
                return

        except Exception as e:
            logger.error("amo_sl_placement_failed",
                        account_id=account.account_id,
                        error=str(e))

        # 3. Both failed — CRITICAL: position unprotected overnight
        logger.critical("overnight_sl_placement_failed_all_methods",
                       account_id=account.account_id,
                       entry_id=entry_order_id,
                       strategy_id=pairing.strategy_id,
                       instrument_id=pairing.instrument_id)
        await telegram.send(CRITICAL,
            f"OVERNIGHT POSITION UNPROTECTED: "
            f"Account {account.account_id}, "
            f"Strategy {pairing.strategy_id}, "
            f"Instrument {pairing.instrument_id}. "
            f"Forever Order AND AMO SL both failed. MANUAL INTERVENTION REQUIRED.")

    async def _verify_sl_active(self, account: "Account", pairing: SLPairing) -> None:
        """REST verification — don't trust WS alone for SL state."""
        rate_limiter = self._oms._account_rate_limiters[account.account_id]

        if pairing.sl_type == "FOREVER":
            await rate_limiter.acquire("POLL")
            resp = await self._oms._dhan_get_forever_order_status(
                account, pairing.sl_order_id)
            pairing.sl_status = resp.orderStatus
        else:
            await rate_limiter.acquire("POLL")
            resp = await self._oms._dhan_get_order_status(account, pairing.sl_order_id)
            pairing.sl_status = resp.orderStatus

        pairing.last_verified_ts = now_ms()

    async def periodic_verification(self) -> None:
        """
        Every 60 seconds, verify all active SL orders via REST.
        Catches silent rejections, status lag, and broker-side cancellations.

        Runs as a background task in the OMS process.
        """
        for key, pairing in self._pairings.items():
            account_id, entry_order_id = key
            if pairing.sl_status not in ("PENDING",):
                continue  # only check active SLs

            account = self._oms._accounts[account_id]
            await self._verify_sl_active(account, pairing)

            if pairing.sl_status not in ("PENDING", "TRIGGERED"):
                logger.critical("sl_unexpectedly_inactive",
                              account_id=account_id,
                              entry_id=entry_order_id,
                              sl_id=pairing.sl_order_id,
                              sl_status=pairing.sl_status)

                # Immediately re-place SL
                await self._re_place_sl(account, pairing)

    async def _re_place_sl(self, account: "Account", pairing: SLPairing) -> None:
        """Re-place a missing/cancelled SL. Emergency path."""
        rate_limiter = self._oms._account_rate_limiters[account.account_id]
        sl_txn = "SELL" if pairing.sl_trigger < self._get_current_price(pairing.instrument_id) else "BUY"

        if pairing.sl_type == "FOREVER":
            await rate_limiter.acquire("SL")
            resp = await self._oms._dhan_place_forever_order(account, DhanForeverOrderRequest(
                dhanClientId=account.dhan_client_id,
                orderFlag="SINGLE",
                transactionType=sl_txn,
                exchangeSegment=self._get_exchange_segment(pairing.instrument_id),
                productType="MARGIN",
                orderType="LIMIT",
                securityId=pairing.instrument_id,
                quantity=pairing.sl_qty,
                price=pairing.sl_limit,
                triggerPrice=pairing.sl_trigger,
                correlationId=f"REPL_SL_{pairing.strategy_id}",
            ))
        else:
            await rate_limiter.acquire("SL")
            resp = await self._oms._dhan_place_order(account, DhanOrderRequest(
                dhanClientId=account.dhan_client_id,
                transactionType=sl_txn,
                exchangeSegment=self._get_exchange_segment(pairing.instrument_id),
                productType="INTRADAY",
                orderType="STOP_LOSS",
                validity="DAY",
                securityId=pairing.instrument_id,
                quantity=pairing.sl_qty,
                price=pairing.sl_limit,
                triggerPrice=pairing.sl_trigger,
                disclosedQuantity=0,
                afterMarketOrder=False,
                amoTime=None,
                boProfitValue=None,
                boStopLossValue=None,
                correlationId=f"REPL_SL_{pairing.strategy_id}",
            ))

        pairing.sl_order_id = resp.orderId
        pairing.sl_status = "PENDING"
        pairing.last_verified_ts = now_ms()
        logger.info("sl_re_placed",
                   account_id=account.account_id,
                   entry_id=pairing.entry_order_id,
                   new_sl_id=resp.orderId)
```

#### Race Conditions

**Race 1: Entry fills before SL is placed**

Between entry placement (step 1) and SL placement (step 3) in `place_entry_with_sl`, there is a 100-400ms window where the entry may fill. If the entry fills instantly:

- The SL is still placed with the original quantity (correct — matches filled qty)
- The SL is now protecting the full position
- No issue unless the SL placement itself fails (handled by the cancellation/emergency path above)

At ₹50L scale, worst-case exposure during a flash crash in this 200ms window (65 lots × 200pt move × 200ms): ~₹130. Acceptable. Flag for review at ₹5Cr+.

**Race 2: SL triggers while we're placing an exit order**

When a strategy signals an exit (or EOD flatten begins):

```
1. We decide to exit the position
2. We call sl_manager.on_normal_exit() to cancel the SL
3. Between step 1 and step 2, the SL triggers
4. SL fills: position is already closed
5. Our exit order would create a new opposite position!
```

**Handling:**

```python
async def place_exit_order(
    self,
    account: "Account",
    position: "Position",
    exit_order: "ResolvedOrder",
) -> "OrderResult":
    """
    Place an exit order for an existing position.

    MUST cancel SL first. If SL has triggered, skip the exit.
    """
    entry_order_id = position.entry_order_id

    # 1. Cancel the SL
    await self.sl_manager.on_normal_exit(account, entry_order_id)

    # 2. Check if SL triggered (on_normal_exit updates pairing.sl_status)
    key = (account.account_id, entry_order_id)
    pairing = self.sl_manager._pairings.get(key)
    if pairing and pairing.sl_status == "TRIGGERED":
        logger.info("sl_triggered_before_exit_skip",
                   account_id=account.account_id,
                   entry_id=entry_order_id)
        return OrderResult(
            outcome="SKIPPED_SL_TRIGGERED",
            account_id=account.account_id,
            filled_qty=0,
        )

    # 3. Place exit order
    rate_limiter = self._account_rate_limiters[account.account_id]
    await rate_limiter.acquire("EXIT")
    exit_resp = await self._dhan_place_order(account, DhanOrderRequest(
        dhanClientId=account.dhan_client_id,
        transactionType=exit_order.transaction_type,
        exchangeSegment=exit_order.exchange_segment,
        productType=self._product_type(exit_order),
        orderType="LIMIT",
        validity="DAY",
        securityId=exit_order.instrument_id,
        quantity=position.quantity,
        price=exit_order.limit_price,
        triggerPrice=None,
        disclosedQuantity=0,
        afterMarketOrder=False,
        amoTime=None,
        boProfitValue=None,
        boStopLossValue=None,
        correlationId=f"EXIT_{exit_order.signal.strategy_id}_{account.account_id[:8]}",
    ))

    if exit_resp.orderStatus == "REJECTED":
        # Exit rejected — check if SL triggered in the meantime
        await self.sl_manager._verify_sl_active(account, pairing)
        if pairing.sl_status == "TRIGGERED":
            return OrderResult(outcome="SKIPPED_SL_TRIGGERED", ...)
        # Real rejection — Telegram CRITICAL
        logger.critical("exit_order_rejected",
                       account_id=account.account_id)
        return OrderResult(outcome="REJECTED", ...)

    # 4. Run fill management loop for the exit
    return await self.manage_order(exit_resp.orderId, exit_order, account)
```

---

### Fill Management Loop

The fill management loop is the core execution engine. One loop runs per order per account. It monitors order status, reprices when the market moves away, handles partial fills, and decides when to abandon.

#### Order State Machine

```
                  place_order()
                       │
                       ▼
                 ┌───────────┐
                 │  TRANSIT   │  (sent to exchange, awaiting ack)
                 └─────┬─────┘
                       │ exchange acks
                       ▼
                 ┌───────────┐
          ┌─────│  PENDING   │◄────────────────────┐
          │     └──┬──┬──┬───┘                      │
          │        │  │  │                          │
          │        │  │  │ partial fill             │
          │        │  │  ▼                          │
          │        │  │ ┌─────────────┐             │
          │        │  │ │ PART_TRADED │─────────────┘
          │        │  │ │             │  reprice (modify or cancel-replace)
          │        │  │ └──┬──────────┘
          │        │  │    │
          │        │  │    │ fully filled
          │        │  │    ▼
          │        │  │ ┌───────────┐
          │        │  └►│  TRADED   │  terminal: order complete
          │        │    └───────────┘
          │        │
          │        │ timeout or abandon
          │        ▼
          │  ┌───────────┐
          │  │ CANCELLED  │  terminal: order cancelled by us
          │  └───────────┘
          │
          │ rejected by exchange
          ▼
    ┌───────────┐
    │ REJECTED   │  terminal: order rejected
    └───────────┘

    ┌───────────┐
    │  EXPIRED   │  terminal: DAY order expired at session close
    └───────────┘
```

**Terminal states:** TRADED, CANCELLED, REJECTED, EXPIRED. Once terminal, no further actions.

**Non-terminal states:** TRANSIT, PENDING, PART_TRADED. The fill management loop runs while the order is in any non-terminal state.

#### OrderState

```python
class OrderState:
    """
    Mutable per-order state. One instance per order per account.

    This is a process-local object, NOT a Pydantic DTO. It is never serialized
    directly. It contains asyncio primitives (Lock, Event) that cannot be serialized.

    Thread safety: all mutations go through the per-order Lock.
    Ordering: a monotonic sequence number prevents stale-update races.
    """

    def __init__(
        self,
        order_id: str,
        account_id: str,
        original_qty: int,
        current_limit_price: float,
        is_sl: bool = False,
    ):
        self.order_id = order_id
        self.account_id = account_id
        self.lock = asyncio.Lock()
        self.sequence: int = 0
        self.update_event = asyncio.Event()

        self.status: str = "TRANSIT"
        self.original_qty = original_qty
        self.filled_qty: int = 0
        self.remaining_qty: int = original_qty
        self.avg_fill_price: float = 0.0
        self.current_limit_price = current_limit_price
        self.modification_count: int = 0
        self.placed_at_ms: int = now_ms()
        self.last_update_ms: int = now_ms()
        self.is_sl = is_sl

        # For cancel-replace tracking
        self._previous_order_ids: list[str] = []

    def apply_ws_update(self, update: DhanWSOrderUpdate) -> None:
        """
        Apply a WS update to this state. Called UNDER LOCK by the demuxer.

        Sequence number prevents stale updates: if a WS message arrives
        after a REST poll that contained newer data, the WS message is
        discarded.
        """
        self.sequence += 1
        self.status = update.orderStatus
        self.filled_qty = update.filledQty
        self.remaining_qty = update.remainingQuantity
        self.avg_fill_price = update.averageTradedPrice
        self.current_limit_price = update.price
        self.last_update_ms = now_ms()
        self.update_event.set()

    def apply_rest_update(self, detail: DhanOrderDetail) -> None:
        """Apply a REST poll update. Same as WS update but from REST source."""
        self.sequence += 1
        self.status = detail.orderStatus
        self.filled_qty = detail.filledQty
        self.remaining_qty = detail.remainingQuantity
        self.avg_fill_price = detail.averageTradedPrice
        self.current_limit_price = detail.price
        self.last_update_ms = now_ms()
        self.update_event.set()
```

#### Fill Management Loop Implementation

```python
class OrderResult(pydantic.BaseModel):
    """Result of a fill management loop."""
    outcome: Literal[
        "FILLED",             # fully filled
        "PARTIAL_FILL",       # partially filled, remainder abandoned
        "TIMEOUT",            # not filled within patience, cancelled
        "REJECTED",           # rejected by broker/exchange
        "SKIPPED_SL_TRIGGERED",  # SL triggered before exit could be placed
        "CANCELLED",          # cancelled by system (global kill, EOD)
    ]
    account_id: str
    strategy_id: str
    signal_id: str
    order_id: str
    filled_qty: int
    remaining_qty: int
    avg_fill_price: float
    modifications: int
    elapsed_ms: int

async def manage_order(
    self,
    entry_order_id: str,
    order: "ResolvedOrder",
    account: "Account",
) -> OrderResult:
    """
    Run the fill management loop for a single order on a single account.

    This is the core execution engine. It:
    1. Waits for WS updates (or polls REST if WS unhealthy)
    2. On PART_TRADED: syncs SL qty, decides whether to reprice or wait
    3. On timeout: reprices or abandons based on elapsed time vs patience
    4. On TRADED: returns success
    5. On REJECTED: returns rejection
    """
    state = self._order_states.get((account.account_id, entry_order_id))
    if state is None:
        return OrderResult(outcome="REJECTED", filled_qty=0, ...)

    params = order.fill_params
    rate_limiter = self._account_rate_limiters[account.account_id]
    ws_health = self._ws_health[account.account_id]
    start_ms = now_ms()

    while True:
        # Wait for WS update or timeout
        try:
            await asyncio.wait_for(
                state.update_event.wait(),
                timeout=params.reprice_interval_s,
            )
            state.update_event.clear()
        except asyncio.TimeoutError:
            # No WS update within reprice interval
            if not ws_health.is_healthy:
                # WS down — poll REST
                await rate_limiter.acquire("POLL")
                detail = await self._dhan_get_order_status(account, state.order_id)
                async with state.lock:
                    state.apply_rest_update(detail)

        elapsed_ms = now_ms() - start_ms
        elapsed_s = elapsed_ms / 1000

        # Per-order lock: atomic read + decide + act
        async with state.lock:
            match state.status:
                case "TRADED":
                    # Fully filled
                    await self.sl_manager.on_entry_fully_filled(
                        account, entry_order_id, state.filled_qty)
                    return OrderResult(
                        outcome="FILLED",
                        account_id=account.account_id,
                        strategy_id=order.signal.strategy_id,
                        signal_id=order.signal.signal_id,
                        order_id=state.order_id,
                        filled_qty=state.filled_qty,
                        remaining_qty=0,
                        avg_fill_price=state.avg_fill_price,
                        modifications=state.modification_count,
                        elapsed_ms=elapsed_ms,
                    )

                case "PART_TRADED":
                    # Partial fill — IMMEDIATELY sync SL qty
                    await self.sl_manager.on_entry_partial_fill(
                        account, entry_order_id, state.filled_qty)

                    # Notify position tracker
                    await self._notify_partial_fill(account, state, order)

                    if elapsed_s >= params.max_patience_s:
                        # Patience exhausted — cancel remaining
                        await rate_limiter.acquire("MODIFY")
                        try:
                            await self._dhan_cancel_order(account, state.order_id)
                        except OrderAlreadyFilled:
                            # Race: filled between check and cancel
                            await rate_limiter.acquire("POLL")
                            detail = await self._dhan_get_order_status(
                                account, state.order_id)
                            state.apply_rest_update(detail)
                            continue  # re-evaluate in next loop

                        # SL already synced to filled_qty
                        return OrderResult(
                            outcome="PARTIAL_FILL",
                            account_id=account.account_id,
                            strategy_id=order.signal.strategy_id,
                            signal_id=order.signal.signal_id,
                            order_id=state.order_id,
                            filled_qty=state.filled_qty,
                            remaining_qty=state.remaining_qty,
                            avg_fill_price=state.avg_fill_price,
                            modifications=state.modification_count,
                            elapsed_ms=elapsed_ms,
                        )

                    elif self._price_has_moved(state, order, account):
                        # Price moved — reprice
                        await self.reprice_order(state, order, params, account)

                case "PENDING":
                    if elapsed_s >= params.max_patience_s:
                        # Patience exhausted, not filled at all — cancel
                        await rate_limiter.acquire("MODIFY")
                        try:
                            await self._dhan_cancel_order(account, state.order_id)
                        except OrderAlreadyFilled:
                            await rate_limiter.acquire("POLL")
                            detail = await self._dhan_get_order_status(
                                account, state.order_id)
                            state.apply_rest_update(detail)
                            continue

                        # Cancel SL too (no position to protect)
                        await self.sl_manager.on_normal_exit(account, entry_order_id)
                        return OrderResult(
                            outcome="TIMEOUT",
                            account_id=account.account_id,
                            strategy_id=order.signal.strategy_id,
                            signal_id=order.signal.signal_id,
                            order_id=state.order_id,
                            filled_qty=0,
                            remaining_qty=state.original_qty,
                            avg_fill_price=0.0,
                            modifications=state.modification_count,
                            elapsed_ms=elapsed_ms,
                        )

                    elif elapsed_s >= params.reprice_interval_s and \
                         self._price_has_moved(state, order, account):
                        await self.reprice_order(state, order, params, account)

                case "REJECTED":
                    await self.sl_manager.on_normal_exit(account, entry_order_id)
                    return OrderResult(
                        outcome="REJECTED",
                        account_id=account.account_id,
                        strategy_id=order.signal.strategy_id,
                        signal_id=order.signal.signal_id,
                        order_id=state.order_id,
                        filled_qty=0,
                        remaining_qty=state.original_qty,
                        avg_fill_price=0.0,
                        modifications=state.modification_count,
                        elapsed_ms=elapsed_ms,
                    )

                case "CANCELLED" | "EXPIRED":
                    # Cancelled externally or expired
                    if state.filled_qty > 0:
                        # Had partial fills — SL already synced
                        return OrderResult(outcome="PARTIAL_FILL", ...)
                    else:
                        await self.sl_manager.on_normal_exit(account, entry_order_id)
                        return OrderResult(outcome="CANCELLED", ...)

                case "TRANSIT":
                    # Still in transit to exchange — wait
                    if elapsed_s > 10:
                        # 10s in transit is unusual — poll REST
                        await rate_limiter.acquire("POLL")
                        detail = await self._dhan_get_order_status(
                            account, state.order_id)
                        state.apply_rest_update(detail)
```

#### Reprice Logic

```python
def compute_new_limit_price(
    self,
    order: "ResolvedOrder",
    params: FillParams,
    account: "Account",
) -> float:
    """
    Compute a new limit price based on current market data and pricing mode.

    Market data source:
    - Primary: Redis LASTTICK:{symbol} (updated per tick by Data Ingester)
    - Fallback (Redis down): Dhan option chain API direct call

    Pricing modes:
    - PASSIVE:     BUY at best_ask,            SELL at best_bid
    - MIDPOINT:    BUY at ceil(mid/tick)*tick,  SELL at floor(mid/tick)*tick
    - AGGRESSIVE:  BUY at best_ask + 1 tick,   SELL at best_bid - 1 tick
    """
    # Get current market data
    market = self._get_current_market_data(order.instrument_id, account)
    tick_size = order.tick_size

    if order.transaction_type == "BUY":
        match params.pricing_mode:
            case "PASSIVE":
                raw = market.best_ask
            case "MIDPOINT":
                mid = (market.best_bid + market.best_ask) / 2
                raw = math.ceil(mid / tick_size) * tick_size
            case "AGGRESSIVE":
                raw = market.best_ask + tick_size
    else:  # SELL
        match params.pricing_mode:
            case "PASSIVE":
                raw = market.best_bid
            case "MIDPOINT":
                mid = (market.best_bid + market.best_ask) / 2
                raw = math.floor(mid / tick_size) * tick_size
            case "AGGRESSIVE":
                raw = market.best_bid - tick_size

    # Round to tick size
    return round(raw / tick_size) * tick_size

def _price_has_moved(
    self,
    state: OrderState,
    order: "ResolvedOrder",
    account: "Account",
) -> bool:
    """Check if market price has moved away from our limit price enough to reprice."""
    market = self._get_current_market_data(order.instrument_id, account)
    tick_size = order.tick_size

    if order.transaction_type == "BUY":
        # If best ask has moved up by >1 tick from our limit, reprice
        return market.best_ask > state.current_limit_price + tick_size
    else:
        # If best bid has moved down by >1 tick from our limit, reprice
        return market.best_bid < state.current_limit_price - tick_size

def _get_current_market_data(
    self,
    instrument_id: str,
    account: "Account",
) -> "MarketSnapshot":
    """
    Get current bid/ask for an instrument.

    Primary: Redis LASTTICK (updated per tick, <50ms latency)
    Fallback: Dhan option chain API (costs 1 OPS, 200-400ms latency)
    """
    try:
        tick_data = self._redis_sync.get(f"LASTTICK:{instrument_id}")
        if tick_data:
            tick = Tick.model_validate_json(tick_data)
            return MarketSnapshot(
                best_bid=tick.bid,
                best_ask=tick.ask,
                ltp=tick.ltp,
                ts=tick.receive_ts,
            )
    except Exception:
        pass

    # Redis down or key missing — fallback to Dhan API
    # This is the OMS degraded mode market data path
    logger.debug("market_data_fallback_to_api",
               instrument_id=instrument_id,
               account_id=account.account_id)
    return self._fetch_market_from_dhan(instrument_id, account)


class MarketSnapshot(pydantic.BaseModel):
    best_bid: float
    best_ask: float
    ltp: float
    ts: int
```

#### Fill Parameters Per Strategy

| Strategy | `reprice_interval_s` | `max_patience_s` | `pricing_mode` | Rationale |
|----------|---------------------|-------------------|----------------|-----------|
| S1 (ORB) | 5 | 15 | AGGRESSIVE | Time-sensitive breakout — need fast fills |
| S2 (Overnight) | 10 | 30 | MIDPOINT | Patient entry at 15:15, not rushing |
| S3 (VWAP MR) | 5 | 20 | AGGRESSIVE | Mean reversion — entry level matters but speed > precision |
| S4 (Momentum) | 30 | 120 | PASSIVE | Monthly rebalance, large order, minimize impact |
| S5 (Expiry Day) | 3 | 10 | AGGRESSIVE | 0-DTE, every second counts, fast fills critical |
| S6 (Vol Premium) | 15 | 60 | PASSIVE | Selling premium, patient entry |
| S7 (Pairs) | 10 | 45 | MIDPOINT | Pairs — moderate urgency, spread matters |

**Pricing modes explained:**

| Mode | BUY price | SELL price | When to use |
|------|-----------|------------|-------------|
| PASSIVE | best_ask | best_bid | Large orders, no urgency, minimize market impact |
| MIDPOINT | ceil(mid/tick) × tick | floor(mid/tick) × tick | Moderate urgency, balanced fill quality |
| AGGRESSIVE | best_ask + 1 tick | best_bid - 1 tick | Time-sensitive signals, fast fills |

---

### Multi-Account Order Fan-Out

When multi-account mode is active (`len(accounts) > 1`), every signal produces N independent order lifecycles — one per enabled account. Signal generation is shared; everything from sizing onward is per-account.

#### Fan-Out Flow

```
StrategySignal arrives at Signal Router
       │
       ▼
Signal Dedup (shared — one check)
       │
       ▼
For EACH enabled account in config.accounts:
       │
       ├── Does this account run this strategy?
       │     (account.enabled_strategies contains signal.strategy_id)
       │     NO → skip
       │     YES ↓
       │
       ├── Capital Allocator (per-account)
       │     → Uses THIS account's capital, weights, Kelly fraction
       │     → Computes lot count for THIS account
       │     → Checks existing position for THIS account
       │     → Returns AllocationResponse per-account
       │
       ├── Instrument Resolver (shared)
       │     → Same security_id for all accounts (same contract)
       │     → Limit price computed once (shared market data)
       │     → Returns ResolvedOrder (same for all accounts)
       │
       ├── Risk Manager (per-account)
       │     → Checks THIS account's margin
       │     → Checks THIS account's position limits
       │     → Checks THIS account's drawdown
       │     → ALSO checks aggregate exposure across all accounts
       │
       ├── OMS.place_entry_with_sl (per-account)
       │     → Uses THIS account's API key
       │     → Uses THIS account's rate limiter (10 OPS)
       │     → Places entry + SL on THIS account
       │     → Returns (entry_id, sl_id) for THIS account
       │
       └── Fill Management Loop (per-account)
             → Runs independently for THIS account
             → Fills may differ (timing, price, partial fills)
             → SL qty sync per-account
             → OrderResult per-account
```

```python
class MultiAccountFanOut:
    """
    Orchestrates order fan-out across N accounts.

    One signal → N parallel order placements + fill management loops.
    Each account operates independently after the fan-out point.
    """

    def __init__(
        self,
        accounts: list["Account"],
        oms: "OMS",
        capital_allocator: "CapitalAllocator",
        risk_manager: "RiskManager",
        instrument_resolver: "InstrumentResolver",
    ):
        self._accounts = {a.account_id: a for a in accounts}
        self._oms = oms
        self._allocator = capital_allocator
        self._risk = risk_manager
        self._resolver = instrument_resolver

    async def fan_out_signal(self, signal: "StrategySignal") -> list["AccountOrderResult"]:
        """
        Fan out a single signal to all enabled accounts.

        Returns a list of per-account results. Accounts that reject the signal
        (insufficient margin, position limits, not enabled for this strategy)
        are included with rejection reasons.
        """
        results: list[AccountOrderResult] = []

        # Resolve instrument ONCE (shared across accounts — same security_id)
        resolved = await self._resolver.resolve(signal)
        if resolved is None:
            logger.warning("instrument_resolution_failed",
                         strategy_id=signal.strategy_id)
            return results

        # Create per-account tasks for parallel execution
        tasks = []
        for account_id, account in self._accounts.items():
            if not account.enabled:
                continue
            if signal.strategy_id not in account.enabled_strategies:
                continue

            task = asyncio.create_task(
                self._process_for_account(signal, resolved, account),
                name=f"fanout_{account_id}_{signal.signal_id[:8]}",
            )
            tasks.append((account_id, task))

        # Execute all accounts in parallel
        # Each account has its own rate limiter — no OPS contention
        for account_id, task in tasks:
            try:
                result = await task
                results.append(result)
            except Exception:
                logger.exception("fanout_account_error",
                               account_id=account_id,
                               signal_id=signal.signal_id)
                results.append(AccountOrderResult(
                    account_id=account_id,
                    outcome="ERROR",
                    error="unexpected_exception",
                ))

        # Log divergence
        outcomes = {r.account_id: r.outcome for r in results}
        if len(set(outcomes.values())) > 1:
            logger.warning("account_divergence_detected",
                         signal_id=signal.signal_id,
                         outcomes=outcomes)

        return results

    async def _process_for_account(
        self,
        signal: "StrategySignal",
        resolved: "ResolvedOrder",
        account: "Account",
    ) -> "AccountOrderResult":
        """Process a signal for one specific account."""

        # 1. Allocate capital for THIS account
        allocation = await self._allocator.request_allocation(
            AllocationRequest(
                account_id=account.account_id,
                strategy_id=signal.strategy_id,
                direction=signal.direction,
                underlying=signal.underlying,
                instrument_type=resolved.instrument_hint,
                estimated_premium=resolved.limit_price,
                lot_size=resolved.lot_size,
                existing_position_qty=self._get_existing_position(
                    account.account_id, signal.strategy_id),
            ),
            account=account,
        )

        if not allocation.approved:
            return AccountOrderResult(
                account_id=account.account_id,
                outcome="ALLOCATION_REJECTED",
                error=allocation.rejection_reason,
            )

        # 2. Adjust resolved order for this account's sizing
        account_order = resolved.copy(deep=True)
        account_order.quantity = allocation.max_lots * resolved.lot_size
        # Limit price and SL levels remain the same (same market, same instrument)

        # 3. Risk check for THIS account
        risk_ok = await self._risk.pre_trade_check(account_order, account)
        if not risk_ok.passed:
            return AccountOrderResult(
                account_id=account.account_id,
                outcome="RISK_REJECTED",
                error=risk_ok.rejection_reason,
            )

        # 4. Place entry + SL for THIS account
        entry_id, sl_id = await self._oms.place_entry_with_sl(account_order, account)
        if entry_id is None:
            return AccountOrderResult(
                account_id=account.account_id,
                outcome="PLACEMENT_FAILED",
                error="entry_or_sl_rejected",
            )

        # 5. Run fill management loop for THIS account
        fill_result = await self._oms.manage_order(entry_id, account_order, account)

        return AccountOrderResult(
            account_id=account.account_id,
            outcome=fill_result.outcome,
            order_id=fill_result.order_id,
            filled_qty=fill_result.filled_qty,
            avg_fill_price=fill_result.avg_fill_price,
            elapsed_ms=fill_result.elapsed_ms,
        )


class AccountOrderResult(pydantic.BaseModel):
    account_id: str
    outcome: str
    order_id: str | None = None
    filled_qty: int = 0
    remaining_qty: int = 0
    avg_fill_price: float = 0.0
    elapsed_ms: int = 0
    error: str | None = None
```

#### Per-Account Rate Limiters

Each Dhan API key gets its own 10 OPS rate limit. With N accounts, the system has N × 10 OPS total. Each account's rate limiter is independent.

```python
class OMS:
    def __init__(self, accounts: list["Account"]):
        # One rate limiter per account — they don't share OPS budget
        self._account_rate_limiters: dict[str, PriorityRateLimiter] = {
            account.account_id: PriorityRateLimiter(rate=10)
            for account in accounts
        }
```

#### Per-Account WS Connections

Each account has its own order update WebSocket connection and demuxer.

```python
class OMS:
    def __init__(self, accounts: list["Account"]):
        # One WS connection per account
        self._ws_connections: dict[str, DhanOrderWS] = {
            account.account_id: DhanOrderWS(account)
            for account in accounts
        }

        # One demuxer per account
        self._demuxers: dict[str, OrderUpdateDemuxer] = {
            account.account_id: OrderUpdateDemuxer(account)
            for account in accounts
        }

        # One WS health monitor per account
        self._ws_health: dict[str, WSHealthMonitor] = {
            account.account_id: WSHealthMonitor(account.account_id)
            for account in accounts
        }
```

#### Account Divergence Handling

Accounts can diverge when one account fills and another rejects.

**Causes of divergence:**
1. Account B has insufficient margin → allocation rejected
2. Account B's API key expired mid-session → all orders fail
3. Account B's order rejected by exchange (position limit, circuit limit)
4. Account A fills at 245.50, Account C fills at 245.55 (normal price variation)
5. Account A fully fills 10 lots, Account B only fills 6 lots (partial fill, different liquidity)

**Divergence tracking:**

```python
class DivergenceTracker:
    """
    Tracks position divergence across accounts for the same strategy.

    Divergence is EXPECTED (lot rounding, partial fills, margin differences).
    This tracker quantifies it for monitoring and compliance reporting.
    """

    async def record_signal_outcome(
        self,
        signal_id: str,
        strategy_id: str,
        results: list[AccountOrderResult],
    ) -> None:
        divergence = DivergenceRecord(
            signal_id=signal_id,
            strategy_id=strategy_id,
            timestamp_ms=now_ms(),
            account_outcomes={
                r.account_id: {
                    "outcome": r.outcome,
                    "filled_qty": r.filled_qty,
                    "avg_fill_price": r.avg_fill_price,
                }
                for r in results
            },
        )

        # Check for actionable divergence
        filled_accounts = [r for r in results if r.outcome == "FILLED"]
        failed_accounts = [r for r in results
                          if r.outcome in ("REJECTED", "PLACEMENT_FAILED", "ERROR")]

        if filled_accounts and failed_accounts:
            logger.warning("divergence_fill_vs_reject",
                         signal_id=signal_id,
                         filled=[r.account_id for r in filled_accounts],
                         failed=[r.account_id for r in failed_accounts])
            await telegram.send(WARNING,
                f"Account divergence on signal {signal_id[:8]}: "
                f"{len(filled_accounts)} filled, {len(failed_accounts)} failed")

        # Price divergence across filled accounts
        if len(filled_accounts) >= 2:
            prices = [r.avg_fill_price for r in filled_accounts]
            max_price_diff = max(prices) - min(prices)
            if max_price_diff > 5.0:  # >₹5 divergence
                logger.warning("divergence_price",
                             signal_id=signal_id,
                             max_diff=max_price_diff)

        # Store for compliance reporting
        await self._store_divergence(divergence)

    async def get_divergence_report(
        self,
        strategy_id: str | None = None,
        lookback_days: int = 30,
    ) -> list["DivergenceRecord"]:
        """For PMS/AIF compliance: per-client divergence report."""
        ...
```

**Policy on divergence:**
- Divergence is monitored but NOT force-synced. Each account runs independently.
- If Account B consistently fails (>3 consecutive signals rejected), disable it and alert operator.
- Per-client NAV computation (SEBI PMS requirement) uses actual per-account fills, not theoretical fills.
- At no point does the system trade one account to "catch up" to another. Divergence is an operational reality.

#### OPS Budget Analysis: Multi-Account

With N accounts, each with 10 OPS, the system has N × 10 OPS total. This RELAXES the OPS constraint.

**Single account (current):**

```
Total OPS: 10

Normal operation (3 strategies with pending orders, WS healthy):
  Entry modifications:  0.73 OPS/s
  No REST polling:       0.00 OPS/s
  New entry burst:       2 OPS (entry + SL)
  Total typical:         ~1-3 OPS/s  →  well within 10

EOD flatten (5 positions):
  Phase 1 (cancel pending): 2×5 = 10 cancels = 1.0s
  Phase 2 (exit orders):    5 orders = 0.5s
  Phase 3 (fill management): shared 10 OPS
  Total: ~3-5 seconds
```

**3 accounts:**

```
Total OPS: 30 (10 per account, independent)

Normal operation:
  Per-account: same ~1-3 OPS/s
  Cross-account: NO contention — each account has its own bucket
  Signal fan-out: 3 entries + 3 SLs = 6 OPS total
    But: 2 OPS per account, all 3 accounts in parallel = 2 OPS time cost
  Total wall-clock time: SAME as single account

EOD flatten (5 positions × 3 accounts = 15 positions):
  Per-account (parallel):
    5 cancels = 0.5s per account
    5 exits = 0.5s per account
    Fill management on 10 OPS per account
  Wall-clock time: SAME as single account (3 accounts run in parallel)
  Total API calls: 3× but on 3× the OPS budget = no congestion
```

**OPS scaling table:**

| Accounts | Total OPS | Per-Signal OPS (entry+SL) | Wall-Clock Time | Concurrent Fill Loops |
|----------|-----------|--------------------------|-----------------|----------------------|
| 1 | 10 | 2 | 200ms | Up to 10 per second |
| 2 | 20 | 4 total, 2 per account | 200ms (parallel) | Up to 20 per second |
| 3 | 30 | 6 total, 2 per account | 200ms (parallel) | Up to 30 per second |
| 5 | 50 | 10 total, 2 per account | 200ms (parallel) | Up to 50 per second |
| 10 | 100 | 20 total, 2 per account | 200ms (parallel) | Up to 100 per second |

**The key insight:** Multi-account does NOT multiply latency. All accounts execute in parallel. The OPS budget scales linearly while wall-clock time stays constant.

---

### Priority Rate Limiter

One `PriorityRateLimiter` instance per account. Ensures high-priority operations (exits, SL) are never starved by low-priority operations (polling, new entries).

```python
import asyncio
import heapq
import time

class PriorityRateLimiter:
    """
    Token bucket with priority-based dispatch.

    When multiple operations compete for a limited OPS budget, higher-priority
    operations dequeue first. This prevents a burst of POLL requests from
    delaying an EXIT order.

    Token refill: 1 token per (1/rate) seconds. Tokens accumulate up to
    max_tokens (= rate). Burst capacity = rate tokens.
    """

    PRIORITIES: dict[str, int] = {
        "EXIT":   0,    # highest — closing positions, never delayed
        "SL":     1,    # SL placement/modification — safety-critical
        "MODIFY": 2,    # fill management repricing
        "NEW":    3,    # new entry orders
        "POLL":   4,    # REST status checks — lowest priority
    }

    def __init__(self, rate: int = 10):
        """
        Args:
            rate: Maximum operations per second (Dhan limit: 10 per API key).
        """
        self.rate = rate
        self.max_tokens = rate
        self._tokens: float = rate
        self._last_refill: float = time.monotonic()
        self._queue: list[tuple[int, int, asyncio.Event]] = []  # min-heap
        self._counter: int = 0  # tiebreaker for same-priority items
        self._lock = asyncio.Lock()
        self._dispatch_task: asyncio.Task | None = None

    async def acquire(self, priority: str, timeout: float = 10.0) -> None:
        """
        Acquire a rate limit token at the given priority level.

        Blocks until a token is available. Higher-priority requests dequeue first.
        If no token is available within timeout seconds, raises TimeoutError.

        Args:
            priority: One of EXIT, SL, MODIFY, NEW, POLL
            timeout: Max wait time in seconds (default 10s)

        Raises:
            asyncio.TimeoutError: if token not acquired within timeout
            ValueError: if priority is unknown
        """
        if priority not in self.PRIORITIES:
            raise ValueError(f"Unknown priority '{priority}'. "
                           f"Valid: {list(self.PRIORITIES.keys())}")

        event = asyncio.Event()
        self._counter += 1

        async with self._lock:
            heapq.heappush(
                self._queue,
                (self.PRIORITIES[priority], self._counter, event)
            )

        # Try to dispatch immediately
        await self._try_dispatch()

        # Wait for our turn
        await asyncio.wait_for(event.wait(), timeout=timeout)

    async def _try_dispatch(self) -> None:
        """Try to dispatch queued requests if tokens are available."""
        async with self._lock:
            self._refill_tokens()

            while self._queue and self._tokens >= 1.0:
                _, _, event = heapq.heappop(self._queue)
                self._tokens -= 1.0
                event.set()

            # If there are still items in the queue, schedule a delayed dispatch
            if self._queue and self._dispatch_task is None:
                self._dispatch_task = asyncio.create_task(self._delayed_dispatch())

    async def _delayed_dispatch(self) -> None:
        """Wait for token refill, then dispatch."""
        await asyncio.sleep(1.0 / self.rate)
        self._dispatch_task = None
        await self._try_dispatch()

    def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.max_tokens,
            self._tokens + elapsed * self.rate
        )
        self._last_refill = now

    @property
    def available_tokens(self) -> float:
        """Current available tokens (for monitoring)."""
        return self._tokens

    @property
    def queue_depth(self) -> int:
        """Number of queued requests (for monitoring)."""
        return len(self._queue)
```

**Concrete OPS budget scenario with 3 accounts:**

```
Scenario: EOD flatten with 5 positions per account at 15:20 IST.

Account "prop" (10 OPS):
  15:20:00  Cancel 5 pending entries:        5 OPS  (EXIT priority)
  15:20:00  Cancel 5 paired SLs:             5 OPS  (SL priority)
            → 10 OPS consumed in 1.0 second
  15:20:01  Place 5 exit orders:             5 OPS  (EXIT priority)
            → 0.5 seconds
  15:20:02  Fill management (5 parallel):    shared 10 OPS for repricing
            → ~2 OPS/s for modifications
  15:22:00  IOC escalation (if needed):      5 OPS  (EXIT priority)
            → 0.5 seconds

Account "client_001" (10 OPS — independent, runs in parallel):
  15:20:00  Same sequence, same timing, different API key
            → All 10 OPS consumed independently

Account "client_002" (10 OPS — independent, runs in parallel):
  15:20:00  Same sequence, same timing, different API key

Wall-clock time: ~2-3 seconds for all 3 accounts (parallel)
If single account: same 2-3 seconds
Multi-account adds ZERO latency to the flatten sequence.
```

---

### EOD Flatten

The EOD flatten sequence closes all intraday positions before market close. It runs independently and in parallel across all accounts.

```python
class EODFlatten:
    """
    End-of-day position flattening.

    Triggered at 15:20 IST for intraday strategies.
    Overnight positions (S2, S6 multi-day, S7) are NOT flattened — their SLs
    are converted to Forever Orders instead.

    The sequence is identical for each account but runs in parallel.
    """

    PHASE_1_TIME = "15:20:00"  # cancel pending
    PHASE_2_TIME = "15:20:01"  # place exits
    PHASE_3_TIME = "15:20:02"  # fill management
    PHASE_4_TIME = "15:22:00"  # IOC escalation
    PHASE_5_TIME = "15:25:00"  # hard deadline

    async def run(self, accounts: list["Account"]) -> dict[str, "FlattenResult"]:
        """Run EOD flatten for all accounts in parallel."""

        # Launch flatten for each account concurrently
        tasks = {
            account.account_id: asyncio.create_task(
                self._flatten_account(account),
                name=f"flatten_{account.account_id}",
            )
            for account in accounts
        }

        results = {}
        for account_id, task in tasks.items():
            try:
                results[account_id] = await task
            except Exception:
                logger.exception("flatten_account_error", account_id=account_id)
                results[account_id] = FlattenResult(
                    account_id=account_id, success=False, error="exception")

        return results

    async def _flatten_account(self, account: "Account") -> "FlattenResult":
        """
        5-phase flatten for one account.

        Phase 1: Cancel all pending entry orders + their SL pairs
        Phase 2: Place all exit orders in burst (AGGRESSIVE pricing)
        Phase 3: Parallel fill management for all exits
        Phase 4: IOC escalation for unfilled exits
        Phase 5: Hard deadline — alert, broker auto-squares at 15:30
        """
        rate_limiter = self._oms._account_rate_limiters[account.account_id]
        position_tracker = self._oms._position_tracker

        # ---- Phase 1: Cancel all pending entries ----
        logger.info("flatten_phase_1_cancel_pending", account_id=account.account_id)

        pending_orders = self._oms.get_pending_orders(account.account_id)
        cancel_tasks = []
        for order_state in pending_orders:
            if order_state.is_sl:
                continue  # SLs cancelled in Phase 2 as part of exit

            async def cancel_one(os=order_state):
                await rate_limiter.acquire("EXIT")
                try:
                    await self._oms._dhan_cancel_order(account, os.order_id)
                except OrderAlreadyFilled:
                    pass  # filled between check and cancel — fine
                # Cancel the paired SL too
                pairing = self._oms.sl_manager.get_pairing(
                    account.account_id, os.order_id)
                if pairing:
                    await self._oms.sl_manager.on_normal_exit(
                        account, os.order_id)

            cancel_tasks.append(asyncio.create_task(cancel_one()))

        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)

        # ---- Phase 2: Place all exit orders ----
        logger.info("flatten_phase_2_place_exits", account_id=account.account_id)

        intraday_positions = position_tracker.get_intraday_positions(
            account.account_id)
        # Filter out overnight positions (S2, S7, and S6 if multi-day)
        intraday_positions = [
            p for p in intraday_positions
            if p.strategy_id not in self._overnight_strategies(account)
        ]

        exit_tasks = []
        for pos in intraday_positions:
            exit_order = self._build_exit_order(pos, urgency="URGENT")

            # Cancel the entry's SL — we're managing the exit now
            await self._oms.sl_manager.on_normal_exit(
                account, pos.entry_order_id)

            # Check if SL triggered
            pairing = self._oms.sl_manager.get_pairing(
                account.account_id, pos.entry_order_id)
            if pairing and pairing.sl_status == "TRIGGERED":
                logger.info("flatten_skip_sl_triggered",
                           account_id=account.account_id,
                           instrument=pos.trading_symbol)
                continue  # position already closed by SL

            await rate_limiter.acquire("EXIT")
            resp = await self._oms._dhan_place_order(account, exit_order)

            if resp.orderStatus != "REJECTED":
                exit_state = OrderState(
                    order_id=resp.orderId,
                    account_id=account.account_id,
                    original_qty=pos.quantity,
                    current_limit_price=exit_order.price,
                )
                self._oms._demuxers[account.account_id].register_order(
                    resp.orderId, exit_state)
                exit_tasks.append((pos, exit_state, exit_order))

        # ---- Phase 3: Parallel fill management ----
        logger.info("flatten_phase_3_fill_management",
                   account_id=account.account_id,
                   exit_count=len(exit_tasks))

        fill_loops = []
        for pos, state, exit_order in exit_tasks:
            fill_params = FillParams(
                reprice_interval_s=3,     # fast repricing for EOD
                max_patience_s=110,       # until Phase 4 at 15:22
                pricing_mode="AGGRESSIVE",
            )
            loop = asyncio.create_task(
                self._oms.manage_order(state.order_id, exit_order, account))
            fill_loops.append((pos, state, loop))

        # Wait until Phase 4 time or all loops complete
        phase_4_deadline = self._time_until("15:22:00")
        done, pending = await asyncio.wait(
            [loop for _, _, loop in fill_loops],
            timeout=phase_4_deadline,
        )

        # ---- Phase 4: IOC escalation ----
        unfilled = [(pos, state) for pos, state, loop in fill_loops
                    if loop in pending]

        if unfilled:
            logger.warning("flatten_phase_4_ioc_escalation",
                         account_id=account.account_id,
                         unfilled_count=len(unfilled))

            for pos, state in unfilled:
                # Cancel the pending exit
                await rate_limiter.acquire("EXIT")
                try:
                    await self._oms._dhan_cancel_order(account, state.order_id)
                except Exception:
                    pass

                # Re-place as IOC at aggressive price (+2 ticks)
                market = self._oms._get_current_market_data(
                    pos.instrument_id, account)
                tick_size = pos.tick_size

                if pos.direction == "LONG":
                    # Sell to close: bid - 2 ticks
                    ioc_price = market.best_bid - 2 * tick_size
                else:
                    # Buy to close: ask + 2 ticks
                    ioc_price = market.best_ask + 2 * tick_size

                # +2 ticks, not +5: SEBI limit-order rule means our limit must
                # still be a reasonable limit, not a de facto market order
                ioc_price = round(ioc_price / tick_size) * tick_size

                remaining_qty = state.remaining_qty if state.filled_qty > 0 else pos.quantity
                await rate_limiter.acquire("EXIT")
                await self._oms._dhan_place_order(account, DhanOrderRequest(
                    dhanClientId=account.dhan_client_id,
                    transactionType="SELL" if pos.direction == "LONG" else "BUY",
                    exchangeSegment=pos.exchange_segment,
                    productType="INTRADAY",
                    orderType="LIMIT",
                    validity="IOC",          # Immediate or Cancel
                    securityId=pos.instrument_id,
                    quantity=remaining_qty,
                    price=ioc_price,
                    triggerPrice=None,
                    disclosedQuantity=0,
                    afterMarketOrder=False,
                    amoTime=None,
                    boProfitValue=None,
                    boStopLossValue=None,
                    correlationId=f"IOC_FLAT_{account.account_id[:8]}",
                ))

        # ---- Phase 5: Hard deadline (15:25) ----
        phase_5_deadline = self._time_until("15:25:00")
        await asyncio.sleep(max(0, phase_5_deadline))

        # Check for any remaining positions
        remaining_positions = position_tracker.get_intraday_positions(
            account.account_id)
        remaining_intraday = [
            p for p in remaining_positions
            if p.strategy_id not in self._overnight_strategies(account)
        ]

        if remaining_intraday:
            logger.critical("flatten_incomplete_at_hard_deadline",
                          account_id=account.account_id,
                          remaining=[p.trading_symbol for p in remaining_intraday])
            await telegram.send(CRITICAL,
                f"EOD FLATTEN INCOMPLETE: Account {account.account_id} has "
                f"{len(remaining_intraday)} positions at 15:25. "
                f"Dhan auto-squares at ~15:30 at market price.")

        # ---- Convert overnight SLs to Forever Orders ----
        overnight_positions = position_tracker.get_overnight_positions(
            account.account_id)
        for pos in overnight_positions:
            await self._oms.sl_manager.on_eod_for_overnight(
                account, pos.entry_order_id)

        return FlattenResult(
            account_id=account.account_id,
            success=len(remaining_intraday) == 0,
            positions_closed=len(intraday_positions) - len(remaining_intraday),
            positions_remaining=len(remaining_intraday),
            overnight_sl_converted=len(overnight_positions),
        )


class FlattenResult(pydantic.BaseModel):
    account_id: str
    success: bool
    positions_closed: int = 0
    positions_remaining: int = 0
    overnight_sl_converted: int = 0
    error: str | None = None
```

**OPS budget for N-account flatten:**

```
Per-account (5 positions, 10 OPS):
  Phase 1: 5 cancel entry + 5 cancel SL = 10 OPS → 1.0 second
  Phase 2: 5 exit orders = 5 OPS → 0.5 second
  Phase 3: fill management, ~2-3 OPS/s for repricing → 2 minutes
  Phase 4: 5 cancel + 5 IOC = 10 OPS → 1.0 second (if needed)
  Total per account: ~3-5 seconds of OPS-intensive work

With 3 accounts (parallel):
  Wall-clock = max(per-account) = same 3-5 seconds
  Total API calls = 3x but on 3x the OPS budget = no congestion

With 5 accounts (parallel):
  Wall-clock = same 3-5 seconds
  Total API calls = 5x but on 5x the OPS budget = no congestion
```

**Dhan auto-square backstop:** Dhan auto-squares all INTRADAY positions at approximately 15:30 IST at market price. This is the last line of defense if our EOD flatten fails completely. The auto-square is undesirable (market price = potentially worse than our limit) but prevents carrying unintended overnight risk.

---

### OMS Degraded Mode (Redis Down)

When Redis is unavailable, the OMS enters degraded mode. It continues managing existing orders and positions using broker WS + REST directly. No new signals arrive (strategy processes can't publish without Redis).

#### What Still Works

| Function | Status | Mechanism |
|----------|--------|-----------|
| Dhan order update WS | WORKS | Direct WebSocket connection, no Redis dependency |
| Dhan REST API (place, modify, cancel, status) | WORKS | Direct HTTPS, no Redis dependency |
| Fill management loops (running orders) | WORKS | OrderState is in-memory, WS updates route to it |
| SL lifecycle management | WORKS | SLPairing objects are in-memory |
| Per-order locking | WORKS | asyncio.Lock is in-memory |
| Rate limiters | WORKS | Token bucket state is in-memory |
| EOD flatten | WORKS | Uses broker API directly |
| Global kill switch | PARTIAL | OMS checks an in-memory flag; the Redis `HALT:global` check fails but the CLI can set the in-memory flag directly |
| Server-side SL orders | WORKS | SL orders are on broker infra, independent of our system |

#### What Breaks

| Function | Status | Impact |
|----------|--------|--------|
| Strategy signal delivery | BROKEN | Strategies can't publish to `STREAM:SIGNAL` |
| New entry orders | BLOCKED | No signals arriving → no new orders |
| LASTTICK for repricing | BROKEN | Can't read `LASTTICK:{symbol}` from Redis |
| Position state updates | BROKEN | Can't write `POSITION:strategy:{sid}` |
| Heartbeats | BROKEN | Strategies appear dead (heartbeat keys expire) |
| State snapshots | BROKEN | Can't save strategy state to Redis |
| Config hot-reload | BROKEN | Can't read `CONFIG:strategy:{sid}` |

#### Fallback for Market Prices

When Redis `LASTTICK` is unavailable, the OMS falls back to Dhan's option chain API for market data:

```python
def _fetch_market_from_dhan(
    self,
    instrument_id: str,
    account: "Account",
) -> MarketSnapshot:
    """
    Fallback market data source when Redis is down.

    Uses Dhan's quote API to get current bid/ask for an instrument.
    Costs 1 OPS per call. Used ONLY for repricing active fill management loops.

    Endpoint: GET https://api.dhan.co/v2/marketfeed/ltp
    or: POST https://api.dhan.co/v2/marketfeed/quote

    This is expensive (1 OPS per reprice check) but acceptable in degraded mode
    because new entries are blocked (no signals) so OPS headroom is available.
    """
    rate_limiter = self._account_rate_limiters[account.account_id]
    # Don't acquire rate limit here — caller already acquired
    resp = self._dhan_client.get_quote(
        account.dhan_access_token,
        security_id=instrument_id,
    )
    return MarketSnapshot(
        best_bid=resp.get("bestBidPrice", 0),
        best_ask=resp.get("bestOfferPrice", 0),
        ltp=resp.get("lastTradedPrice", 0),
        ts=now_ms(),
    )
```

#### Recovery When Redis Reconnects

```python
async def on_redis_reconnect(self) -> None:
    """
    Called when Redis connection is re-established.

    Sequence:
    1. Publish accumulated state changes (order results, position updates)
    2. Re-enable signal reception (unblock STREAM:SIGNAL consumer)
    3. Strategy processes will auto-reconnect their consumer groups
    4. Normal operation resumes
    """
    logger.info("redis_reconnected_oms_recovery")

    # 1. Sync all current order states to Redis
    for (account_id, order_id), state in self._order_states.items():
        await redis.hset(f"ORDER:{account_id}:{order_id}", mapping={
            "status": state.status,
            "filled_qty": str(state.filled_qty),
            "remaining_qty": str(state.remaining_qty),
            "avg_fill_price": str(state.avg_fill_price),
        })

    # 2. Sync all SL pairings
    for key, pairing in self.sl_manager._pairings.items():
        account_id, entry_id = key
        await redis.hset(f"SL:{account_id}:{entry_id}", mapping={
            "sl_order_id": pairing.sl_order_id,
            "sl_qty": str(pairing.sl_qty),
            "sl_status": pairing.sl_status,
            "sl_type": pairing.sl_type,
        })

    # 3. Clear HALT flag if it was set due to Redis outage
    # (but NOT if global kill was manually triggered)
    if not self._manual_kill_active:
        await redis.delete("HALT:redis_outage")

    # 4. Resume normal mode
    self._degraded_mode = False
    logger.info("oms_degraded_mode_exited")
```

---

### Global Kill Switch

```python
async def global_kill(self) -> None:
    """
    One command: cancel everything, flatten everything, NOW.
    Runs across ALL accounts in parallel.

    Triggered by:
    - CLI: python -m live.cli kill
    - Redis: SET HALT:global 1
    - Telegram: /kill command (authenticated)
    - Risk Manager: portfolio drawdown > 10%
    """
    logger.critical("GLOBAL_KILL_ACTIVATED")
    self._manual_kill_active = True

    # 1. Block all new signals immediately
    try:
        await redis.set("HALT:global", "1")
    except Exception:
        pass  # Redis may be down — in-memory flag is the backup

    # 2. Cancel ALL pending orders across ALL accounts
    cancel_tasks = []
    for account_id, account in self._accounts.items():
        rate_limiter = self._account_rate_limiters[account_id]
        for key, state in self._order_states.items():
            if key[0] != account_id:
                continue
            if state.status in ("PENDING", "PART_TRADED", "TRANSIT"):
                async def cancel_one(a=account, s=state, rl=rate_limiter):
                    try:
                        await rl.acquire("EXIT")
                        await self._dhan_cancel_order(a, s.order_id)
                    except Exception:
                        pass  # best effort
                cancel_tasks.append(asyncio.create_task(cancel_one()))

    if cancel_tasks:
        await asyncio.gather(*cancel_tasks, return_exceptions=True)

    # 3. Flatten ALL positions across ALL accounts with AGGRESSIVE pricing
    flatten_tasks = []
    for account_id, account in self._accounts.items():
        positions = self._position_tracker.get_all_open(account_id)
        rate_limiter = self._account_rate_limiters[account_id]

        for pos in positions:
            async def flatten_one(a=account, p=pos, rl=rate_limiter):
                exit_order = self._build_exit_order(p, urgency="URGENT")
                await rl.acquire("EXIT")
                await self._dhan_place_order(a, exit_order)
            flatten_tasks.append(asyncio.create_task(flatten_one()))

    if flatten_tasks:
        await asyncio.gather(*flatten_tasks, return_exceptions=True)

    # 4. Alert
    await telegram.send(CRITICAL,
        f"GLOBAL KILL: all orders cancelled across {len(self._accounts)} accounts, "
        f"flattening all positions")

    logger.critical("GLOBAL_KILL_COMPLETE")
```

**Resuming after global kill:** requires manual intervention:
1. `DEL HALT:global` in Redis
2. Full system restart
3. Operator verifies all positions are flat across all accounts
4. Operator reviews kill reason in audit log

---

### State Table

| State Item | Storage | Scope | Lifecycle | Writer | Readers |
|-----------|---------|-------|-----------|--------|---------|
| OrderState objects | In-memory | Per-account, per-order | Created on placement, removed on terminal state | OMS (demuxer applies updates) | Fill management loop |
| SLPairing objects | In-memory | Per-account, per-entry | Entry → SL mapping, lifetime of position | SL Lifecycle Manager | OMS, EOD Flatten |
| Rate limiter tokens | In-memory | Per-account | Session | PriorityRateLimiter | All OMS operations |
| Rate limiter queue | In-memory | Per-account | Transient (drain on each dispatch) | PriorityRateLimiter | All OMS operations |
| WS connection | In-memory (socket) | Per-account | Session (reconnects on drop) | DhanOrderWS | OrderUpdateDemuxer |
| WS health state | In-memory | Per-account | Continuous | WSHealthMonitor | Fill management loops |
| Demuxer routing table | In-memory | Per-account | Order lifetime | OMS (register/unregister) | OrderUpdateDemuxer |
| Placed signal_ids (dedup) | In-memory set | Global (across accounts) | Session | OMS | OMS (before placement) |
| Daily order count | Redis `OMS:daily_count:{account_id}` | Per-account | Reset 08:30 | OMS | Risk Manager, Monitoring |
| Order state mirror | Redis `ORDER:{account_id}:{order_id}` | Per-account, per-order | Updated on every state change | OMS | Position Tracker, Monitoring |
| SL pairing mirror | Redis `SL:{account_id}:{entry_id}` | Per-account | Updated on SL changes | SL Lifecycle Manager | Monitoring |
| Degraded mode flag | In-memory | Global | Set on Redis loss, cleared on reconnect | OMS | OMS (all operations) |
| Manual kill flag | In-memory + Redis `HALT:global` | Global | Set by kill, cleared by operator | OMS, CLI, Telegram | All components |
| Account auth tokens | In-memory | Per-account | Session (refreshed daily) | Auth Manager | DhanOrderWS, REST calls |
| Divergence records | DuckDB | Per-signal | Persistent | DivergenceTracker | Compliance reporting |

---

### Failure Modes

| # | Failure | Detection | Impact | Recovery |
|---|---------|-----------|--------|----------|
| 1 | **Dhan order WS disconnects (one account)** | `ConnectionClosed` exception in demuxer | That account's fill management loops lose real-time updates. Other accounts unaffected. | Exponential backoff reconnect: 1s, 2s, 4s, 8s, max 30s. Fill management loops switch to REST polling (costs OPS). |
| 2 | **Dhan order WS disconnects (all accounts)** | All WSHealthMonitor.is_healthy = False | All fill management loops switch to REST polling. OPS consumption increases significantly. | Same reconnect logic. If all WS down for >60s: Telegram CRITICAL, suppress new entries. |
| 3 | **Dhan REST API returns 5xx** | HTTP status code | That specific API call fails. | Retry with backoff: 500ms, 1s, 2s. Max 3 retries. If persistent: mark account as degraded. |
| 4 | **Dhan REST API returns 429 (rate limited)** | HTTP 429 | Server-side rate limit hit (we exceeded 10 OPS). | Back off for 1 second. Should not happen if our client-side rate limiter is working correctly. If it does: log ERROR, investigate. |
| 5 | **Account API key expires mid-session** | HTTP 401 from any Dhan call | All operations for that account fail. Other accounts continue. | Mark account as AUTH_FAILED. Stop all new orders for that account. Existing positions protected by server-side SLs. Telegram CRITICAL: "Account {id} auth expired — manual re-auth required." |
| 6 | **Entry order rejected by exchange** | `orderStatus: "REJECTED"` | No position taken. SL is cancelled (if already placed) or never placed. | Log rejection reason. Signal is consumed (not retried — market may have moved). |
| 7 | **SL order rejected by exchange** | `orderStatus: "REJECTED"` for SL | Entry order may be pending or filled WITHOUT SL protection. | CRITICAL: Cancel entry immediately. If entry already filled: place emergency market exit. Telegram CRITICAL. |
| 8 | **SL triggers while modifying entry** | SL WS update shows TRADED while entry modify is in-flight | Position is closed by SL. Any pending modify is irrelevant. | Detect in fill management loop: if SL has triggered, stop managing the entry. Position already flat. |
| 9 | **Partial fill → SL qty mismatch** | SL qty != entry filled_qty after PART_TRADED | If SL triggers with wrong qty: over-exit (short shares we don't own) or under-exit (residual position). | CRITICAL-2 fix: `on_entry_partial_fill` called on EVERY PART_TRADED update, not just on abandon. |
| 10 | **DAY SL expires at EOD for overnight position** | 15:30 IST session close | Overnight position has zero SL protection until next morning. | CRITICAL-1 fix: `on_eod_for_overnight` converts to Forever Order before DAY SL expires. Falls back to AMO if Forever fails. |
| 11 | **Order modify fails silently** | Post-modify REST verification shows price didn't change | Order is at stale price. May not fill or fill at wrong level. | Every modify is followed by a REST verification poll. If price mismatch: re-send modify. |
| 12 | **Cancel-replace creates duplicate position** | After cancel, old order fills before we detect the fill. New order also placed. | Double position: 2x intended exposure. | Check if old order filled before placing new order. If old order is TRADED: skip new placement. Use the order's correlation_id to detect cancel-replace pairs in reconciliation. |
| 13 | **Redis down during fill management** | `aioredis.ConnectionError` | Can't update LASTTICK for repricing. Can't publish results. | OMS enters degraded mode: use Dhan API for market data, hold state in-memory, sync to Redis on reconnect. |
| 14 | **OMS process crashes** | systemd detects exit | All order tracking state lost. Fill management loops stop. | Restart: query Dhan for all active orders + positions. Rebuild OrderState from Dhan API. SL orders are on broker — still active. |
| 15 | **Dhan daily order limit (5,000) reached** | HTTP rejection with order-limit error | No more orders for that account for the day. | Telegram CRITICAL. Account enters read-only mode. Existing positions manage via SL only. Should not happen in normal operation (7 strategies × 2 trades/day × 2 orders/trade = 28 orders/day). |
| 16 | **Account A fills, Account B rejects (divergence)** | MultiAccountFanOut compares outcomes | Accounts hold different positions for the same strategy signal. | Log divergence. Do NOT force-sync (that would require additional trades). Track for compliance reporting. If Account B fails >3 consecutive signals: disable it, Telegram CRITICAL. |
| 17 | **Freeze qty exceeded (large order)** | `quantity > freeze_qty` check before placement | Exchange rejects orders above freeze limit. | Auto-switch to slicing endpoint (POST /v2/orders/slicing). Track child orders independently. |
| 18 | **Network partition between OMS and Dhan** | Connection timeouts on all REST + WS | Complete broker blackout. Cannot place, modify, or cancel. | Server-side SLs are the sole protection. Telegram CRITICAL (if Telegram is reachable). Wait for network restoration. If >5 min: operator intervention required. |
| 19 | **Dhan WS sends duplicate update** | Sequence number in OrderState is already >= update's implied sequence | State could be applied twice, corrupting filled_qty. | Sequence number check: if WS update would decrease sequence, discard it. |
| 20 | **Global kill during active fill management** | `HALT:global` flag checked on every loop iteration | Fill loops must abort immediately. | Set in-memory `_halt` flag. All fill loops check this flag and exit cleanly on next iteration. Cancel remaining orders. |

---

### Edge Cases

#### 1. Entry fills in 2ms, before SL API call starts

The entry LIMIT order is placed and fills within 2ms (happens with aggressive pricing in liquid instruments). By the time we place the SL:
- Position is already filled at known price
- SL placement uses the filled price (not the limit price) for trigger calculation
- SL is placed with exact filled_qty
- No issue — the position is protected once SL is confirmed

#### 2. Modify returns success but price doesn't change

Dhan may accept a modify request but the exchange hasn't processed it yet (TRANSIT state). Our post-modify REST poll may show the OLD price.

**Handling:** If post-modify poll shows the old price, wait 500ms and poll again. If still old price after 3 polls: assume modify failed, consider cancel-replace.

#### 3. SL triggers during EOD flatten's cancel-SL step

```
1. EOD flatten calls sl_manager.on_normal_exit() to cancel SL
2. While our cancel request is in-flight, the market moves and SL triggers
3. Dhan fills the SL (position closed)
4. Our cancel request returns "order not cancellable"
5. We place an exit order (but position is already flat!)
6. Exit order creates a NEW position (wrong direction)
```

**Handling:** After cancel-SL fails with "not cancellable", verify SL status. If TRIGGERED: skip exit order. The `place_exit_order` method checks this (see Race Condition 2 above).

#### 4. Sliced order: some children fill, others don't

A 1950-qty order (above 1800 freeze limit) is sliced into two children: 1800 + 150. The 1800-qty child fills, the 150-qty child is rejected (insufficient lot).

**Handling:** Track each child independently. SL is placed for total intended qty. On partial (child rejection): adjust SL qty to match total filled across all children. Treat as PARTIAL_FILL in the fill management result.

#### 5. Account added mid-session (hot-add)

v1 does not support hot-adding accounts mid-session. All accounts must be configured at startup. The system restart required to add an account is the same restart required to add a strategy (see Strategy Manager hot-add decision).

#### 6. Two signals arrive for the same strategy within 100ms

Signal deduplication (in Signal Router) handles this upstream. By the time signals reach the OMS, they are already deduplicated. Additionally, the OMS tracks `placed_signal_ids` and rejects any `signal_id` it has already processed.

#### 7. Dhan changes modify qty semantics

If Dhan changes the modify API to interpret `quantity` as remaining (not original total), our orders would be cancelled unexpectedly.

**Mitigation:** Paper trading validation: place a 650-qty order, get 325 filled, modify with qty=650, verify remaining is still 325. If behavior changes: flip a config flag `MODIFY_QTY_IS_REMAINING = True` and adjust the reprice logic to send `remaining_qty` instead of `original_qty`.

```python
# In reprice_order:
if config.MODIFY_QTY_IS_REMAINING:
    modify_qty = state.remaining_qty
else:
    modify_qty = state.original_qty  # default: Dhan current behavior
```

#### 8. Forever Order triggers overnight, WS not connected

Our system is shut down overnight (08:00 startup). A Forever Order SL triggers at 03:00 AM (unlikely for equity derivatives, but possible for currency futures or if the trigger is very close to the closing price and a gap opens).

**Handling:** The Forever Order triggers on Dhan's infrastructure without our WS connected. The fill happens on the exchange. At our 08:40 startup (Phase 4: Position Recovery), we query broker positions and discover the position is flat. We reconcile: mark the SL as TRIGGERED, mark the position as closed, log the event. No action needed — the SL did its job.

#### 9. Multi-account: different lot sizes due to rounding

Account A (₹50L, 0.25 Kelly) allocates 3.7 lots → rounds to 3 lots.
Account B (₹1Cr, 0.25 Kelly) allocates 7.4 lots → rounds to 7 lots.
Account C (₹2Cr, 0.25 Kelly) allocates 14.8 lots → rounds to 15 lots.

Accounts are NOT proportional (3:7:15 vs the theoretical 1:2:4 capital ratio). This is inherent to discrete lot sizes.

**Rounding rule:** Always round DOWN (floor). Never round up — that would exceed the allocation.

```python
def compute_lots(capital: float, kelly: float, weight: float,
                premium: float, lot_size: int) -> int:
    raw_capital = capital * kelly * weight
    raw_qty = raw_capital / premium
    raw_lots = raw_qty / lot_size
    return max(1, int(raw_lots))  # floor, minimum 1 lot
```

#### 10. OMS places order, Dhan returns success, but order never appears in order book

Rare: Dhan REST returns 200 with orderId, but the order is silently dropped before reaching the exchange.

**Detection:** Order stays in TRANSIT for >10 seconds. Fill management loop's TRANSIT timeout triggers a REST poll. If REST also shows TRANSIT after 10s: cancel and re-place. If REST shows the order doesn't exist: treat as rejected.

---


## Component 6: Multi-Broker Router

### Responsibility

- Maintain per-account broker client instances with independent auth, connections, and failover state
- Provide a unified `BrokerAdapter` interface that insulates the OMS from broker-specific API details
- Map canonical instrument identifiers (Dhan `security_id`) to equivalent identifiers on backup brokers (Upstox `instrument_key`)
- Authenticate all accounts in parallel at system startup (08:25 IST) and manage token lifecycle in Redis
- Detect per-account broker health degradation and execute independent failover: hedge open positions on backup broker while monitoring primary for recovery
- Enforce per-account rate limits via delegation to the OMS's `PriorityRateLimiter` (10 OPS per Dhan account, 50 OPS per Upstox account)
- **Does NOT** generate signals, manage positions, or track fills (those are upstream/downstream components)

---

### Per-Account Broker Clients

Each trading account gets its own broker client instance. Accounts do not share connections, tokens, or rate-limit budgets. If Account A's broker connection fails, Account B continues operating on its own independent connection.

#### Account Model

```python
class Account(pydantic.BaseModel):
    """
    Canonical Account model (defined in Component 11: Account Replication Layer).
    Reproduced here for reference — the single source of truth is in section 10.

    The Broker Router uses these fields:
    - account_id, dhan_client_id, dhan_access_token: for DhanClient creation
    - capital: Decimal (exact arithmetic at ₹10Cr scale)
    - primary_broker: which broker to use by default
    """
    account_id: str
    dhan_client_id: str
    dhan_access_token: str
    capital: Decimal
    enabled_strategies: list[str]
    strategy_weights: dict[str, float]
    kelly_fraction: float
    max_drawdown_pct: float
    status: Literal["ACTIVE", "SUSPENDED", "AUTH_FAILED", "MARGIN_CALL"] = "ACTIVE"
    # Broker Router extensions (not in canonical model):
    upstox_client_id: str | None = None
    upstox_api_key: str | None = None
    primary_broker: Literal["dhan", "upstox"] = "dhan"
```

#### DhanClient

One `DhanClient` per account. Handles authentication headers, base URL, and retry logic for all REST calls.

```python
import aiohttp
import asyncio
import time
import structlog

logger = structlog.get_logger()


class DhanClient:
    """
    HTTP client for the Dhan REST API. One instance per account.

    Handles:
    - Auth headers (per-account access token)
    - Base URL routing
    - Retry with exponential backoff on transient errors (5xx, timeout)
    - Request/response logging for audit trail
    - Connection pooling via aiohttp.ClientSession
    """

    BASE_URL = "https://api.dhan.co"
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 0.5   # seconds: 0.5, 1.0, 2.0
    REQUEST_TIMEOUT = 10.0     # seconds per request

    def __init__(self, account: Account):
        self.account = account
        self._client_id = account.dhan_client_id
        self._token: str | None = None
        self._session: aiohttp.ClientSession | None = None
        self._request_count: int = 0
        self._error_count: int = 0
        self._last_request_ts: float = 0.0

    @property
    def auth_headers(self) -> dict[str, str]:
        if self._token is None:
            raise RuntimeError(
                f"DhanClient for {self.account.account_id} not authenticated"
            )
        return {
            "access-token": self._token,
            "Content-Type": "application/json",
        }

    def set_token(self, token: str) -> None:
        """Set the access token after authentication."""
        self._token = token

    async def start(self) -> None:
        """Create the aiohttp session. Call once at startup."""
        timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
        self._session = aiohttp.ClientSession(
            base_url=self.BASE_URL,
            headers=self.auth_headers,
            timeout=timeout,
        )

    async def close(self) -> None:
        """Close the aiohttp session. Call at shutdown."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
    ) -> dict:
        """
        Make an authenticated request to Dhan API with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API path (e.g., "/v2/orders")
            body: JSON body for POST/PUT requests

        Returns:
            Parsed JSON response as dict.

        Raises:
            DhanAPIError: on non-retryable failure (4xx except 429)
            DhanRateLimitError: on HTTP 429
            DhanAuthError: on HTTP 401
            DhanTimeoutError: after MAX_RETRIES exhausted
        """
        if self._session is None:
            raise RuntimeError("DhanClient session not started")

        last_exception: Exception | None = None

        for attempt in range(self.MAX_RETRIES):
            try:
                self._request_count += 1
                self._last_request_ts = time.monotonic()

                async with self._session.request(
                    method,
                    path,
                    json=body,
                    headers=self.auth_headers,
                ) as resp:
                    response_body = await resp.json()

                    if resp.status == 200:
                        return response_body

                    if resp.status == 401:
                        self._error_count += 1
                        raise DhanAuthError(
                            account_id=self.account.account_id,
                            message=response_body.get("remarks", "auth_failed"),
                        )

                    if resp.status == 429:
                        self._error_count += 1
                        raise DhanRateLimitError(
                            account_id=self.account.account_id,
                        )

                    if 400 <= resp.status < 500:
                        self._error_count += 1
                        raise DhanAPIError(
                            account_id=self.account.account_id,
                            status=resp.status,
                            error_code=response_body.get("errorCode"),
                            message=response_body.get("remarks", "unknown"),
                        )

                    # 5xx — retryable
                    last_exception = DhanAPIError(
                        account_id=self.account.account_id,
                        status=resp.status,
                        error_code=response_body.get("errorCode"),
                        message=response_body.get("remarks", "server_error"),
                    )
                    logger.warning(
                        "dhan_api_5xx_retry",
                        account_id=self.account.account_id,
                        status=resp.status,
                        attempt=attempt + 1,
                        path=path,
                    )

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exception = DhanTimeoutError(
                    account_id=self.account.account_id,
                    message=str(e),
                )
                logger.warning(
                    "dhan_api_timeout_retry",
                    account_id=self.account.account_id,
                    attempt=attempt + 1,
                    path=path,
                    error=str(e),
                )

            # Exponential backoff before retry
            if attempt < self.MAX_RETRIES - 1:
                backoff = self.RETRY_BACKOFF_BASE * (2 ** attempt)
                await asyncio.sleep(backoff)

        # All retries exhausted
        self._error_count += 1
        raise last_exception or DhanTimeoutError(
            account_id=self.account.account_id,
            message=f"All {self.MAX_RETRIES} retries exhausted for {method} {path}",
        )

    # --- Convenience methods ---

    async def place_order(self, body: dict) -> dict:
        return await self.request("POST", "/v2/orders", body)

    async def modify_order(self, order_id: str, body: dict) -> dict:
        return await self.request("PUT", f"/v2/orders/{order_id}", body)

    async def cancel_order(self, order_id: str) -> dict:
        return await self.request("DELETE", f"/v2/orders/{order_id}")

    async def get_order(self, order_id: str) -> dict:
        return await self.request("GET", f"/v2/orders/{order_id}")

    async def get_trades(self, order_id: str) -> dict:
        return await self.request("GET", f"/v2/trades/{order_id}")

    async def place_forever_order(self, body: dict) -> dict:
        return await self.request("POST", "/v2/forever/orders", body)

    async def modify_forever_order(self, order_id: str, body: dict) -> dict:
        return await self.request("PUT", f"/v2/forever/orders/{order_id}", body)

    async def cancel_forever_order(self, order_id: str) -> dict:
        return await self.request("DELETE", f"/v2/forever/orders/{order_id}")

    async def get_positions(self) -> dict:
        return await self.request("GET", "/v2/positions")

    async def get_holdings(self) -> dict:
        return await self.request("GET", "/v2/holdings")

    async def place_slicing_order(self, body: dict) -> dict:
        return await self.request("POST", "/v2/orders/slicing", body)

    @property
    def stats(self) -> dict:
        return {
            "account_id": self.account.account_id,
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._request_count
                if self._request_count > 0
                else 0.0
            ),
            "last_request_ts": self._last_request_ts,
        }


class DhanAPIError(Exception):
    def __init__(self, account_id: str, status: int,
                 error_code: str | None, message: str):
        self.account_id = account_id
        self.status = status
        self.error_code = error_code
        self.message = message
        super().__init__(f"[{account_id}] Dhan {status}: {error_code} — {message}")


class DhanAuthError(DhanAPIError):
    def __init__(self, account_id: str, message: str):
        super().__init__(account_id, 401, "AUTH_FAILED", message)


class DhanRateLimitError(DhanAPIError):
    def __init__(self, account_id: str):
        super().__init__(account_id, 429, "RATE_LIMITED", "Too many requests")


class DhanTimeoutError(Exception):
    def __init__(self, account_id: str, message: str):
        self.account_id = account_id
        self.message = message
        super().__init__(f"[{account_id}] Dhan timeout: {message}")
```

#### Per-Account Failover State

Each account tracks its own broker health independently. Account A might be operating on Dhan while Account B has failed over to Upstox.

```python
from enum import Enum


class BrokerHealth(str, Enum):
    HEALTHY = "HEALTHY"             # primary broker operating normally
    DEGRADED = "DEGRADED"           # intermittent failures, still usable
    FAILED = "FAILED"               # primary broker unreachable
    FAILOVER_ACTIVE = "FAILOVER"    # using backup broker
    SUSPENDED = "SUSPENDED"         # account disabled (auth failure, operator action)


class AccountBrokerState(pydantic.BaseModel):
    """
    Per-account broker health and failover state. One instance per account.
    Stored in-memory, mirrored to Redis for monitoring.
    """
    account_id: str
    primary_broker: Literal["dhan", "upstox"] = "dhan"
    active_broker: Literal["dhan", "upstox"] = "dhan"
    health: BrokerHealth = BrokerHealth.HEALTHY

    # Failure tracking for failover trigger
    consecutive_failures: int = 0
    last_failure_ts: float = 0.0
    last_success_ts: float = 0.0

    # WS state
    ws_connected: bool = False
    ws_disconnect_ts: float | None = None

    # Failover tracking
    failover_ts: float | None = None
    failover_reason: str | None = None
    failback_eligible_ts: float | None = None   # earliest allowed failback

    # Hedging state (populated during failover)
    hedge_positions: list["HedgePosition"] = []

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.last_success_ts = time.monotonic()
        if self.health == BrokerHealth.DEGRADED:
            self.health = BrokerHealth.HEALTHY

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_ts = time.monotonic()
        if self.consecutive_failures >= 3:
            self.health = BrokerHealth.FAILED

    def record_ws_disconnect(self) -> None:
        self.ws_connected = False
        self.ws_disconnect_ts = time.monotonic()

    def record_ws_reconnect(self) -> None:
        self.ws_connected = True
        self.ws_disconnect_ts = None
```

#### Per-Account WS Connections

Each account maintains its own WebSocket connection for order updates. This is established in the OMS (Component 5) via `DhanOrderWS`. The broker router owns the lifecycle:

```python
class BrokerRouter:
    """
    Central broker routing layer. Manages per-account broker clients,
    failover, symbol mapping, and auth lifecycle.
    """

    def __init__(self, accounts: list[Account]):
        self._accounts = {a.account_id: a for a in accounts}

        # Per-account Dhan clients
        self._dhan_clients: dict[str, DhanClient] = {
            a.account_id: DhanClient(a) for a in accounts
        }

        # Per-account Upstox clients (created on demand during failover)
        self._upstox_clients: dict[str, "UpstoxClient"] = {}

        # Per-account broker state
        self._broker_state: dict[str, AccountBrokerState] = {
            a.account_id: AccountBrokerState(
                account_id=a.account_id,
                primary_broker=a.primary_broker,
                active_broker=a.primary_broker,
            )
            for a in accounts
        }

        # Per-account WS connections (for order updates)
        self._order_ws: dict[str, DhanOrderWS] = {
            a.account_id: DhanOrderWS(a) for a in accounts
        }

        # Symbol mapper (shared across all accounts)
        self._symbol_mapper: SymbolMapper | None = None

        # Auth manager
        self._auth_manager = AuthManager(self)

    def get_client(self, account_id: str) -> "BrokerAdapter":
        """
        Return the active broker adapter for this account.
        If failover is active, returns the backup broker's adapter.
        """
        state = self._broker_state[account_id]

        if state.active_broker == "dhan":
            return DhanAdapter(self._dhan_clients[account_id])
        elif state.active_broker == "upstox":
            if account_id not in self._upstox_clients:
                raise BrokerUnavailableError(
                    f"Upstox client not initialized for {account_id}"
                )
            return UpstoxAdapter(self._upstox_clients[account_id])

        raise BrokerUnavailableError(
            f"No active broker for {account_id}, state={state.health}"
        )

    def get_state(self, account_id: str) -> AccountBrokerState:
        return self._broker_state[account_id]
```

---

### BrokerAdapter Interface

The OMS interacts with brokers exclusively through the `BrokerAdapter` abstract base class. Broker-specific details (endpoint URLs, request schemas, field names) are encapsulated in concrete implementations.

```python
from abc import ABC, abstractmethod


class OrderResponse(pydantic.BaseModel):
    """Normalized order response from any broker."""
    order_id: str
    status: Literal["TRANSIT", "PENDING", "REJECTED"]
    broker: Literal["dhan", "upstox"]
    raw_response: dict                  # original broker response for audit


class OrderDetail(pydantic.BaseModel):
    """Normalized order status from any broker."""
    order_id: str
    status: Literal[
        "TRANSIT", "PENDING", "TRADED", "PART_TRADED",
        "CANCELLED", "REJECTED", "EXPIRED",
    ]
    filled_qty: int
    remaining_qty: int
    avg_fill_price: float
    exchange_order_id: str | None
    exchange_time: str | None
    broker: Literal["dhan", "upstox"]


class PositionDetail(pydantic.BaseModel):
    """Normalized position from any broker."""
    instrument_id: str                  # canonical (Dhan security_id)
    exchange_segment: str
    product_type: str
    quantity: int                       # signed: positive = long, negative = short
    avg_price: float
    pnl: float
    broker: Literal["dhan", "upstox"]


class BrokerAdapter(ABC):
    """
    Abstract interface for broker integration. The OMS calls these methods
    without knowing which broker is active for a given account.

    Every method accepts an Account and the canonical instrument identifiers.
    The adapter translates to broker-specific field names internally.
    """

    @abstractmethod
    async def place_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        validity: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        """Place an order. Returns normalized response."""
        ...

    @abstractmethod
    async def modify_order(
        self,
        account: Account,
        order_id: str,
        order_type: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        validity: str = "DAY",
    ) -> OrderResponse:
        """Modify a pending order."""
        ...

    @abstractmethod
    async def cancel_order(
        self, account: Account, order_id: str,
    ) -> OrderResponse:
        """Cancel a pending order."""
        ...

    @abstractmethod
    async def get_order_status(
        self, account: Account, order_id: str,
    ) -> OrderDetail:
        """Poll order status via REST."""
        ...

    @abstractmethod
    async def get_positions(self, account: Account) -> list[PositionDetail]:
        """Get all positions for this account."""
        ...

    @abstractmethod
    async def place_forever_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        """Place a GTT / Forever order."""
        ...

    @abstractmethod
    async def place_slicing_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        validity: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        """Place an order that auto-slices above freeze quantity."""
        ...

    @abstractmethod
    async def get_trades(self, account: Account, order_id: str) -> list[dict]:
        """Get individual trade executions for an order."""
        ...

    @property
    @abstractmethod
    def broker_name(self) -> str:
        """Return broker identifier."""
        ...
```

#### DhanAdapter Implementation

```python
class DhanAdapter(BrokerAdapter):
    """
    Dhan broker adapter. Translates BrokerAdapter calls into
    Dhan REST API requests via the per-account DhanClient.
    """

    def __init__(self, client: DhanClient):
        self._client = client

    @property
    def broker_name(self) -> str:
        return "dhan"

    async def place_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        validity: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        body = {
            "dhanClientId": account.dhan_client_id,
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,
            "productType": product_type,
            "orderType": order_type,
            "validity": validity,
            "securityId": security_id,
            "quantity": quantity,
            "price": price,
            "triggerPrice": trigger_price,
            "disclosedQuantity": 0,
            "afterMarketOrder": False,
            "amoTime": None,
            "boProfitValue": None,
            "boStopLossValue": None,
            "correlationId": correlation_id,
        }

        raw = await self._client.place_order(body)

        return OrderResponse(
            order_id=raw["orderId"],
            status=raw["orderStatus"],
            broker="dhan",
            raw_response=raw,
        )

    async def modify_order(
        self,
        account: Account,
        order_id: str,
        order_type: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        validity: str = "DAY",
    ) -> OrderResponse:
        body = {
            "dhanClientId": account.dhan_client_id,
            "orderId": order_id,
            "orderType": order_type,
            "legName": None,
            "quantity": quantity,
            "price": price,
            "triggerPrice": trigger_price,
            "disclosedQuantity": 0,
            "validity": validity,
        }

        raw = await self._client.modify_order(order_id, body)

        return OrderResponse(
            order_id=raw["orderId"],
            status=raw.get("orderStatus", "PENDING"),
            broker="dhan",
            raw_response=raw,
        )

    async def cancel_order(
        self, account: Account, order_id: str,
    ) -> OrderResponse:
        raw = await self._client.cancel_order(order_id)
        return OrderResponse(
            order_id=raw["orderId"],
            status=raw.get("orderStatus", "CANCELLED"),
            broker="dhan",
            raw_response=raw,
        )

    async def get_order_status(
        self, account: Account, order_id: str,
    ) -> OrderDetail:
        raw = await self._client.get_order(order_id)
        return OrderDetail(
            order_id=raw["orderId"],
            status=raw["orderStatus"],
            filled_qty=raw["filledQty"],
            remaining_qty=raw["remainingQuantity"],
            avg_fill_price=raw["averageTradedPrice"],
            exchange_order_id=raw.get("exchangeOrderId"),
            exchange_time=raw.get("exchangeTime"),
            broker="dhan",
        )

    async def get_positions(self, account: Account) -> list[PositionDetail]:
        raw = await self._client.get_positions()
        positions = []
        for p in raw.get("data", []):
            net_qty = p.get("netQty", 0)
            if net_qty == 0:
                continue
            positions.append(PositionDetail(
                instrument_id=p["securityId"],
                exchange_segment=p["exchangeSegment"],
                product_type=p["productType"],
                quantity=net_qty,
                avg_price=p.get("averagePrice", 0.0),
                pnl=p.get("realizedProfit", 0.0) + p.get("unrealizedProfit", 0.0),
                broker="dhan",
            ))
        return positions

    async def place_forever_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        body = {
            "dhanClientId": account.dhan_client_id,
            "orderFlag": "SINGLE",
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,
            "productType": product_type,
            "orderType": order_type,
            "securityId": security_id,
            "quantity": quantity,
            "price": price,
            "triggerPrice": trigger_price,
            "correlationId": correlation_id,
        }

        raw = await self._client.place_forever_order(body)

        return OrderResponse(
            order_id=raw["orderId"],
            status=raw.get("orderStatus", "PENDING"),
            broker="dhan",
            raw_response=raw,
        )

    async def place_slicing_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        validity: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        body = {
            "dhanClientId": account.dhan_client_id,
            "transactionType": transaction_type,
            "exchangeSegment": exchange_segment,
            "productType": product_type,
            "orderType": order_type,
            "validity": validity,
            "securityId": security_id,
            "quantity": quantity,
            "price": price,
            "triggerPrice": trigger_price,
            "disclosedQuantity": 0,
            "afterMarketOrder": False,
            "amoTime": None,
            "boProfitValue": None,
            "boStopLossValue": None,
            "correlationId": correlation_id,
        }

        raw = await self._client.place_slicing_order(body)

        return OrderResponse(
            order_id=raw["orderId"],
            status=raw.get("orderStatus", "PENDING"),
            broker="dhan",
            raw_response=raw,
        )

    async def get_trades(self, account: Account, order_id: str) -> list[dict]:
        raw = await self._client.get_trades(order_id)
        return raw.get("trades", raw.get("data", []))
```

#### UpstoxAdapter (Failover Stub)

```python
class UpstoxClient:
    """
    HTTP client for the Upstox REST API. Structurally identical to DhanClient.
    Created on demand during failover.

    Base URL: https://api.upstox.com/v2
    Auth header: Authorization: Bearer {access_token}
    Rate limit: 50 OPS per API key (server-side)
    """

    BASE_URL = "https://api.upstox.com/v2"
    HFT_BASE_URL = "https://api-hft.upstox.com/v2"
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 0.5
    REQUEST_TIMEOUT = 10.0

    def __init__(self, account: Account):
        self.account = account
        self._token: str | None = None
        self._session: aiohttp.ClientSession | None = None

    @property
    def auth_headers(self) -> dict[str, str]:
        if self._token is None:
            raise RuntimeError(
                f"UpstoxClient for {self.account.account_id} not authenticated"
            )
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def set_token(self, token: str) -> None:
        self._token = token

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT)
        self._session = aiohttp.ClientSession(
            base_url=self.BASE_URL,
            headers=self.auth_headers,
            timeout=timeout,
        )

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def request(
        self, method: str, path: str, body: dict | None = None,
    ) -> dict:
        """Same retry logic as DhanClient."""
        # Implementation mirrors DhanClient.request with Upstox-specific
        # error codes and status handling.
        ...

    async def place_order(self, body: dict) -> dict:
        return await self.request("POST", "/order/place", body)

    async def modify_order(self, body: dict) -> dict:
        return await self.request("PUT", "/order/modify", body)

    async def cancel_order(self, order_id: str) -> dict:
        return await self.request("DELETE", f"/order/cancel?order_id={order_id}")

    async def get_order(self, order_id: str) -> dict:
        return await self.request("GET", f"/order/details?order_id={order_id}")

    async def get_positions(self) -> dict:
        return await self.request("GET", "/portfolio/short-term-positions")


class UpstoxAdapter(BrokerAdapter):
    """
    Upstox broker adapter for failover. Translates BrokerAdapter calls
    into Upstox REST API requests.

    Field mapping differences from Dhan:
    - security_id → instrument_key (via SymbolMapper)
    - exchangeSegment "NSE_FNO" → exchange "NSE", segment "FO"
    - productType "INTRADAY" → product "I"
    - orderType "LIMIT" → order_type "LIMIT" (same name, different casing)
    """

    PRODUCT_MAP = {
        "INTRADAY": "I",
        "CNC": "D",
        "MARGIN": "D",
    }

    def __init__(self, client: UpstoxClient, symbol_mapper: "SymbolMapper"):
        self._client = client
        self._mapper = symbol_mapper

    @property
    def broker_name(self) -> str:
        return "upstox"

    async def place_order(
        self,
        account: Account,
        transaction_type: Literal["BUY", "SELL"],
        exchange_segment: str,
        product_type: str,
        order_type: str,
        validity: str,
        security_id: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        correlation_id: str | None = None,
    ) -> OrderResponse:
        # Translate canonical security_id to Upstox instrument_key
        instrument_key = self._mapper.dhan_to_upstox(security_id)
        if instrument_key is None:
            raise SymbolMappingError(
                f"No Upstox mapping for Dhan security_id {security_id}"
            )

        body = {
            "quantity": quantity,
            "product": self.PRODUCT_MAP.get(product_type, "I"),
            "validity": validity,
            "price": price,
            "tag": correlation_id or "",
            "instrument_token": instrument_key,
            "order_type": order_type.upper(),
            "transaction_type": transaction_type.upper(),
            "disclosed_quantity": 0,
            "trigger_price": trigger_price or 0,
            "is_amo": False,
        }

        raw = await self._client.place_order(body)
        data = raw.get("data", {})

        return OrderResponse(
            order_id=data.get("order_id", ""),
            status=self._normalize_status(data.get("status", "")),
            broker="upstox",
            raw_response=raw,
        )

    async def modify_order(
        self,
        account: Account,
        order_id: str,
        order_type: str,
        quantity: int,
        price: float,
        trigger_price: float | None = None,
        validity: str = "DAY",
    ) -> OrderResponse:
        body = {
            "order_id": order_id,
            "quantity": quantity,
            "validity": validity,
            "price": price,
            "order_type": order_type.upper(),
            "trigger_price": trigger_price or 0,
            "disclosed_quantity": 0,
        }

        raw = await self._client.modify_order(body)
        data = raw.get("data", {})

        return OrderResponse(
            order_id=data.get("order_id", order_id),
            status=self._normalize_status(data.get("status", "PENDING")),
            broker="upstox",
            raw_response=raw,
        )

    async def cancel_order(
        self, account: Account, order_id: str,
    ) -> OrderResponse:
        raw = await self._client.cancel_order(order_id)
        data = raw.get("data", {})
        return OrderResponse(
            order_id=data.get("order_id", order_id),
            status="CANCELLED",
            broker="upstox",
            raw_response=raw,
        )

    async def get_order_status(
        self, account: Account, order_id: str,
    ) -> OrderDetail:
        raw = await self._client.get_order(order_id)
        data = raw.get("data", {})

        return OrderDetail(
            order_id=data.get("order_id", order_id),
            status=self._normalize_status(data.get("status", "")),
            filled_qty=data.get("filled_quantity", 0),
            remaining_qty=data.get("pending_quantity", 0),
            avg_fill_price=data.get("average_price", 0.0),
            exchange_order_id=data.get("exchange_order_id"),
            exchange_time=data.get("exchange_timestamp"),
            broker="upstox",
        )

    async def get_positions(self, account: Account) -> list[PositionDetail]:
        raw = await self._client.get_positions()
        positions = []
        for p in raw.get("data", []):
            net_qty = p.get("quantity", 0)
            if net_qty == 0:
                continue
            # Reverse-map Upstox instrument_key to Dhan security_id
            canonical_id = self._mapper.upstox_to_dhan(
                p.get("instrument_token", "")
            )
            positions.append(PositionDetail(
                instrument_id=canonical_id or p.get("instrument_token", ""),
                exchange_segment=p.get("exchange", ""),
                product_type=p.get("product", ""),
                quantity=net_qty,
                avg_price=p.get("average_price", 0.0),
                pnl=p.get("pnl", 0.0),
                broker="upstox",
            ))
        return positions

    async def place_forever_order(self, account, **kwargs) -> OrderResponse:
        # Upstox uses GTT API: POST /v2/gtt/orders
        # Structurally similar to Dhan Forever Orders
        raise NotImplementedError("Upstox GTT not implemented in v1 failover")

    async def place_slicing_order(self, account, **kwargs) -> OrderResponse:
        # Upstox does not have an auto-slicing endpoint.
        # Manual slicing required at the OMS level.
        raise NotImplementedError("Manual slicing required for Upstox")

    async def get_trades(self, account: Account, order_id: str) -> list[dict]:
        raw = await self._client.request("GET", f"/order/trades?order_id={order_id}")
        return raw.get("data", [])

    @staticmethod
    def _normalize_status(upstox_status: str) -> str:
        """Map Upstox order status to canonical status."""
        mapping = {
            "open": "PENDING",
            "complete": "TRADED",
            "partial fill": "PART_TRADED",
            "cancelled": "CANCELLED",
            "rejected": "REJECTED",
            "after market order req received": "TRANSIT",
            "modify pending": "TRANSIT",
            "cancel pending": "TRANSIT",
            "not cancelled": "PENDING",
            "not modified": "PENDING",
        }
        return mapping.get(upstox_status.lower(), "TRANSIT")
```

---

### Symbol Mapping

#### Canonical Identifier

The system uses Dhan `security_id` as the canonical instrument identifier everywhere: in signals, order requests, position tracking, risk calculations, and audit logs. When interacting with a non-Dhan broker, the `SymbolMapper` translates to that broker's native identifier.

| Broker | Identifier Field | Example (NIFTY 24500 CE weekly) |
|--------|-----------------|--------------------------------|
| Dhan | `security_id` (string of integer) | `"43925"` |
| Upstox | `instrument_key` (exchange\|segment\|symbol) | `"NSE_FO\|NIFTY24MAR24500CE"` |
| Angel One | `symboltoken` (string of integer) | `"58127"` |

#### SymbolMapper Class

```python
import csv
import io
import aiohttp
import asyncio
from datetime import date


class SymbolMapper:
    """
    Cross-broker instrument identifier mapping.

    Builds mapping tables from broker CSV instrument masters at 08:30 IST daily.
    The Dhan security_id is the canonical key. Maps to Upstox instrument_key
    and Angel One symboltoken using a composite match key:
    (exchange, symbol, expiry, strike, option_type).

    The composite key avoids ambiguity: two brokers may assign different
    numeric IDs to the same instrument, but the exchange-level attributes
    (underlying, expiry date, strike price, CE/PE) uniquely identify it.
    """

    DHAN_CSV_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
    UPSTOX_CSV_URL = "https://assets.upstox.com/market-quote/instruments/exchange/complete.csv.gz"
    ANGEL_CSV_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

    def __init__(self):
        # Primary maps: canonical (Dhan security_id) → broker-specific ID
        self._dhan_to_upstox: dict[str, str] = {}
        self._upstox_to_dhan: dict[str, str] = {}
        self._dhan_to_angel: dict[str, str] = {}
        self._angel_to_dhan: dict[str, str] = {}

        # Metadata for the canonical ID
        self._dhan_meta: dict[str, InstrumentMeta] = {}

        # Build status
        self._built: bool = False
        self._build_date: date | None = None
        self._build_errors: list[str] = []

    async def build(self) -> None:
        """
        Download all three broker CSVs and build cross-mapping tables.
        Called at 08:30 IST during system startup (Phase 2: Data Prep).

        Downloads happen in parallel. If Upstox or Angel CSV fails,
        the system continues with Dhan-only mode (failover will be
        degraded but primary broker still works).
        """
        logger.info("symbol_mapper_build_start")

        # Download all CSVs in parallel
        dhan_task = asyncio.create_task(self._download_dhan_csv())
        upstox_task = asyncio.create_task(self._download_upstox_csv())

        dhan_instruments = await dhan_task

        try:
            upstox_instruments = await upstox_task
        except Exception as e:
            logger.error("upstox_csv_download_failed", error=str(e))
            upstox_instruments = {}
            self._build_errors.append(f"Upstox CSV failed: {e}")

        # Build Dhan metadata index (canonical)
        for inst in dhan_instruments:
            meta = InstrumentMeta(
                security_id=inst["SEM_SMST_SECURITY_ID"],
                trading_symbol=inst["SEM_TRADING_SYMBOL"],
                exchange=inst["SEM_EXM_EXCH_ID"],
                segment=inst.get("SEM_SEGMENT", ""),
                instrument_type=inst.get("SEM_INSTRUMENT_NAME", ""),
                underlying=inst.get("SEM_CUSTOM_SYMBOL", ""),
                expiry=inst.get("SEM_EXPIRY_DATE", ""),
                strike=float(inst.get("SEM_STRIKE_PRICE", 0)),
                option_type=inst.get("SEM_OPTION_TYPE", ""),
                lot_size=int(inst.get("SEM_LOT_UNITS", 1)),
                tick_size=float(inst.get("SEM_TICK_SIZE", 0.05)),
                freeze_qty=int(inst.get("FREEZE_QTY", 0)),
            )
            self._dhan_meta[meta.security_id] = meta

        # Build Upstox composite key index
        upstox_by_key: dict[str, str] = {}
        for inst in upstox_instruments:
            composite = self._make_composite_key(
                exchange=inst.get("exchange", ""),
                symbol=inst.get("tradingsymbol", ""),
                expiry=inst.get("expiry", ""),
                strike=float(inst.get("strike", 0)),
                option_type=inst.get("option_type", ""),
            )
            upstox_by_key[composite] = inst.get("instrument_key", "")

        # Cross-map: for each Dhan instrument, find Upstox equivalent
        matched = 0
        unmatched = 0
        for sec_id, meta in self._dhan_meta.items():
            composite = self._make_composite_key(
                exchange=meta.exchange,
                symbol=meta.trading_symbol,
                expiry=meta.expiry,
                strike=meta.strike,
                option_type=meta.option_type,
            )
            upstox_key = upstox_by_key.get(composite)
            if upstox_key:
                self._dhan_to_upstox[sec_id] = upstox_key
                self._upstox_to_dhan[upstox_key] = sec_id
                matched += 1
            else:
                unmatched += 1

        self._built = True
        self._build_date = date.today()

        logger.info(
            "symbol_mapper_build_complete",
            dhan_instruments=len(dhan_instruments),
            upstox_instruments=len(upstox_instruments),
            matched=matched,
            unmatched=unmatched,
        )

    @staticmethod
    def _make_composite_key(
        exchange: str,
        symbol: str,
        expiry: str,
        strike: float,
        option_type: str,
    ) -> str:
        """
        Composite key for cross-broker matching.
        Normalizes fields to handle formatting differences between brokers.
        """
        # Normalize exchange: NSE, BSE
        norm_exchange = exchange.upper().replace("_EQ", "").replace("_FNO", "")
        # Normalize expiry to YYYY-MM-DD
        norm_expiry = expiry[:10] if expiry else ""
        # Normalize strike: remove trailing zeros
        norm_strike = f"{strike:.2f}".rstrip("0").rstrip(".")
        # Normalize option type
        norm_opt = option_type.upper()[:2] if option_type else ""

        return f"{norm_exchange}|{symbol}|{norm_expiry}|{norm_strike}|{norm_opt}"

    async def _download_dhan_csv(self) -> list[dict]:
        """Download and parse Dhan instrument master CSV."""
        async with aiohttp.ClientSession() as session:
            async with session.get(self.DHAN_CSV_URL, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Dhan CSV download failed: HTTP {resp.status}")
                text = await resp.text()

        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    async def _download_upstox_csv(self) -> list[dict]:
        """Download and parse Upstox instrument master CSV (gzipped)."""
        import gzip

        async with aiohttp.ClientSession() as session:
            async with session.get(self.UPSTOX_CSV_URL, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Upstox CSV download failed: HTTP {resp.status}")
                raw_bytes = await resp.read()

        text = gzip.decompress(raw_bytes).decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        return list(reader)

    # --- Lookup methods ---

    def dhan_to_upstox(self, security_id: str) -> str | None:
        """Translate Dhan security_id to Upstox instrument_key."""
        return self._dhan_to_upstox.get(security_id)

    def upstox_to_dhan(self, instrument_key: str) -> str | None:
        """Translate Upstox instrument_key to Dhan security_id."""
        return self._upstox_to_dhan.get(instrument_key)

    def dhan_to_angel(self, security_id: str) -> str | None:
        """Translate Dhan security_id to Angel One symboltoken."""
        return self._dhan_to_angel.get(security_id)

    def get_meta(self, security_id: str) -> "InstrumentMeta | None":
        """Get instrument metadata by canonical ID."""
        return self._dhan_meta.get(security_id)

    def validate_mapping(self, security_id: str, target_broker: str) -> bool:
        """Check if a mapping exists for the given security_id and broker."""
        if target_broker == "upstox":
            return security_id in self._dhan_to_upstox
        if target_broker == "angel":
            return security_id in self._dhan_to_angel
        return True  # Dhan-to-Dhan always valid

    @property
    def build_status(self) -> dict:
        return {
            "built": self._built,
            "build_date": str(self._build_date),
            "dhan_count": len(self._dhan_meta),
            "upstox_mapped": len(self._dhan_to_upstox),
            "angel_mapped": len(self._dhan_to_angel),
            "errors": self._build_errors,
        }


class InstrumentMeta(pydantic.BaseModel):
    """Instrument metadata from the canonical (Dhan) instrument master."""
    security_id: str
    trading_symbol: str
    exchange: str
    segment: str
    instrument_type: str              # OPTIDX, FUTIDX, EQUITY, etc.
    underlying: str                    # NIFTY, BANKNIFTY, etc.
    expiry: str                        # YYYY-MM-DD or empty for equity
    strike: float                      # 0 for futures/equity
    option_type: str                   # CE, PE, or empty
    lot_size: int
    tick_size: float
    freeze_qty: int


class SymbolMappingError(Exception):
    pass
```

#### Startup Validation

At 08:30, after the SymbolMapper builds, a validation pass runs to confirm all instruments the strategies will trade are mapped:

```python
async def validate_strategy_instruments(
    mapper: SymbolMapper,
    strategy_configs: list["StrategyConfig"],
) -> list[str]:
    """
    Validate that all instruments the strategies might trade have
    cross-broker mappings. Returns list of warnings for unmapped instruments.

    Called at startup, before trading begins.
    """
    warnings = []

    for config in strategy_configs:
        for underlying in config.underlyings:
            # Check that the underlying's FNO segment instruments
            # have Upstox mappings (for failover readiness)
            meta_list = [
                m for m in mapper._dhan_meta.values()
                if m.underlying == underlying
                and m.segment in ("NSE_FNO", "FNO")
            ]

            mapped = sum(
                1 for m in meta_list
                if mapper.validate_mapping(m.security_id, "upstox")
            )
            total = len(meta_list)

            if total > 0 and mapped / total < 0.95:
                msg = (
                    f"Strategy {config.strategy_id}: "
                    f"{underlying} has only {mapped}/{total} "
                    f"({100*mapped/total:.1f}%) Upstox mappings"
                )
                warnings.append(msg)
                logger.warning("symbol_mapping_coverage_low", message=msg)

    return warnings
```

---

### Auth Lifecycle

#### Parallel Account Authentication

All accounts authenticate in parallel at 08:25 IST (Phase 1 of the startup sequence). Each account's token is stored in Redis for cross-process access.

```python
class AuthManager:
    """
    Manages per-account authentication with all configured brokers.

    08:25 IST: authenticate all accounts in parallel
    Token storage: Redis AUTH:{broker}:{account_id}:token
    Failed auth: SUSPEND that account (no orders, existing SLs remain)
    Shutdown: logout all accounts via atexit
    """

    DHAN_LOGIN_URL = "https://api.dhan.co/v2/token"
    TOKEN_TTL_SECONDS = 86400          # 24 hours (Dhan tokens valid for 1 day)
    AUTH_TIMEOUT = 30.0                # seconds per auth attempt
    MAX_AUTH_RETRIES = 3

    def __init__(self, router: "BrokerRouter"):
        self._router = router
        self._redis: "aioredis.Redis | None" = None

    async def authenticate_all(
        self,
        accounts: list[Account],
        redis: "aioredis.Redis",
    ) -> dict[str, bool]:
        """
        Authenticate all accounts in parallel. Returns dict of
        account_id -> success.

        Failed accounts are set to SUSPENDED state. They receive no new
        orders but existing server-side SLs remain active on the broker.
        """
        self._redis = redis
        results: dict[str, bool] = {}

        tasks = [
            asyncio.create_task(
                self._auth_single_account(account),
                name=f"auth_{account.account_id}",
            )
            for account in accounts
        ]

        completed = await asyncio.gather(*tasks, return_exceptions=True)

        for account, result in zip(accounts, completed):
            if isinstance(result, Exception):
                logger.critical(
                    "account_auth_failed",
                    account_id=account.account_id,
                    error=str(result),
                )
                self._router._broker_state[account.account_id].health = (
                    BrokerHealth.SUSPENDED
                )
                results[account.account_id] = False
                await telegram.send(
                    CRITICAL,
                    f"AUTH FAILED: Account {account.account_id}. "
                    f"Suspended for today. Error: {result}",
                )
            else:
                results[account.account_id] = True

        auth_ok = sum(1 for v in results.values() if v)
        auth_fail = sum(1 for v in results.values() if not v)
        logger.info(
            "auth_all_complete",
            total=len(accounts),
            success=auth_ok,
            failed=auth_fail,
        )

        if auth_ok == 0:
            raise SystemStartupError(
                "All accounts failed authentication. Cannot start trading."
            )

        return results

    async def _auth_single_account(self, account: Account) -> None:
        """
        Authenticate a single account with its primary broker.

        For Dhan:
        1. Use the API key to obtain an access token
        2. Store token in Redis with TTL
        3. Set token on the DhanClient instance
        4. Verify token by making a test API call (GET /v2/orders)
        """
        last_error: Exception | None = None

        for attempt in range(self.MAX_AUTH_RETRIES):
            try:
                # Dhan auth: API key is the access token (no separate login flow)
                # The api_key provided at account setup IS the access token
                token = account.dhan_access_token

                # Verify token is valid by making a lightweight call
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{DhanClient.BASE_URL}/v2/orders",
                        headers={
                            "access-token": token,
                            "Content-Type": "application/json",
                        },
                        timeout=aiohttp.ClientTimeout(total=self.AUTH_TIMEOUT),
                    ) as resp:
                        if resp.status == 401:
                            raise DhanAuthError(
                                account.account_id,
                                "Token rejected (HTTP 401)",
                            )
                        if resp.status >= 500:
                            raise DhanAPIError(
                                account.account_id,
                                resp.status,
                                None,
                                "Auth verification failed (server error)",
                            )
                        # 200 or other 2xx/4xx = token is valid (4xx might be
                        # "no orders" which is fine for verification)

                # Store in Redis
                redis_key = f"AUTH:dhan:{account.account_id}:token"
                await self._redis.set(redis_key, token, ex=self.TOKEN_TTL_SECONDS)

                # Set on the DhanClient instance
                client = self._router._dhan_clients[account.account_id]
                client.set_token(token)
                await client.start()

                # Update broker state
                state = self._router._broker_state[account.account_id]
                state.health = BrokerHealth.HEALTHY
                state.record_success()

                logger.info(
                    "account_auth_success",
                    account_id=account.account_id,
                    broker="dhan",
                    attempt=attempt + 1,
                )
                return

            except (DhanAuthError, DhanAPIError) as e:
                last_error = e
                logger.warning(
                    "account_auth_attempt_failed",
                    account_id=account.account_id,
                    attempt=attempt + 1,
                    error=str(e),
                )
                if attempt < self.MAX_AUTH_RETRIES - 1:
                    await asyncio.sleep(2.0 * (attempt + 1))

        raise last_error or RuntimeError(
            f"Auth failed for {account.account_id} after {self.MAX_AUTH_RETRIES} attempts"
        )

    async def refresh_token(self, account: Account) -> bool:
        """
        Refresh an account's token mid-session. Called when a 401 is
        detected during normal operation.

        Returns True if refresh succeeded, False if account must be suspended.
        """
        try:
            await self._auth_single_account(account)
            logger.info("token_refresh_success", account_id=account.account_id)
            return True
        except Exception as e:
            logger.critical(
                "token_refresh_failed",
                account_id=account.account_id,
                error=str(e),
            )
            self._router._broker_state[account.account_id].health = (
                BrokerHealth.SUSPENDED
            )
            await telegram.send(
                CRITICAL,
                f"TOKEN REFRESH FAILED: Account {account.account_id}. "
                f"Suspended. Manual re-auth required.",
            )
            return False

    async def logout_all(self) -> None:
        """
        Clean shutdown: close all broker sessions. Registered via
        atexit.register(asyncio.run, auth_manager.logout_all).

        Tokens are NOT invalidated (Dhan tokens expire naturally after 24h).
        This only closes HTTP connections and cleans up resources.
        """
        for account_id, client in self._router._dhan_clients.items():
            try:
                await client.close()
                logger.info("client_closed", account_id=account_id, broker="dhan")
            except Exception:
                pass  # best-effort on shutdown

        for account_id, client in self._router._upstox_clients.items():
            try:
                await client.close()
                logger.info("client_closed", account_id=account_id, broker="upstox")
            except Exception:
                pass

        logger.info("logout_all_complete")
```

#### Token Redis Layout

```
AUTH:dhan:{account_id}:token        → access token string (TTL 86400s)
AUTH:dhan:{account_id}:auth_ts      → ISO timestamp of last auth
AUTH:dhan:{account_id}:status       → "active" | "expired" | "suspended"
AUTH:upstox:{account_id}:token      → access token string (TTL 86400s)
AUTH:upstox:{account_id}:auth_ts    → ISO timestamp of last auth
AUTH:upstox:{account_id}:status     → "active" | "expired" | "suspended"
```

#### atexit Registration

```python
import atexit
import asyncio


def register_shutdown(auth_manager: AuthManager) -> None:
    """Register clean shutdown handler at system startup."""

    def _sync_logout():
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(auth_manager.logout_all())
            else:
                loop.run_until_complete(auth_manager.logout_all())
        except Exception:
            pass  # best-effort

    atexit.register(_sync_logout)
```

---

### Failover Logic

#### Health Monitoring

The broker router runs a continuous health monitor per account. Health is assessed from two signals: REST API success/failure rates and WebSocket connection state.

```python
class BrokerHealthMonitor:
    """
    Continuous health monitor for per-account broker connections.
    Runs as a background asyncio task. Checks health every 10 seconds.
    """

    CHECK_INTERVAL = 10.0              # seconds
    CONSECUTIVE_FAILURE_THRESHOLD = 3   # failures before FAILED state
    WS_DISCONNECT_THRESHOLD = 30.0     # seconds of WS disconnect before FAILED
    FAILBACK_COOLDOWN = 300.0          # seconds: wait 5 min before failback

    def __init__(self, router: "BrokerRouter"):
        self._router = router
        self._running = False

    async def run(self) -> None:
        """Main monitoring loop. One loop covers all accounts."""
        self._running = True

        while self._running:
            for account_id, state in self._router._broker_state.items():
                if state.health == BrokerHealth.SUSPENDED:
                    continue  # skip suspended accounts

                await self._check_account_health(account_id, state)

            await asyncio.sleep(self.CHECK_INTERVAL)

    async def _check_account_health(
        self, account_id: str, state: AccountBrokerState,
    ) -> None:
        # Check WS disconnect duration
        if state.ws_disconnect_ts is not None:
            disconnect_duration = time.monotonic() - state.ws_disconnect_ts
            if disconnect_duration > self.WS_DISCONNECT_THRESHOLD:
                if state.health != BrokerHealth.FAILED:
                    logger.warning(
                        "ws_disconnect_threshold_exceeded",
                        account_id=account_id,
                        duration_s=disconnect_duration,
                    )
                    state.health = BrokerHealth.FAILED

        # Check consecutive REST failures
        if state.consecutive_failures >= self.CONSECUTIVE_FAILURE_THRESHOLD:
            if state.health not in (
                BrokerHealth.FAILED, BrokerHealth.FAILOVER_ACTIVE,
            ):
                state.health = BrokerHealth.FAILED

        # Trigger failover if FAILED and not already in failover
        if (
            state.health == BrokerHealth.FAILED
            and state.active_broker == state.primary_broker
        ):
            await self._initiate_failover(account_id, state)

        # Check for failback eligibility
        if state.health == BrokerHealth.FAILOVER_ACTIVE:
            await self._check_failback(account_id, state)

    async def _initiate_failover(
        self, account_id: str, state: AccountBrokerState,
    ) -> None:
        """
        Failover procedure for a single account:
        1. Hedge open positions on backup broker
        2. Switch active broker to backup
        3. Continue monitoring primary for recovery
        """
        logger.critical(
            "failover_initiated",
            account_id=account_id,
            primary=state.primary_broker,
            reason=f"consecutive_failures={state.consecutive_failures}, "
                   f"ws_connected={state.ws_connected}",
        )

        account = self._router._accounts[account_id]

        # Initialize backup broker client if needed
        backup_broker = "upstox" if state.primary_broker == "dhan" else "dhan"

        if backup_broker == "upstox" and account_id not in self._router._upstox_clients:
            if not account.upstox_client_id or not account.upstox_api_key:
                logger.critical(
                    "failover_no_backup_configured",
                    account_id=account_id,
                )
                await telegram.send(
                    CRITICAL,
                    f"FAILOVER FAILED: Account {account_id} has no Upstox "
                    f"credentials configured. Positions protected by server-side SLs only.",
                )
                return

            # Auth with backup broker
            try:
                upstox_client = UpstoxClient(account)
                upstox_client.set_token(account.upstox_api_key)
                await upstox_client.start()
                self._router._upstox_clients[account_id] = upstox_client
            except Exception as e:
                logger.critical(
                    "failover_backup_auth_failed",
                    account_id=account_id,
                    error=str(e),
                )
                await telegram.send(
                    CRITICAL,
                    f"FAILOVER FAILED: Upstox auth failed for {account_id}. "
                    f"Positions protected by server-side SLs only. Error: {e}",
                )
                return

        # Hedge open positions on backup broker
        await self._router._hedge_manager.hedge_account(account_id, backup_broker)

        # Switch active broker
        state.active_broker = backup_broker
        state.health = BrokerHealth.FAILOVER_ACTIVE
        state.failover_ts = time.monotonic()
        state.failover_reason = (
            f"consecutive_failures={state.consecutive_failures}, "
            f"ws_connected={state.ws_connected}"
        )
        state.failback_eligible_ts = time.monotonic() + self.FAILBACK_COOLDOWN

        await telegram.send(
            CRITICAL,
            f"FAILOVER ACTIVE: Account {account_id} switched from "
            f"{state.primary_broker} to {backup_broker}. "
            f"Hedge positions placed. Monitoring primary for recovery.",
        )

    async def _check_failback(
        self, account_id: str, state: AccountBrokerState,
    ) -> None:
        """
        Check if primary broker has recovered and failback is safe.

        Failback requirements:
        1. At least FAILBACK_COOLDOWN seconds since failover
        2. Primary broker responds to a health-check API call
        3. Primary broker WS reconnects successfully
        """
        if state.failback_eligible_ts and time.monotonic() < state.failback_eligible_ts:
            return  # cooldown not elapsed

        # Probe primary broker with a lightweight call
        try:
            account = self._router._accounts[account_id]

            if state.primary_broker == "dhan":
                client = self._router._dhan_clients[account_id]
                await client.request("GET", "/v2/orders")
            else:
                # Upstox health probe
                pass

            # Primary is back. Initiate failback.
            await self._initiate_failback(account_id, state)

        except Exception:
            # Primary still down. Check again next cycle.
            pass

    async def _initiate_failback(
        self, account_id: str, state: AccountBrokerState,
    ) -> None:
        """
        Failback to primary broker:
        1. Unwind hedge positions on backup broker
        2. Switch active broker back to primary
        3. Re-establish WS connections
        """
        logger.info("failback_initiated", account_id=account_id)

        # Unwind hedges on backup broker
        await self._router._hedge_manager.unwind_hedges(account_id)

        # Switch back to primary
        state.active_broker = state.primary_broker
        state.health = BrokerHealth.HEALTHY
        state.consecutive_failures = 0
        state.failover_ts = None
        state.failover_reason = None
        state.failback_eligible_ts = None

        # Re-establish WS
        ws = self._router._order_ws.get(account_id)
        if ws:
            try:
                await ws.connect()
                state.ws_connected = True
                state.ws_disconnect_ts = None
            except Exception as e:
                logger.warning(
                    "failback_ws_reconnect_failed",
                    account_id=account_id,
                    error=str(e),
                )

        await telegram.send(
            WARNING,
            f"FAILBACK COMPLETE: Account {account_id} returned to "
            f"{state.primary_broker}. Hedges unwound.",
        )
```

#### Failover State Machine

```
                       ┌──────────────────────────────────┐
                       │                                  │
                       ▼                                  │
                 ┌───────────┐                            │
          ┌─────│  HEALTHY   │◄───────────────────┐       │
          │     └─────┬──────┘                    │       │
          │           │                            │       │
          │           │ intermittent failure        │       │
          │           │ (1-2 consecutive)           │       │
          │           ▼                            │       │
          │     ┌───────────┐                      │       │
          │     │  DEGRADED  │                     │       │
          │     └─────┬──────┘                     │       │
          │           │                            │       │
          │           │ success                    │       │
          │           │ (resets count)              │       │
          │           └────────────────────────────┘       │
          │                                                │
          │  3 consecutive failures                        │
          │  OR WS disconnect > 30s                        │
          │                                                │
          ▼                                                │
    ┌───────────┐                                          │
    │   FAILED   │                                         │
    └─────┬──────┘                                         │
          │                                                │
          │ backup broker available                        │
          │ + hedge positions placed                       │
          ▼                                                │
    ┌───────────────┐                                      │
    │ FAILOVER_ACTIVE│                                     │
    └─────┬─────────┘                                      │
          │                                                │
          │ primary recovers                               │
          │ + cooldown elapsed (5 min)                     │
          │ + hedges unwound                               │
          │                                                │
          └────────────────────────────────────────────────┘

    ┌───────────┐
    │ SUSPENDED  │  ← auth failure or operator action
    └───────────┘    (no automatic recovery)
```

---

### Cross-Broker Hedging (Finding 4)

When Dhan fails for a specific account that has open positions, those positions must be hedged on the backup broker. Each account is hedged independently. Account A's Dhan failure does not affect Account B.

#### Hedge Strategy

| Position Type | Hedge Method | Hedge Instrument |
|--------------|-------------|------------------|
| Long futures | Short same future on Upstox | Same underlying, same expiry |
| Short futures | Long same future on Upstox | Same underlying, same expiry |
| Long call option | Short ATM call on Upstox (delta-adjusted) | Same underlying, nearest strike, same expiry |
| Short call option | Long ATM call on Upstox (delta-adjusted) | Same underlying, nearest strike, same expiry |
| Long put option | Short ATM put on Upstox (delta-adjusted) | Same underlying, nearest strike, same expiry |
| Short put option | Long ATM put on Upstox (delta-adjusted) | Same underlying, nearest strike, same expiry |

**Delta adjustment for options:** The hedge quantity is scaled by the position's delta. A 10-lot long call at delta 0.5 requires a 5-lot hedge (short call at delta ~0.5) or a 5-lot short futures hedge. The system uses the simpler option-for-option hedge where possible, falling back to futures for illiquid strikes.

#### HedgeManager Implementation

```python
class HedgePosition(pydantic.BaseModel):
    """Tracks a hedge position placed on the backup broker."""
    account_id: str
    original_security_id: str           # canonical ID of the position being hedged
    hedge_security_id: str              # canonical ID of the hedge instrument
    hedge_broker: Literal["dhan", "upstox"]
    hedge_order_id: str
    hedge_qty: int                      # signed: positive = long, negative = short
    hedge_price: float
    hedge_status: Literal["PENDING", "FILLED", "PARTIAL", "FAILED"]
    delta_ratio: float                  # 1.0 for futures, 0.0-1.0 for options
    placed_ts: float


class HedgeManager:
    """
    Places and manages cross-broker hedges during failover.

    Hedging rules:
    1. Futures: perfect hedge (1:1 opposite position on backup broker)
    2. Options: delta-adjusted hedge (scale qty by position delta)
    3. Each account hedged independently
    4. Hedge uses IOC validity to avoid hanging orders
    5. Failed hedges: Telegram CRITICAL, operator must intervene
    """

    def __init__(self, router: "BrokerRouter"):
        self._router = router
        self._hedges: dict[str, list[HedgePosition]] = {}  # account_id → hedges

    async def hedge_account(
        self, account_id: str, backup_broker: str,
    ) -> list[HedgePosition]:
        """
        Hedge all open positions for one account on the backup broker.

        Steps:
        1. Query primary broker for open positions (may fail — use cached)
        2. For each position, determine hedge instrument on backup broker
        3. Place hedge orders on backup broker
        4. Track hedge positions for later unwinding
        """
        account = self._router._accounts[account_id]
        hedges: list[HedgePosition] = []

        # Try to get positions from primary broker
        positions = await self._get_positions_best_effort(account_id)

        if not positions:
            logger.warning(
                "hedge_no_positions",
                account_id=account_id,
                reason="no_open_positions_or_query_failed",
            )
            return hedges

        # Get backup broker adapter
        adapter = self._router.get_client(account_id)
        if adapter.broker_name != backup_broker:
            # Force backup broker
            if backup_broker == "upstox":
                upstox_client = self._router._upstox_clients.get(account_id)
                if not upstox_client:
                    logger.critical(
                        "hedge_no_backup_client", account_id=account_id,
                    )
                    return hedges
                adapter = UpstoxAdapter(upstox_client, self._router._symbol_mapper)

        for position in positions:
            try:
                hedge = await self._hedge_single_position(
                    account, position, adapter, backup_broker,
                )
                if hedge:
                    hedges.append(hedge)
            except Exception as e:
                logger.critical(
                    "hedge_position_failed",
                    account_id=account_id,
                    security_id=position.instrument_id,
                    error=str(e),
                )
                await telegram.send(
                    CRITICAL,
                    f"HEDGE FAILED: Account {account_id}, "
                    f"instrument {position.instrument_id}. "
                    f"Position unhedged. Error: {e}",
                )

        self._hedges[account_id] = hedges

        # Update broker state with hedge info
        self._router._broker_state[account_id].hedge_positions = hedges

        logger.info(
            "hedge_account_complete",
            account_id=account_id,
            positions_found=len(positions),
            hedges_placed=len(hedges),
            hedges_failed=len(positions) - len(hedges),
        )

        return hedges

    async def _hedge_single_position(
        self,
        account: Account,
        position: PositionDetail,
        adapter: BrokerAdapter,
        backup_broker: str,
    ) -> HedgePosition | None:
        """Place a hedge for a single position."""
        meta = self._router._symbol_mapper.get_meta(position.instrument_id)
        if meta is None:
            logger.error(
                "hedge_no_meta",
                security_id=position.instrument_id,
            )
            return None

        # Determine hedge direction (opposite of position)
        hedge_txn = "SELL" if position.quantity > 0 else "BUY"
        hedge_qty = abs(position.quantity)

        # Delta adjustment for options
        delta_ratio = 1.0
        if meta.option_type in ("CE", "PE"):
            delta = self._estimate_delta(meta)
            delta_ratio = abs(delta)
            hedge_qty = max(
                meta.lot_size,
                int(hedge_qty * delta_ratio / meta.lot_size) * meta.lot_size,
            )

        # Translate security_id for backup broker
        if backup_broker == "upstox":
            hedge_security_id = self._router._symbol_mapper.dhan_to_upstox(
                position.instrument_id
            )
            if not hedge_security_id:
                logger.error(
                    "hedge_no_symbol_mapping",
                    security_id=position.instrument_id,
                    broker=backup_broker,
                )
                return None
        else:
            hedge_security_id = position.instrument_id

        # Place hedge order with MARKET + IOC for immediate fill
        try:
            resp = await adapter.place_order(
                account=account,
                transaction_type=hedge_txn,
                exchange_segment="NSE_FNO",
                product_type="INTRADAY",
                order_type="MARKET",
                validity="IOC",
                security_id=position.instrument_id,  # adapter translates internally
                quantity=hedge_qty,
                price=0,
                trigger_price=None,
                correlation_id=f"HEDGE_{account.account_id[:8]}_{position.instrument_id[:8]}",
            )

            return HedgePosition(
                account_id=account.account_id,
                original_security_id=position.instrument_id,
                hedge_security_id=hedge_security_id,
                hedge_broker=backup_broker,
                hedge_order_id=resp.order_id,
                hedge_qty=-hedge_qty if hedge_txn == "SELL" else hedge_qty,
                hedge_price=0.0,  # market order — actual price from fill
                hedge_status="PENDING",
                delta_ratio=delta_ratio,
                placed_ts=time.monotonic(),
            )

        except Exception as e:
            logger.critical(
                "hedge_order_placement_failed",
                account_id=account.account_id,
                security_id=position.instrument_id,
                error=str(e),
            )
            return None

    def _estimate_delta(self, meta: InstrumentMeta) -> float:
        """
        Estimate option delta for hedge sizing. Uses a rough approximation
        based on moneyness (ATM ~ 0.5, deep ITM ~ 0.9, deep OTM ~ 0.1).

        A full Black-Scholes delta calculation is not needed here because:
        1. Hedge is temporary (minutes to hours)
        2. Approximate hedge is better than no hedge
        3. Exact delta requires IV which may not be available during failover
        """
        # Without current spot price during failover, assume ATM
        # This is conservative: over-hedges OTM, under-hedges ITM
        if meta.option_type == "CE":
            return 0.5
        elif meta.option_type == "PE":
            return -0.5
        return 1.0  # futures

    async def _get_positions_best_effort(
        self, account_id: str,
    ) -> list[PositionDetail]:
        """
        Try to get positions from the primary broker. If that fails
        (which is likely during failover), fall back to the cached
        position state from the Position Tracker (Redis).
        """
        try:
            client = self._router._dhan_clients[account_id]
            adapter = DhanAdapter(client)
            account = self._router._accounts[account_id]
            return await adapter.get_positions(account)
        except Exception:
            # Primary is down — query cached positions from Redis
            return await self._get_cached_positions(account_id)

    async def _get_cached_positions(
        self, account_id: str,
    ) -> list[PositionDetail]:
        """Read cached positions from Redis POS:{account_id}:* keys."""
        # Position Tracker writes POS:{account_id}:{security_id} on every fill
        positions = []
        redis = self._router._auth_manager._redis
        if redis is None:
            return positions

        cursor = 0
        pattern = f"POS:{account_id}:*"
        while True:
            cursor, keys = await redis.scan(cursor, match=pattern, count=100)
            for key in keys:
                data = await redis.hgetall(key)
                if data and int(data.get(b"quantity", 0)) != 0:
                    positions.append(PositionDetail(
                        instrument_id=data.get(b"security_id", b"").decode(),
                        exchange_segment=data.get(b"exchange_segment", b"").decode(),
                        product_type=data.get(b"product_type", b"").decode(),
                        quantity=int(data.get(b"quantity", 0)),
                        avg_price=float(data.get(b"avg_price", 0)),
                        pnl=float(data.get(b"pnl", 0)),
                        broker="dhan",
                    ))
            if cursor == 0:
                break

        return positions

    async def unwind_hedges(self, account_id: str) -> None:
        """
        Unwind all hedge positions for an account after failback.
        Places opposite orders on the backup broker to close hedges.
        """
        hedges = self._hedges.get(account_id, [])
        if not hedges:
            return

        for hedge in hedges:
            try:
                # Close the hedge: opposite transaction
                close_txn = "BUY" if hedge.hedge_qty < 0 else "SELL"
                close_qty = abs(hedge.hedge_qty)

                if hedge.hedge_broker == "upstox":
                    client = self._router._upstox_clients.get(account_id)
                    if client:
                        adapter = UpstoxAdapter(client, self._router._symbol_mapper)
                        account = self._router._accounts[account_id]
                        await adapter.place_order(
                            account=account,
                            transaction_type=close_txn,
                            exchange_segment="NSE_FNO",
                            product_type="INTRADAY",
                            order_type="MARKET",
                            validity="IOC",
                            security_id=hedge.original_security_id,
                            quantity=close_qty,
                            price=0,
                            correlation_id=f"UNHEDGE_{account_id[:8]}",
                        )

                logger.info(
                    "hedge_unwound",
                    account_id=account_id,
                    security_id=hedge.original_security_id,
                    qty=close_qty,
                )

            except Exception as e:
                logger.critical(
                    "hedge_unwind_failed",
                    account_id=account_id,
                    security_id=hedge.original_security_id,
                    error=str(e),
                )
                await telegram.send(
                    CRITICAL,
                    f"HEDGE UNWIND FAILED: Account {account_id}, "
                    f"instrument {hedge.original_security_id}. "
                    f"Manual close required on {hedge.hedge_broker}.",
                )

        # Clear hedge tracking
        self._hedges.pop(account_id, None)
        self._router._broker_state[account_id].hedge_positions = []
```

---

### Rate Limiting

Per-account rate limits are enforced by the `PriorityRateLimiter` defined in the OMS (Component 5). The broker router does not implement its own rate limiter but delegates to the OMS's per-account instances.

| Broker | OPS Limit | Scope | Enforced By |
|--------|-----------|-------|-------------|
| Dhan | 10 per API key | Per-account | `PriorityRateLimiter` in OMS |
| Upstox | 50 per API key | Per-account | Separate limiter created on failover |

```python
# During failover, create a new rate limiter for the Upstox account
async def _initiate_failover(self, account_id, state):
    # ... (failover setup) ...

    # Upstox gets its own rate limiter at 50 OPS
    self._router._failover_rate_limiters[account_id] = PriorityRateLimiter(rate=50)
```

The OMS checks `broker_router.get_state(account_id).active_broker` to decide which rate limiter to use. If the account is on Dhan, it uses the 10 OPS limiter. If on Upstox failover, it uses the 50 OPS limiter.

---

### State Table

| State Item | Storage | Scope | Lifecycle | Writer | Readers |
|-----------|---------|-------|-----------|--------|---------|
| DhanClient instances | In-memory | Per-account | Session | BrokerRouter (startup) | OMS, HedgeManager |
| UpstoxClient instances | In-memory | Per-account | Created on failover, destroyed on failback | BrokerHealthMonitor | OMS (during failover) |
| AccountBrokerState | In-memory + Redis mirror | Per-account | Session | BrokerHealthMonitor | OMS, Risk Monitor, Monitoring |
| Auth tokens (Dhan) | Redis `AUTH:dhan:{account_id}:token` | Per-account | TTL 86400s | AuthManager | DhanClient |
| Auth tokens (Upstox) | Redis `AUTH:upstox:{account_id}:token` | Per-account | TTL 86400s | AuthManager | UpstoxClient |
| SymbolMapper tables | In-memory | Shared (all accounts) | Built once at 08:30, immutable for session | SymbolMapper | BrokerRouter, OMS, HedgeManager |
| Dhan instrument CSV | Downloaded at 08:30 | Shared | Session | SymbolMapper | SymbolMapper |
| Upstox instrument CSV | Downloaded at 08:30 | Shared | Session | SymbolMapper | SymbolMapper |
| HedgePosition records | In-memory | Per-account | Created on failover, cleared on failback | HedgeManager | BrokerHealthMonitor, Monitoring |
| Failover rate limiters | In-memory | Per-account | Created on failover, destroyed on failback | BrokerHealthMonitor | OMS |
| Broker state mirror | Redis `BROKER:{account_id}:state` | Per-account | Updated on every state change | BrokerHealthMonitor | Monitoring, Grafana |
| Symbol mapping stats | Redis `SYMBOLS:build_status` | Shared | Updated at 08:30 | SymbolMapper | Monitoring |

**Per-account vs shared:**

| Category | Per-Account | Shared |
|----------|------------|--------|
| Broker clients | DhanClient, UpstoxClient | -- |
| Auth tokens | Yes (separate credentials) | -- |
| Failover state | Yes (independent failover) | -- |
| WS connections | Yes (one per account) | -- |
| Rate limiters | Yes (10 OPS each) | -- |
| Symbol mapping | -- | Yes (one mapper, all accounts read) |
| Instrument CSVs | -- | Yes (downloaded once) |

---

### Failure Modes

| # | Failure | Detection | Impact | Recovery |
|---|---------|-----------|--------|----------|
| 1 | **Dhan auth failure at startup** | HTTP 401 during `_auth_single_account` | Account cannot trade. Other accounts unaffected. | SUSPEND account. Telegram CRITICAL. Operator provides fresh API key and restarts. |
| 2 | **Dhan token expires mid-session** | HTTP 401 on any REST call during trading hours | All operations for that account fail. Server-side SLs remain active. | `refresh_token()` attempt. If refresh fails: SUSPEND account, Telegram CRITICAL. |
| 3 | **Dhan API timeout (single request)** | `asyncio.TimeoutError` after 10s | That specific operation delayed. Retried automatically. | Exponential backoff retry (0.5s, 1s, 2s). Max 3 retries. |
| 4 | **Dhan API 5xx (server error)** | HTTP 5xx response | That specific operation fails. Retried automatically. | Same retry logic. If 3 retries fail: increment `consecutive_failures`. |
| 5 | **Dhan WS disconnect (one account)** | `ConnectionClosed` exception in WS reader | That account's fill management loops lose real-time updates. | Reconnect with backoff. If disconnect > 30s: trigger FAILED state. |
| 6 | **Dhan WS disconnect (all accounts)** | All `WSHealthMonitor.ws_connected = False` | All fill management loops switch to REST polling. OPS pressure increases. | Individual account reconnects. If all down > 60s: Telegram CRITICAL, suppress new entries. |
| 7 | **Broker maintenance window** | Dhan announces scheduled downtime or returns 503 | All API calls fail during window. | If pre-announced: suppress new entries for the window. If unexpected: treated as consecutive failures, may trigger failover. |
| 8 | **Cross-broker symbol mismatch** | `SymbolMapper.dhan_to_upstox()` returns None during failover hedge | Cannot place hedge on Upstox for that instrument. | Telegram CRITICAL with the specific instrument. Position protected only by Dhan server-side SL. Operator must manually hedge or accept the risk. |
| 9 | **Hedge order rejected on backup broker** | `OrderResponse.status == "REJECTED"` from Upstox | Position on primary broker is unhedged during failover. | Telegram CRITICAL. Try alternative hedge (futures instead of options). If all fail: operator intervention. |
| 10 | **Partial failover (some instruments hedge, others fail)** | Hedge loop tracks success/failure per position | Some positions hedged, others exposed. | Each position is independent. Hedged positions are safe. Unhedged positions rely on primary broker SLs. Log the specific unhedged instruments. |
| 11 | **Upstox auth fails during failover** | Exception in `_initiate_failover` when creating UpstoxClient | Cannot establish backup broker connection. Failover aborted. | Positions protected by Dhan server-side SLs only. Telegram CRITICAL. No automatic retry for Upstox auth. |
| 12 | **Failback causes double position** | Hedge unwind fails, primary broker recovers, orders resume | Account has positions on both brokers simultaneously. | Unwind verification: after failback, query both brokers for positions. If residual on backup: force-close with MARKET IOC. |
| 13 | **Symbol CSV download fails at 08:30** | HTTP error or timeout during `_download_dhan_csv` | System cannot start — Dhan CSV is mandatory for all operations. | Retry 3 times with 10s backoff. If Dhan CSV fails after retries: abort startup. If only Upstox CSV fails: start with degraded failover capability. |
| 14 | **Token stored in Redis but client not initialized** | Client `_token` is None despite Redis having the token | Process restart reads Redis token but does not restore client state. | Full re-auth on startup — never rely on stale Redis tokens. |
| 15 | **Rate limit server-side (HTTP 429)** | Dhan returns HTTP 429 | OPS limit exceeded despite client-side rate limiter. | Back off 1s. Log ERROR. Investigate why client-side limiter allowed it (clock skew, burst). |

---

### Edge Cases

#### 1. Failover triggers during an active fill management loop

Account A's Dhan fails while the OMS is in the middle of managing a fill (waiting for PART_TRADED to complete). The fill management loop is using the Dhan adapter.

**Handling:** The fill management loop detects the broker switch on its next API call (which will fail with the Dhan client's error). It checks `broker_router.get_state(account_id).health`. If FAILOVER_ACTIVE: the loop cancels the pending order on Dhan (best-effort, may fail), and the position is hedged by the HedgeManager. The fill loop does NOT re-place the order on Upstox — that would require re-entering the fill management lifecycle from the OMS, which is not within the fill loop's scope.

#### 2. Two accounts failover simultaneously

Account A and Account B both hit 3 consecutive Dhan failures at the same time (plausible if Dhan has a system-wide outage).

**Handling:** Each account's failover is independent. Both create their own UpstoxClient, authenticate separately, and hedge independently. Their Upstox rate limits are also independent (50 OPS each). No cross-account coordination is needed.

#### 3. Symbol mapping stale after market-hours contract rollover

A new weekly expiry contract is listed after the 08:30 symbol build. A strategy signals a trade on the new contract at 09:30.

**Handling:** The SymbolMapper build runs once at 08:30 and is immutable for the session. NSE lists new weekly options before 08:30, so this case is unlikely. If it occurs: the Instrument Resolver (Component 2) fetches the contract from the Dhan on-demand chain API (which uses Dhan security_id natively). The Upstox mapping will be missing, meaning failover for that specific instrument is degraded. Telegram WARNING issued.

#### 4. Failback attempt while hedge unwind order is in-flight

The primary broker recovers and failback begins. The hedge unwind order on Upstox is placed but not yet filled. Meanwhile, the system switches active broker back to Dhan.

**Handling:** The failback procedure waits for hedge unwind confirmation before switching the active broker. The `unwind_hedges` method awaits each close order's response. If a close order hangs (no response within 30s): force-close with IOC, log WARNING.

#### 5. Dhan returns 200 for order placement but the order is phantom

During degraded conditions, Dhan returns a success response with an `orderId`, but the order never appears in the order book. This is the same edge case documented in OMS Edge Case 10.

**Handling:** The fill management loop's TRANSIT timeout (10s) catches this. The broker router does not add additional detection — it delegates to the OMS's existing handling.

#### 6. Upstox instrument_key format changes

Upstox changes their CSV format or instrument_key structure between sessions.

**Handling:** The SymbolMapper's `_download_upstox_csv` parses column headers dynamically. If expected columns are missing: the build logs an error for Upstox and continues with Dhan-only mode. The system operates normally on the primary broker, with degraded failover. Operator alerted to update the Upstox CSV parser.

#### 7. Account authenticated on Dhan and Upstox simultaneously during failover

During failover, the account has active sessions on both brokers: primary (Dhan) for existing server-side SLs, backup (Upstox) for new hedge positions.

**Handling:** This is the expected state during failover. The DhanClient remains initialized (its SLs are still active on Dhan's servers). The UpstoxClient is created for hedge placement. Both sessions are valid. The `active_broker` field determines which broker receives new orders — Upstox during failover. Dhan SLs continue to protect existing positions independently of the active broker selection.

#### 8. Redis down during auth — tokens cannot be stored

Redis is unavailable at 08:25 when authentication runs. Tokens are obtained from Dhan but cannot be persisted.

**Handling:** Tokens are set on the in-memory DhanClient regardless of Redis. Redis is used for cross-process sharing and monitoring visibility, not as the primary token store. The DhanClient operates normally without Redis. A warning is logged and the OMS enters degraded mode (as described in OMS Component 5). When Redis recovers, the auth manager writes the tokens on the next health check cycle.

#### 9. Failover races with EOD flatten

At 15:20, the EOD flatten sequence begins for all accounts. At 15:21, Account A's Dhan connection fails and failover triggers.

**Handling:** EOD flatten has priority. If failover triggers during EOD flatten, the hedging step is skipped — the positions are being closed anyway. The `HedgeManager.hedge_account` checks with the OMS whether EOD flatten is in progress. If it is: log a warning and skip hedging. The flatten sequence on Dhan will either complete (using the already-placed cancel/exit orders) or fail (in which case the server-side SLs catch it at session close).

#### 10. Upstox HFT endpoint vs standard endpoint selection

Upstox offers `api-hft.upstox.com` for lower-latency order placement. During failover, should the system use the HFT endpoint?

**Handling:** v1 uses the standard `api.upstox.com` endpoint for failover. The HFT endpoint requires separate approval and has different rate limits. Since failover is an emergency path (not the primary execution path), the standard endpoint's latency is acceptable. The `UpstoxClient.BASE_URL` is configurable to switch to HFT if approved.

---


## Component 7: Risk Management Layer

### Responsibility

- Pre-trade validation gate: all checks must pass before order placement, evaluated per-account
- Post-trade continuous monitoring: drawdown, margin, VIX regime, evaluated per-account
- Kill conditions per strategy (shared across all accounts) and portfolio halt (per-account)
- Broker position check per-account to prevent ghost positions (Finding 8)
- Aggregate exposure monitoring across all accounts (informational, for regulatory awareness)
- Reconciliation of broker-reported vs locally-tracked positions per-account

### Interface

```python
from decimal import Decimal

class RiskCheckResult(pydantic.BaseModel):
    """Result of a pre-trade risk gate evaluation for one account."""
    passed: bool
    account_id: str
    checks_run: int
    checks_passed: int
    rejection_reason: str | None = None
    rejected_by: str | None = None          # which check failed first
    ghost_positions: list[str] | None = None # instrument_ids of ghosts found
    elapsed_ms: int = 0

class AggregateExposureReport(pydantic.BaseModel):
    """Aggregate exposure across all accounts. Informational only."""
    total_notional_inr: float
    total_margin_used_inr: float
    total_unrealized_pnl_inr: float
    account_count: int
    per_account: dict[str, float]           # account_id → notional
    warning: bool = False                   # True if aggregate > N × single limit
    warning_message: str | None = None

class KillConditionState(pydantic.BaseModel):
    """Tracks the current state of a strategy kill condition."""
    strategy_id: str
    metric_name: str
    current_value: float
    threshold: float
    triggered: bool = False
    triggered_at: datetime | None = None
    scope: Literal["shared", "per_account"]
    action: Literal["stop_new_entries", "flatten_immediately", "halt_all"]

class AccountRiskState(pydantic.BaseModel):
    """Per-account risk state tracked in Redis and memory."""
    account_id: str
    daily_trade_count: int = 0
    daily_order_count: int = 0
    current_drawdown_pct: float = 0.0
    peak_equity_inr: Decimal
    current_equity_inr: Decimal
    margin_used_inr: Decimal = Decimal("0")
    margin_available_inr: Decimal
    halted: bool = False
    halted_reason: str | None = None
    last_ghost_check_ts: int = 0
    last_reconciliation_ts: int = 0

class SharedRiskState(pydantic.BaseModel):
    """Shared risk state — same across all accounts."""
    killed_strategies: set[str] = set()     # strategy_ids currently killed
    vix_current: float = 0.0
    vix_regime: Literal["NORMAL", "ELEVATED", "EXTREME"] = "NORMAL"
    s1_active: bool = False                 # for S1↔S3 correlation check
    global_halted: bool = False
```

---

### Pre-Trade Checks

Every order passes through 11 pre-trade checks before placement. The checks run per-account (each account has its own capital, margin, position limits), except for the cost hurdle check which is shared (same cost structure regardless of account) and the correlation check which reads shared signal state.

#### Pre-Trade Check Table

| # | Check | Rule | Per-Account or Shared | Blocking | OPS Cost |
|---|-------|------|-----------------------|----------|----------|
| 1 | Position size | `qty * price <= account.allocation * 1.2` | Per-Account | Yes | 0 |
| 2 | Daily trade count | `account.daily_trade_count < 50` per strategy, `< 200` portfolio | Per-Account | Yes | 0 |
| 3 | Margin | `required_margin <= account.margin_available` | Per-Account | Yes | 1 (REST) |
| 4 | Cost hurdle | `expected_edge_bps > 2 * cost_bps` | Shared | Yes | 0 |
| 5 | Correlation (S1/S3) | If S1 has open position, block S3 entry (and vice versa) | Shared (reads shared state) | Yes | 0 |
| 6 | Strategy not killed | `KILLED:{strategy_id}` not set in Redis | Shared | Yes | 0 |
| 7 | Portfolio not halted | `HALT:global` not set AND `HALT:{account_id}` not set | Per-Account + Shared | Yes | 0 |
| 8 | Session hours | `09:15 <= IST_now <= 15:25` | Shared | Yes | 0 |
| 9 | Daily order limit | `account.daily_order_count < 4500` (buffer below Dhan's 5000) | Per-Account | Yes | 0 |
| 10 | Ghost position check | `GET /v2/positions` for this account, compare to local state | Per-Account | Yes | 1 (REST) |
| 11 | VIX regime check | If VIX > 25, suppress S1 and S5 | Shared | Yes | 0 |

**Total OPS cost per pre-trade gate:** 2 (margin check + ghost check) per account. With N accounts processing the same signal concurrently, this is 2N OPS total, but spread across N independent rate limiters (each account has its own 10 OPS budget), so the effective cost per rate limiter is 2 OPS.

#### Check 1: Position Size

Validates that the notional value of the order does not exceed 120% of the account's allocated capital for this strategy. The 20% buffer accommodates intraday price movement between signal generation and order placement.

```python
def check_position_size(
    self,
    order: "ResolvedOrder",
    account: "Account",
    allocation: "AllocationResponse",
) -> tuple[bool, str | None]:
    """
    Verify order notional does not exceed 120% of allocated capital.

    Args:
        order: The resolved order with quantity and limit price.
        account: The account placing the order.
        allocation: The capital allocation for this strategy on this account.

    Returns:
        (passed, rejection_reason) tuple.
    """
    notional = order.quantity * order.limit_price
    max_notional = allocation.allocated_capital * 1.2

    if notional > max_notional:
        return False, (
            f"position_size_exceeded: notional={notional:.0f} > "
            f"max={max_notional:.0f} (120% of {allocation.allocated_capital:.0f}) "
            f"account={account.account_id}"
        )
    return True, None
```

#### Check 2: Daily Trade Count

Each account maintains its own daily trade count in Redis. Strategies are capped at 50 trades/day each, with a portfolio-wide cap of 200 trades/day per account.

```python
async def check_daily_trade_count(
    self,
    strategy_id: str,
    account: "Account",
) -> tuple[bool, str | None]:
    """
    Check that neither the per-strategy nor portfolio trade count
    is exceeded for this account.

    Args:
        strategy_id: The strategy generating the signal.
        account: The account to check.

    Returns:
        (passed, rejection_reason) tuple.
    """
    strategy_key = f"RISK:trade_count:{account.account_id}:{strategy_id}"
    portfolio_key = f"RISK:trade_count:{account.account_id}:portfolio"

    strategy_count = int(await self._redis.get(strategy_key) or 0)
    portfolio_count = int(await self._redis.get(portfolio_key) or 0)

    if strategy_count >= 50:
        return False, (
            f"strategy_trade_limit: {strategy_id} has {strategy_count} trades "
            f"today on account {account.account_id} (limit: 50)"
        )

    if portfolio_count >= 200:
        return False, (
            f"portfolio_trade_limit: account {account.account_id} has "
            f"{portfolio_count} trades today (limit: 200)"
        )

    return True, None
```

#### Check 3: Margin

Queries the broker margin API for this specific account. Each Dhan account has independent margin.

```python
async def check_margin(
    self,
    order: "ResolvedOrder",
    account: "Account",
) -> tuple[bool, str | None]:
    """
    Verify this account has sufficient margin for the order.

    Calls GET /v2/fundlimit for this account's API key.
    Costs 1 OPS on this account's rate limiter.

    Args:
        order: The resolved order to check margin for.
        account: The account whose margin to check.

    Returns:
        (passed, rejection_reason) tuple.
    """
    rate_limiter = self._account_rate_limiters[account.account_id]
    await rate_limiter.acquire("RISK")

    try:
        funds = await self._broker.get_fund_limits(account)
    except BrokerAPIError as e:
        # Margin check failure is a HARD BLOCK — cannot proceed without knowing margin
        return False, f"margin_api_failed: {e} account={account.account_id}"

    required = self._estimate_margin(order)
    available = funds.available_margin

    # Update cached state
    self._account_states[account.account_id].margin_used_inr = funds.utilized_margin
    self._account_states[account.account_id].margin_available_inr = available

    if required > available:
        return False, (
            f"insufficient_margin: required={required:.0f} > "
            f"available={available:.0f} account={account.account_id}"
        )

    return True, None
```

#### Check 4: Cost Hurdle (Shared)

The cost hurdle check is shared because transaction costs on NSE are the same regardless of which Dhan account places the order. The full NSE cost structure includes brokerage, STT, exchange fees, SEBI turnover fee, GST, and stamp duty. The expected edge must exceed 2x the total cost to be worth trading.

```python
def check_cost_hurdle(
    self,
    order: "ResolvedOrder",
    signal: "StrategySignal",
) -> tuple[bool, str | None]:
    """
    Verify that the expected edge exceeds 2x transaction costs.

    This check is SHARED — cost structure is identical across accounts.
    Same instrument, same exchange, same cost model.

    NSE cost components (per-leg, for options):
        - Brokerage: flat per order (Dhan: Rs 20 or 0.03% whichever is lower)
        - STT: 0.0625% of premium (buy-side for options, sell-side for futures)
        - Exchange txn fee: 0.0495% (NSE FnO)
        - SEBI turnover fee: 0.0001%
        - GST: 18% on (brokerage + exchange txn fee + SEBI fee)
        - Stamp duty: 0.003% (buy-side only, state-dependent)

    Args:
        order: The resolved order with instrument details.
        signal: The strategy signal with expected edge.

    Returns:
        (passed, rejection_reason) tuple.
    """
    cost_bps = self._cost_model.compute_roundtrip_bps(
        instrument_type=order.instrument_type,
        premium=order.limit_price,
        quantity=order.quantity,
        lot_size=order.lot_size,
    )
    expected_edge_bps = signal.expected_edge_bps

    if expected_edge_bps <= 2 * cost_bps:
        return False, (
            f"cost_hurdle_failed: edge={expected_edge_bps:.1f}bps <= "
            f"2x cost={2 * cost_bps:.1f}bps"
        )

    return True, None
```

#### Check 5: S1/S3 Correlation Block

S1 (momentum) and S3 (mean-reversion) are anti-correlated by construction. Running both simultaneously on the same underlying creates a hedged-but-costly position. If S1 holds an active position, S3 entries are blocked, and vice versa.

```python
async def check_correlation(
    self,
    strategy_id: str,
) -> tuple[bool, str | None]:
    """
    Block simultaneous S1 and S3 positions.

    Reads SHARED state — the correlation block applies regardless of which
    account the position is on. If Account A has an S1 position, Account B
    cannot enter S3 either (same underlying, same signal logic).

    Args:
        strategy_id: The strategy requesting entry.

    Returns:
        (passed, rejection_reason) tuple.
    """
    if strategy_id == "S1":
        s3_active = await self._redis.get("POSITION:strategy:S3")
        if s3_active:
            return False, "correlation_block: S3 active, blocking S1 entry"

    elif strategy_id == "S3":
        s1_active = await self._redis.get("POSITION:strategy:S1")
        if s1_active:
            return False, "correlation_block: S1 active, blocking S3 entry"

    return True, None
```

#### Check 6: Strategy Not Killed

```python
async def check_strategy_alive(
    self,
    strategy_id: str,
) -> tuple[bool, str | None]:
    """
    Verify the strategy has not been killed by a kill condition.

    SHARED check — if S1 is killed, it is killed for ALL accounts.

    Args:
        strategy_id: The strategy to check.

    Returns:
        (passed, rejection_reason) tuple.
    """
    killed = await self._redis.get(f"KILLED:{strategy_id}")
    if killed:
        return False, f"strategy_killed: {strategy_id} killed at {killed}"

    return True, None
```

#### Check 7: Portfolio Not Halted

Checks both the global halt (shared) and the per-account halt. An account can be halted independently (e.g., Account B hit its own drawdown limit) while other accounts continue trading.

```python
async def check_not_halted(
    self,
    account: "Account",
) -> tuple[bool, str | None]:
    """
    Verify neither the global system nor this specific account is halted.

    Global halt (HALT:global): SHARED — all accounts stop.
    Account halt (HALT:{account_id}): PER-ACCOUNT — only this account stops.

    Args:
        account: The account to check.

    Returns:
        (passed, rejection_reason) tuple.
    """
    global_halt = await self._redis.get("HALT:global")
    if global_halt:
        return False, "global_halt_active"

    account_halt = await self._redis.get(f"HALT:{account.account_id}")
    if account_halt:
        return False, f"account_halted: {account.account_id} reason={account_halt}"

    return True, None
```

#### Check 8: Session Hours

```python
def check_session_hours(self) -> tuple[bool, str | None]:
    """
    Verify we are within NSE trading session hours.

    SHARED check — market hours are the same for all accounts.
    Window: 09:15 IST to 15:25 IST (last new entry 5 min before close).

    Returns:
        (passed, rejection_reason) tuple.
    """
    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    last_entry = now_ist.replace(hour=15, minute=25, second=0, microsecond=0)

    if now_ist < market_open:
        return False, f"pre_market: {now_ist.strftime('%H:%M:%S')} < 09:15"

    if now_ist > last_entry:
        return False, f"post_cutoff: {now_ist.strftime('%H:%M:%S')} > 15:25"

    return True, None
```

#### Check 9: Daily Order Limit

Each Dhan account has a 5,000 order/day hard limit. We set a buffer at 4,500 to leave room for EOD flatten orders and SL modifications.

```python
async def check_daily_order_limit(
    self,
    account: "Account",
) -> tuple[bool, str | None]:
    """
    Verify this account has not exceeded its daily order limit.

    PER-ACCOUNT check — each Dhan API key has its own 5,000 limit.
    We block at 4,500 to leave headroom for EOD flatten + SL activity.

    Args:
        account: The account to check.

    Returns:
        (passed, rejection_reason) tuple.
    """
    key = f"OMS:daily_count:{account.account_id}"
    count = int(await self._redis.get(key) or 0)

    if count >= 4500:
        return False, (
            f"daily_order_limit: account {account.account_id} "
            f"has {count} orders today (limit: 4500)"
        )

    return True, None
```

#### Check 10: Ghost Position Check (Finding 8)

This is the most critical per-account check. Before placing ANY order, the system queries `GET /v2/positions` on the specific Dhan account to verify no ghost positions exist for the instrument being traded. A ghost position is a position the broker knows about but the local system does not, typically caused by a missed fill notification (WS disconnect, message corruption, process restart).

**Why this must be per-account:** Each Dhan account has its own independent position book. Account A may have a ghost NIFTY CE position while Account B is clean. The ghost check queries each account's API key separately. There is no cross-account position endpoint on Dhan.

**Why this is worth 1 OPS per trade:** The alternative is a double-position catastrophe where the system enters a second position because it never recorded the first fill. For a 10-lot NIFTY option position, a double entry is roughly Rs 3L of unintended exposure. The 1 OPS cost (out of 10 OPS budget) is trivially justified.

```python
async def check_ghost_positions(
    self,
    order: "ResolvedOrder",
    account: "Account",
) -> tuple[bool, str | None]:
    """
    Query broker positions API for this account and compare to local state.

    If the broker shows a position for this instrument_id that local state
    does not track, this is a ghost position. Block the order and trigger
    reconciliation.

    PER-ACCOUNT: each account has its own position book on the broker.
    Costs 1 OPS on this account's rate limiter.

    Args:
        order: The order about to be placed (need instrument_id).
        account: The account to query.

    Returns:
        (passed, rejection_reason) tuple.
    """
    rate_limiter = self._account_rate_limiters[account.account_id]
    await rate_limiter.acquire("RISK")

    try:
        broker_positions = await self._broker.get_positions(account)
    except BrokerAPIError as e:
        # Ghost check failure is a HARD BLOCK
        return False, (
            f"ghost_check_api_failed: {e} account={account.account_id}. "
            f"Cannot verify position state — blocking order."
        )

    # Build set of instrument_ids the broker reports for this account
    broker_instrument_ids = {
        p.security_id
        for p in broker_positions
        if p.net_qty != 0  # only non-zero positions
    }

    # Build set of instrument_ids we track locally for this account
    local_instrument_ids = {
        pos.instrument_id
        for pos in self._position_tracker.get_account_positions(account.account_id)
        if pos.quantity != 0
    }

    # Ghost = broker has it, we don't
    ghosts = broker_instrument_ids - local_instrument_ids

    if ghosts:
        # Trigger async reconciliation (non-blocking)
        asyncio.create_task(
            self._position_tracker.force_sync_from_broker(account.account_id),
            name=f"ghost_recon_{account.account_id}",
        )

        await self._telegram.send(
            "CRITICAL",
            f"GHOST POSITION detected on {account.account_id}: "
            f"instruments={ghosts}. Reconciliation triggered. "
            f"Order for {order.instrument_id} BLOCKED.",
        )

        return False, (
            f"ghost_position_detected: broker has positions for "
            f"{ghosts} not tracked locally on account {account.account_id}"
        )

    # Reverse ghost = we track it, broker doesn't
    reverse_ghosts = local_instrument_ids - broker_instrument_ids
    if reverse_ghosts:
        logger.warning(
            "reverse_ghost_detected",
            account_id=account.account_id,
            instruments=reverse_ghosts,
        )
        # Reverse ghosts don't block the current order, but trigger recon
        asyncio.create_task(
            self._position_tracker.force_sync_from_broker(account.account_id),
            name=f"reverse_ghost_recon_{account.account_id}",
        )

    # Update last check timestamp
    self._account_states[account.account_id].last_ghost_check_ts = int(
        time.time() * 1000
    )

    return True, None
```

#### Check 11: VIX Regime Check

INDIA VIX above 25 suppresses S1 (momentum) and S5 (expiry-day) strategies. These strategies have poor risk-adjusted returns in high-volatility regimes based on backtesting.

```python
def check_vix_regime(
    self,
    strategy_id: str,
) -> tuple[bool, str | None]:
    """
    Suppress certain strategies during high-VIX regimes.

    SHARED check — VIX is the same for all accounts. Same market.

    Regime thresholds:
        VIX <= 20: NORMAL — all strategies active
        20 < VIX <= 25: ELEVATED — warning, all strategies active
        VIX > 25: EXTREME — S1 and S5 suppressed

    Args:
        strategy_id: The strategy requesting entry.

    Returns:
        (passed, rejection_reason) tuple.
    """
    vix = self._shared_state.vix_current

    if vix > 25 and strategy_id in ("S1", "S5"):
        return False, (
            f"vix_regime_block: VIX={vix:.1f} > 25, "
            f"strategy {strategy_id} suppressed in EXTREME regime"
        )

    return True, None
```

---

### Pre-Trade Check Implementation

The risk gate function runs all 11 checks sequentially for a given account. Checks are ordered by cost (free checks first, OPS-consuming checks last) to fail fast without spending API calls.

```python
class RiskManager:
    """
    Central risk management layer.

    Runs pre-trade checks per-account, monitors post-trade risk
    metrics per-account, and manages kill conditions (shared).
    """

    def __init__(
        self,
        accounts: list["Account"],
        redis: "Redis",
        broker: "BrokerAdapter",
        position_tracker: "PositionTracker",
        cost_model: "CostModel",
        telegram: "TelegramNotifier",
        config: "RiskConfig",
    ):
        self._accounts = {a.account_id: a for a in accounts}
        self._redis = redis
        self._broker = broker
        self._position_tracker = position_tracker
        self._cost_model = cost_model
        self._telegram = telegram
        self._config = config

        # Per-account rate limiters (shared with OMS — same reference)
        self._account_rate_limiters: dict[str, "PriorityRateLimiter"] = {}

        # Per-account risk state
        self._account_states: dict[str, AccountRiskState] = {}
        for a in accounts:
            self._account_states[a.account_id] = AccountRiskState(
                account_id=a.account_id,
                peak_equity_inr=a.capital_inr,
                current_equity_inr=a.capital_inr,
                margin_available_inr=a.capital_inr,
            )

        # Shared state
        self._shared_state = SharedRiskState()

        # Lock for shared state writes (kill conditions, VIX update)
        self._shared_lock = asyncio.Lock()

        # Per-account locks for account state writes
        self._account_locks: dict[str, asyncio.Lock] = {
            a.account_id: asyncio.Lock() for a in accounts
        }

    def set_rate_limiters(
        self,
        limiters: dict[str, "PriorityRateLimiter"],
    ) -> None:
        """
        Inject per-account rate limiters (shared with OMS).

        Called during system initialization. The OMS and RiskManager share
        the same rate limiter instances so OPS budget is correctly tracked.

        Args:
            limiters: Mapping of account_id to PriorityRateLimiter.
        """
        self._account_rate_limiters = limiters

    async def pre_trade_check(
        self,
        order: "ResolvedOrder",
        account: "Account",
        signal: "StrategySignal | None" = None,
        allocation: "AllocationResponse | None" = None,
    ) -> RiskCheckResult:
        """
        Run all 11 pre-trade checks for a specific account.

        Checks are ordered by cost: free checks first, then checks
        that consume OPS (margin, ghost check). This minimizes API
        calls for signals that will be rejected on cheaper checks.

        Args:
            order: The resolved order to validate.
            account: The account placing the order.
            signal: The original strategy signal (for cost hurdle check).
            allocation: Capital allocation response (for position size check).

        Returns:
            RiskCheckResult with pass/fail and rejection details.
        """
        t0 = time.monotonic()
        checks_passed = 0
        total_checks = 11

        # --- Free checks first (0 OPS) ---

        # 1. Session hours (shared)
        ok, reason = self.check_session_hours()
        if not ok:
            return RiskCheckResult(
                passed=False, account_id=account.account_id,
                checks_run=1, checks_passed=0,
                rejection_reason=reason, rejected_by="session_hours",
                elapsed_ms=_elapsed(t0),
            )
        checks_passed += 1

        # 2. Portfolio/account not halted (per-account + shared)
        ok, reason = await self.check_not_halted(account)
        if not ok:
            return RiskCheckResult(
                passed=False, account_id=account.account_id,
                checks_run=2, checks_passed=checks_passed,
                rejection_reason=reason, rejected_by="halt_check",
                elapsed_ms=_elapsed(t0),
            )
        checks_passed += 1

        # 3. Strategy not killed (shared)
        ok, reason = await self.check_strategy_alive(order.strategy_id)
        if not ok:
            return RiskCheckResult(
                passed=False, account_id=account.account_id,
                checks_run=3, checks_passed=checks_passed,
                rejection_reason=reason, rejected_by="strategy_killed",
                elapsed_ms=_elapsed(t0),
            )
        checks_passed += 1

        # 4. VIX regime (shared)
        ok, reason = self.check_vix_regime(order.strategy_id)
        if not ok:
            return RiskCheckResult(
                passed=False, account_id=account.account_id,
                checks_run=4, checks_passed=checks_passed,
                rejection_reason=reason, rejected_by="vix_regime",
                elapsed_ms=_elapsed(t0),
            )
        checks_passed += 1

        # 5. Correlation (shared)
        ok, reason = await self.check_correlation(order.strategy_id)
        if not ok:
            return RiskCheckResult(
                passed=False, account_id=account.account_id,
                checks_run=5, checks_passed=checks_passed,
                rejection_reason=reason, rejected_by="correlation",
                elapsed_ms=_elapsed(t0),
            )
        checks_passed += 1

        # 6. Daily trade count (per-account)
        ok, reason = await self.check_daily_trade_count(
            order.strategy_id, account,
        )
        if not ok:
            return RiskCheckResult(
                passed=False, account_id=account.account_id,
                checks_run=6, checks_passed=checks_passed,
                rejection_reason=reason, rejected_by="daily_trade_count",
                elapsed_ms=_elapsed(t0),
            )
        checks_passed += 1

        # 7. Daily order limit (per-account)
        ok, reason = await self.check_daily_order_limit(account)
        if not ok:
            return RiskCheckResult(
                passed=False, account_id=account.account_id,
                checks_run=7, checks_passed=checks_passed,
                rejection_reason=reason, rejected_by="daily_order_limit",
                elapsed_ms=_elapsed(t0),
            )
        checks_passed += 1

        # 8. Position size (per-account)
        if allocation is not None:
            ok, reason = self.check_position_size(order, account, allocation)
            if not ok:
                return RiskCheckResult(
                    passed=False, account_id=account.account_id,
                    checks_run=8, checks_passed=checks_passed,
                    rejection_reason=reason, rejected_by="position_size",
                    elapsed_ms=_elapsed(t0),
                )
        checks_passed += 1

        # 9. Cost hurdle (shared)
        if signal is not None:
            ok, reason = self.check_cost_hurdle(order, signal)
            if not ok:
                return RiskCheckResult(
                    passed=False, account_id=account.account_id,
                    checks_run=9, checks_passed=checks_passed,
                    rejection_reason=reason, rejected_by="cost_hurdle",
                    elapsed_ms=_elapsed(t0),
                )
        checks_passed += 1

        # --- OPS-consuming checks (1 OPS each) ---

        # 10. Margin (per-account, 1 OPS)
        ok, reason = await self.check_margin(order, account)
        if not ok:
            return RiskCheckResult(
                passed=False, account_id=account.account_id,
                checks_run=10, checks_passed=checks_passed,
                rejection_reason=reason, rejected_by="margin",
                elapsed_ms=_elapsed(t0),
            )
        checks_passed += 1

        # 11. Ghost position check (per-account, 1 OPS)
        ok, reason = await self.check_ghost_positions(order, account)
        if not ok:
            ghost_ids = list(reason.split("instruments=")[1].split("}")[0]) \
                if "instruments=" in (reason or "") else None
            return RiskCheckResult(
                passed=False, account_id=account.account_id,
                checks_run=11, checks_passed=checks_passed,
                rejection_reason=reason, rejected_by="ghost_position",
                ghost_positions=ghost_ids,
                elapsed_ms=_elapsed(t0),
            )
        checks_passed += 1

        # All 11 checks passed
        return RiskCheckResult(
            passed=True, account_id=account.account_id,
            checks_run=total_checks, checks_passed=total_checks,
            elapsed_ms=_elapsed(t0),
        )


def _elapsed(t0: float) -> int:
    """Helper: milliseconds since t0."""
    return int((time.monotonic() - t0) * 1000)
```

---

### Aggregate Checks (Multi-Account)

When multiple accounts are active, the system computes aggregate exposure across all accounts. These checks are informational, not blocking, because each Dhan account is a legally separate entity with its own margin, position limits, and regulatory standing. Blocking one account based on another account's exposure would be incorrect.

```python
class AggregateMonitor:
    """
    Monitors aggregate exposure across all accounts.

    All checks here are informational — they produce warnings
    and log entries but do NOT block orders. Each account is
    a legally separate trading entity.
    """

    def __init__(
        self,
        accounts: list["Account"],
        position_tracker: "PositionTracker",
        telegram: "TelegramNotifier",
        config: "RiskConfig",
    ):
        self._accounts = {a.account_id: a for a in accounts}
        self._position_tracker = position_tracker
        self._telegram = telegram
        self._config = config

        # Single-account exposure limit (from config)
        self._single_account_limit = config.max_notional_per_account

    async def compute_aggregate_exposure(self) -> AggregateExposureReport:
        """
        Compute total notional exposure across all accounts.

        Returns:
            AggregateExposureReport with per-account breakdown.
        """
        per_account: dict[str, float] = {}
        total_notional = 0.0
        total_margin = 0.0
        total_pnl = 0.0

        for account_id in self._accounts:
            positions = self._position_tracker.get_account_positions(account_id)
            account_notional = sum(
                abs(p.quantity * p.current_price) for p in positions
            )
            per_account[account_id] = account_notional
            total_notional += account_notional

        n = len(self._accounts)
        aggregate_limit = n * self._single_account_limit
        warning = total_notional > aggregate_limit

        report = AggregateExposureReport(
            total_notional_inr=total_notional,
            total_margin_used_inr=total_margin,
            total_unrealized_pnl_inr=total_pnl,
            account_count=n,
            per_account=per_account,
            warning=warning,
            warning_message=(
                f"Aggregate exposure {total_notional:.0f} exceeds "
                f"{n} x {self._single_account_limit:.0f} = {aggregate_limit:.0f}"
            ) if warning else None,
        )

        if warning:
            await self._telegram.send(
                "WARNING",
                f"AGGREGATE EXPOSURE WARNING: total notional "
                f"Rs {total_notional / 100_000:.1f}L across {n} accounts. "
                f"Limit: Rs {aggregate_limit / 100_000:.1f}L.",
            )

        return report

    async def compute_aggregate_drawdown(self) -> dict[str, float]:
        """
        Compute aggregate drawdown across all accounts.

        This is purely informational — portfolio halt is per-account,
        not aggregate. The aggregate number is useful for the operator
        to understand total system health.

        Returns:
            Dict with 'total_drawdown_pct', 'total_pnl_inr',
            'total_peak_inr', 'total_current_inr'.
        """
        total_peak = 0.0
        total_current = 0.0

        for account_id in self._accounts:
            state = self._risk_manager._account_states.get(account_id)
            if state:
                total_peak += state.peak_equity_inr
                total_current += state.current_equity_inr

        if total_peak == 0:
            return {
                "total_drawdown_pct": 0.0,
                "total_pnl_inr": 0.0,
                "total_peak_inr": 0.0,
                "total_current_inr": 0.0,
            }

        drawdown_pct = (total_peak - total_current) / total_peak * 100

        return {
            "total_drawdown_pct": drawdown_pct,
            "total_pnl_inr": total_current - total_peak,
            "total_peak_inr": total_peak,
            "total_current_inr": total_current,
        }
```

---

### Kill Conditions

Kill conditions determine when a strategy should be stopped or the entire portfolio should be halted. Strategy-level kills are SHARED (if S1 Sharpe drops below 0.5, S1 is killed for ALL accounts simultaneously). Portfolio-level drawdown halt is PER-ACCOUNT (each account has its own equity curve and drawdown limit).

#### Kill Conditions Table

| Strategy | Metric | Threshold | Action | Scope | Evaluation Frequency |
|----------|--------|-----------|--------|-------|---------------------|
| S1 | After-cost Sharpe (60-day rolling) | < 0.5 | `stop_new_entries` | Shared | Every 10s |
| S2 | Consecutive negative PnL days | >= 40 | `stop_new_entries` | Shared | Daily EOD |
| S3 | After-cost Sharpe (60-day rolling) | < 0.3 | `stop_new_entries` | Shared | Every 10s |
| S4 | Underperformance vs NIFTY50 | > 15% over 12 months | `stop_new_entries` | Shared | Daily EOD |
| S5 | Consecutive expiry-day losses | >= 3 | `stop_new_entries` | Shared | After each expiry |
| S6 | Monthly drawdown of allocation | > -15% | `flatten_immediately` | Shared | Every 10s |
| S7 | After-cost Sharpe (6-month rolling) | < 0.3 | `stop_new_entries` | Shared | Every 10s |
| Per-Account | Account drawdown | > `account.max_drawdown_pct` (default -8%) | `halt_account` | Per-Account | Every 10s |
| Portfolio | Account portfolio drawdown | > -10% | `halt_all` (this account only) | Per-Account | Every 10s |

#### Action Definitions

- **`stop_new_entries`**: Set `KILLED:{strategy_id}` in Redis. All accounts stop entering new positions for this strategy. Open positions exit via their existing stop-losses or normal exit logic. Capital is redistributed at the next daily recomputation.

- **`flatten_immediately`**: Kill the strategy AND urgently exit all positions for that strategy across ALL accounts. Used for S6 (short options = unlimited risk). The OMS receives a flatten request for each account with `urgency="URGENT"`.

- **`halt_account`**: Set `HALT:{account_id}` in Redis. This specific account stops all new entries across all strategies. Other accounts are unaffected. Open positions on the halted account exit via stops. Manual intervention required to resume: `DEL HALT:{account_id}`.

- **`halt_all`**: For the specific account that breached the -10% portfolio drawdown, this triggers the account-level halt. It does NOT halt other accounts. The naming is historical from the single-account architecture; in multi-account mode, "halt_all" means "halt all strategies on THIS account."

#### Kill Condition Implementation

```python
class KillConditionEvaluator:
    """
    Evaluates kill conditions for all strategies.

    Strategy kills are SHARED — evaluated once, applied to all accounts.
    Account drawdown kills are PER-ACCOUNT — evaluated independently.
    """

    # Kill condition definitions
    KILL_CONDITIONS: list[dict] = [
        {
            "strategy_id": "S1",
            "metric": "after_cost_sharpe_60d",
            "threshold": 0.5,
            "comparator": "lt",
            "action": "stop_new_entries",
            "scope": "shared",
        },
        {
            "strategy_id": "S2",
            "metric": "consecutive_negative_pnl_days",
            "threshold": 40,
            "comparator": "gte",
            "action": "stop_new_entries",
            "scope": "shared",
        },
        {
            "strategy_id": "S3",
            "metric": "after_cost_sharpe_60d",
            "threshold": 0.3,
            "comparator": "lt",
            "action": "stop_new_entries",
            "scope": "shared",
        },
        {
            "strategy_id": "S4",
            "metric": "underperformance_vs_nifty50_12m",
            "threshold": 15.0,
            "comparator": "gt",
            "action": "stop_new_entries",
            "scope": "shared",
        },
        {
            "strategy_id": "S5",
            "metric": "consecutive_expiry_day_losses",
            "threshold": 3,
            "comparator": "gte",
            "action": "stop_new_entries",
            "scope": "shared",
        },
        {
            "strategy_id": "S6",
            "metric": "monthly_drawdown_pct",
            "threshold": -15.0,
            "comparator": "lt",
            "action": "flatten_immediately",
            "scope": "shared",
        },
        {
            "strategy_id": "S7",
            "metric": "after_cost_sharpe_6m",
            "threshold": 0.3,
            "comparator": "lt",
            "action": "stop_new_entries",
            "scope": "shared",
        },
    ]

    def __init__(
        self,
        redis: "Redis",
        position_tracker: "PositionTracker",
        oms: "OMS",
        telegram: "TelegramNotifier",
        accounts: list["Account"],
    ):
        self._redis = redis
        self._position_tracker = position_tracker
        self._oms = oms
        self._telegram = telegram
        self._accounts = {a.account_id: a for a in accounts}
        self._kill_states: dict[str, KillConditionState] = {}

    async def evaluate_shared_kills(
        self,
        metrics: dict[str, dict[str, float]],
    ) -> list[KillConditionState]:
        """
        Evaluate all shared (strategy-level) kill conditions.

        Called every 10 seconds by the post-trade monitor.

        Args:
            metrics: Mapping of strategy_id to metric_name to current value.
                     e.g., {"S1": {"after_cost_sharpe_60d": 0.42}}

        Returns:
            List of KillConditionState objects that were triggered this cycle.
        """
        newly_triggered: list[KillConditionState] = []

        for cond in self.KILL_CONDITIONS:
            sid = cond["strategy_id"]
            metric_name = cond["metric"]
            threshold = cond["threshold"]
            comparator = cond["comparator"]

            current_value = metrics.get(sid, {}).get(metric_name)
            if current_value is None:
                continue

            triggered = self._compare(current_value, threshold, comparator)
            state_key = f"{sid}:{metric_name}"

            if triggered and state_key not in self._kill_states:
                state = KillConditionState(
                    strategy_id=sid,
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold=threshold,
                    triggered=True,
                    triggered_at=datetime.now(ZoneInfo("Asia/Kolkata")),
                    scope="shared",
                    action=cond["action"],
                )
                self._kill_states[state_key] = state
                newly_triggered.append(state)

                # Execute the kill action
                await self._execute_kill(state)

        return newly_triggered

    async def evaluate_account_kills(
        self,
        account_states: dict[str, AccountRiskState],
        accounts: dict[str, "Account"],
    ) -> list[KillConditionState]:
        """
        Evaluate per-account drawdown kill conditions.

        Each account has its own max_drawdown_pct (default -8%).
        Portfolio halt (-10%) also applies per-account.

        Args:
            account_states: Current risk state per account.
            accounts: Account configurations (for max_drawdown_pct).

        Returns:
            List of per-account kills triggered this cycle.
        """
        newly_triggered: list[KillConditionState] = []

        for account_id, state in account_states.items():
            account = accounts.get(account_id)
            if account is None or state.halted:
                continue

            # Per-account drawdown check
            if state.current_drawdown_pct <= account.max_drawdown_pct:
                kill_state = KillConditionState(
                    strategy_id=f"ACCOUNT:{account_id}",
                    metric_name="account_drawdown_pct",
                    current_value=state.current_drawdown_pct,
                    threshold=account.max_drawdown_pct,
                    triggered=True,
                    triggered_at=datetime.now(ZoneInfo("Asia/Kolkata")),
                    scope="per_account",
                    action="halt_account",
                )
                newly_triggered.append(kill_state)
                await self._halt_account(account_id, f"drawdown {state.current_drawdown_pct:.1f}%")

            # Portfolio halt (-10%) per-account
            if state.current_drawdown_pct <= -10.0:
                kill_state = KillConditionState(
                    strategy_id=f"PORTFOLIO:{account_id}",
                    metric_name="portfolio_drawdown_pct",
                    current_value=state.current_drawdown_pct,
                    threshold=-10.0,
                    triggered=True,
                    triggered_at=datetime.now(ZoneInfo("Asia/Kolkata")),
                    scope="per_account",
                    action="halt_all",
                )
                newly_triggered.append(kill_state)
                await self._halt_account(account_id, f"portfolio drawdown {state.current_drawdown_pct:.1f}%")

        return newly_triggered

    async def _execute_kill(self, state: KillConditionState) -> None:
        """Execute a shared kill action."""
        sid = state.strategy_id

        if state.action == "stop_new_entries":
            await self._redis.set(
                f"KILLED:{sid}",
                datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            )
            await self._telegram.send(
                "CRITICAL",
                f"KILL: {sid} killed. {state.metric_name}="
                f"{state.current_value:.2f} (threshold: {state.threshold}). "
                f"No new entries on ANY account. Open positions exit via stops.",
            )
            logger.critical(
                "strategy_killed",
                strategy_id=sid,
                metric=state.metric_name,
                value=state.current_value,
                threshold=state.threshold,
            )

        elif state.action == "flatten_immediately":
            await self._redis.set(
                f"KILLED:{sid}",
                datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
            )
            # Flatten across ALL accounts
            for account_id, account in self._accounts.items():
                positions = self._position_tracker.get_strategy_positions(
                    sid, account_id,
                )
                for pos in positions:
                    exit_order = self._oms.build_exit_order(pos, urgency="URGENT")
                    await self._oms.place_exit(exit_order, account)

            await self._telegram.send(
                "CRITICAL",
                f"KILL + FLATTEN: {sid} killed and all positions being flattened "
                f"across ALL accounts. {state.metric_name}="
                f"{state.current_value:.2f} (threshold: {state.threshold}).",
            )

    async def _halt_account(self, account_id: str, reason: str) -> None:
        """Halt a specific account. Other accounts continue."""
        await self._redis.set(f"HALT:{account_id}", reason)
        self._risk_manager._account_states[account_id].halted = True
        self._risk_manager._account_states[account_id].halted_reason = reason

        await self._telegram.send(
            "CRITICAL",
            f"ACCOUNT HALT: {account_id} halted. Reason: {reason}. "
            f"Other accounts continue trading. "
            f"Manual resume: DEL HALT:{account_id}",
        )
        logger.critical(
            "account_halted",
            account_id=account_id,
            reason=reason,
        )

    @staticmethod
    def _compare(value: float, threshold: float, comparator: str) -> bool:
        """Compare value against threshold using the given comparator."""
        if comparator == "lt":
            return value < threshold
        elif comparator == "gt":
            return value > threshold
        elif comparator == "gte":
            return value >= threshold
        elif comparator == "lte":
            return value <= threshold
        return False
```

---

### Post-Order Reconciliation (Finding 8)

Reconciliation runs at three granularities, all per-account:

1. **Periodic (every 5 minutes):** `GET /v2/positions` for each account, compare to local state
2. **Post-order timer:** After every order placement, a timer fires at `max_patience_s + 5s`. If `on_fill()` was never called, force-sync from broker.
3. **Startup (09:16):** Full reconciliation at market open for each account

```python
class PerAccountReconciler:
    """
    Reconciles local position state with broker positions, per-account.

    Each account has independent reconciliation timers and state.
    Discrepancies are resolved by treating the broker as source of truth.
    """

    PERIODIC_INTERVAL_S = 300    # 5 minutes
    POST_ORDER_BUFFER_S = 5      # added to max_patience_s

    def __init__(
        self,
        accounts: list["Account"],
        broker: "BrokerAdapter",
        position_tracker: "PositionTracker",
        oms: "OMS",
        telegram: "TelegramNotifier",
        rate_limiters: dict[str, "PriorityRateLimiter"],
    ):
        self._accounts = {a.account_id: a for a in accounts}
        self._broker = broker
        self._position_tracker = position_tracker
        self._oms = oms
        self._telegram = telegram
        self._rate_limiters = rate_limiters

        # Per-account post-order timers: (account_id, order_id) → asyncio.Task
        self._post_order_timers: dict[tuple[str, str], asyncio.Task] = {}

    async def start_periodic(self) -> None:
        """
        Start the periodic reconciliation loop for all accounts.

        Runs every 5 minutes during market hours. Each account is
        reconciled independently (separate API call, separate rate limiter).
        """
        while True:
            await asyncio.sleep(self.PERIODIC_INTERVAL_S)

            now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
            if not (
                now_ist.hour >= 9
                and (now_ist.hour < 15 or (now_ist.hour == 15 and now_ist.minute <= 35))
            ):
                continue

            # Reconcile all accounts in parallel
            tasks = [
                asyncio.create_task(
                    self._reconcile_account(account_id),
                    name=f"periodic_recon_{account_id}",
                )
                for account_id in self._accounts
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for account_id, result in zip(self._accounts, results):
                if isinstance(result, Exception):
                    logger.error(
                        "periodic_reconciliation_failed",
                        account_id=account_id,
                        error=str(result),
                    )

    def start_post_order_timer(
        self,
        account_id: str,
        order_id: str,
        max_patience_s: float,
    ) -> None:
        """
        Start a post-order reconciliation timer for a specific account + order.

        If on_fill() is never called within (max_patience_s + 5s), the timer
        fires and force-syncs positions from the broker for this account.

        Args:
            account_id: The account that placed the order.
            order_id: The order ID to monitor.
            max_patience_s: The fill patience timeout from FillParams.
        """
        delay = max_patience_s + self.POST_ORDER_BUFFER_S
        key = (account_id, order_id)

        if key in self._post_order_timers:
            self._post_order_timers[key].cancel()

        task = asyncio.create_task(
            self._post_order_timer(account_id, order_id, delay),
            name=f"post_order_recon_{account_id}_{order_id[:8]}",
        )
        self._post_order_timers[key] = task

    def cancel_post_order_timer(
        self,
        account_id: str,
        order_id: str,
    ) -> None:
        """
        Cancel a post-order timer (called when on_fill arrives normally).

        Args:
            account_id: The account that placed the order.
            order_id: The order ID whose timer to cancel.
        """
        key = (account_id, order_id)
        timer = self._post_order_timers.pop(key, None)
        if timer and not timer.done():
            timer.cancel()

    async def _post_order_timer(
        self,
        account_id: str,
        order_id: str,
        delay: float,
    ) -> None:
        """Fire after delay if fill was not received."""
        await asyncio.sleep(delay)

        logger.warning(
            "post_order_timer_fired",
            account_id=account_id,
            order_id=order_id,
            delay_s=delay,
        )

        # Check broker for this specific order
        rate_limiter = self._rate_limiters[account_id]
        await rate_limiter.acquire("POLL")

        account = self._accounts[account_id]
        try:
            order_detail = await self._broker.get_order_status(account, order_id)
        except BrokerAPIError:
            logger.error(
                "post_order_check_failed",
                account_id=account_id,
                order_id=order_id,
            )
            return

        if order_detail.order_status == "TRADED" and order_detail.filled_qty > 0:
            # Fill was missed — force sync
            await self._telegram.send(
                "CRITICAL",
                f"MISSED FILL detected: order {order_id} on account "
                f"{account_id} shows TRADED at broker but no on_fill received. "
                f"Force syncing positions.",
            )
            await self.force_sync_from_broker(account_id)

    async def force_sync_from_broker(self, account_id: str) -> None:
        """
        Force-sync all positions for one account from the broker.

        Broker is source of truth. Local state is overwritten.

        Args:
            account_id: The account to sync.
        """
        rate_limiter = self._rate_limiters[account_id]
        await rate_limiter.acquire("RISK")

        account = self._accounts[account_id]
        try:
            broker_positions = await self._broker.get_positions(account)
        except BrokerAPIError as e:
            logger.error(
                "force_sync_failed",
                account_id=account_id,
                error=str(e),
            )
            await self._telegram.send(
                "CRITICAL",
                f"Force sync FAILED for {account_id}: {e}. "
                f"Manual intervention required.",
            )
            return

        discrepancies = await self._position_tracker.apply_broker_positions(
            account_id, broker_positions,
        )

        if discrepancies:
            await self._telegram.send(
                "CRITICAL",
                f"RECONCILIATION for {account_id}: {len(discrepancies)} "
                f"discrepancies resolved. Broker is source of truth. "
                f"Details: {[d.summary for d in discrepancies[:5]]}",
            )

        logger.info(
            "force_sync_complete",
            account_id=account_id,
            discrepancies=len(discrepancies),
        )

    async def _reconcile_account(self, account_id: str) -> int:
        """
        Quick reconciliation for one account.

        Compares instrument_id + quantity only. Returns number of discrepancies.

        Args:
            account_id: The account to reconcile.

        Returns:
            Number of discrepancies found.
        """
        rate_limiter = self._rate_limiters[account_id]
        await rate_limiter.acquire("RISK")

        account = self._accounts[account_id]
        broker_positions = await self._broker.get_positions(account)

        broker_map: dict[str, int] = {
            p.security_id: p.net_qty
            for p in broker_positions
            if p.net_qty != 0
        }

        local_positions = self._position_tracker.get_account_positions(account_id)
        local_map: dict[str, int] = {
            pos.instrument_id: pos.quantity
            for pos in local_positions
            if pos.quantity != 0
        }

        discrepancy_count = 0

        # Check for mismatches
        all_instruments = set(broker_map.keys()) | set(local_map.keys())
        for inst_id in all_instruments:
            broker_qty = broker_map.get(inst_id, 0)
            local_qty = local_map.get(inst_id, 0)
            if broker_qty != local_qty:
                discrepancy_count += 1
                logger.warning(
                    "position_discrepancy",
                    account_id=account_id,
                    instrument_id=inst_id,
                    broker_qty=broker_qty,
                    local_qty=local_qty,
                )

        if discrepancy_count > 0:
            await self.force_sync_from_broker(account_id)

        return discrepancy_count
```

#### Reconciliation Schedule Summary

| When | Type | Scope | OPS Cost | Action on Discrepancy |
|------|------|-------|----------|-----------------------|
| 09:16 IST | Full | Per-account | 1 per account | Force sync, Telegram CRITICAL |
| Every 5 min | Quick (instrument_id + qty) | Per-account | 1 per account | Force sync, Telegram CRITICAL |
| Post-order (patience + 5s) | Targeted (single order) | Per-account | 1 per account | Force sync if broker shows TRADED |
| 15:31 IST | Full (EOD) | Per-account | 1 per account | Force sync, Telegram CRITICAL |

---

### Post-Trade Monitor

An async background process that runs continuously during market hours. Every 10 seconds, it evaluates per-account drawdown and shared strategy metrics. This is the process that triggers kill conditions.

```python
class PostTradeMonitor:
    """
    Continuous post-trade monitoring process.

    Runs every 10 seconds during market hours. Evaluates:
    - Per-account drawdown and equity curve
    - Shared strategy metrics (Sharpe, consecutive losses, etc.)
    - VIX regime
    - Margin utilization per-account
    """

    MONITOR_INTERVAL_S = 10

    def __init__(
        self,
        risk_manager: "RiskManager",
        kill_evaluator: "KillConditionEvaluator",
        aggregate_monitor: "AggregateMonitor",
        position_tracker: "PositionTracker",
        accounts: list["Account"],
        redis: "Redis",
        telegram: "TelegramNotifier",
    ):
        self._risk = risk_manager
        self._kills = kill_evaluator
        self._aggregate = aggregate_monitor
        self._positions = position_tracker
        self._accounts = {a.account_id: a for a in accounts}
        self._redis = redis
        self._telegram = telegram

    async def run(self) -> None:
        """
        Main monitoring loop. Runs until cancelled.

        Every 10 seconds:
        1. Update per-account equity and drawdown
        2. Update shared strategy metrics
        3. Evaluate kill conditions
        4. Update VIX regime
        5. Update margin utilization
        6. Compute aggregate exposure (informational)
        7. Publish Prometheus metrics
        """
        while True:
            try:
                now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))

                # Only run during market hours (with some buffer)
                if not (
                    now_ist.hour >= 9
                    and (now_ist.hour < 15 or (now_ist.hour == 15 and now_ist.minute <= 35))
                ):
                    await asyncio.sleep(self.MONITOR_INTERVAL_S)
                    continue

                # 1. Per-account equity and drawdown
                for account_id in self._accounts:
                    await self._update_account_equity(account_id)

                # 2. Shared strategy metrics
                metrics = await self._compute_strategy_metrics()

                # 3. Evaluate kill conditions
                shared_kills = await self._kills.evaluate_shared_kills(metrics)
                account_kills = await self._kills.evaluate_account_kills(
                    self._risk._account_states,
                    self._accounts,
                )

                if shared_kills:
                    logger.critical(
                        "shared_kills_triggered",
                        kills=[k.strategy_id for k in shared_kills],
                    )

                if account_kills:
                    logger.critical(
                        "account_kills_triggered",
                        kills=[
                            f"{k.strategy_id}:{k.current_value:.2f}"
                            for k in account_kills
                        ],
                    )

                # 4. VIX regime update
                await self._update_vix_regime()

                # 5. Per-account margin utilization
                for account_id in self._accounts:
                    await self._check_margin_utilization(account_id)

                # 6. Aggregate exposure (informational)
                await self._aggregate.compute_aggregate_exposure()

                # 7. Prometheus metrics
                self._publish_metrics()

            except Exception:
                logger.exception("post_trade_monitor_error")

            await asyncio.sleep(self.MONITOR_INTERVAL_S)

    async def _update_account_equity(self, account_id: str) -> None:
        """
        Update equity, PnL, and drawdown for one account.

        Reads positions from the position tracker, computes
        unrealized PnL, updates peak equity and current drawdown.

        Args:
            account_id: The account to update.
        """
        state = self._risk._account_states[account_id]
        positions = self._positions.get_account_positions(account_id)

        unrealized_pnl = sum(p.unrealized_pnl for p in positions)
        realized_pnl_today = float(
            await self._redis.get(f"CACHE:realized_pnl_today:{account_id}") or 0
        )

        account = self._accounts[account_id]
        current_equity = account.capital_inr + realized_pnl_today + unrealized_pnl

        async with self._risk._account_locks[account_id]:
            state.current_equity_inr = current_equity

            if current_equity > state.peak_equity_inr:
                state.peak_equity_inr = current_equity

            if state.peak_equity_inr > 0:
                state.current_drawdown_pct = (
                    (state.peak_equity_inr - current_equity)
                    / state.peak_equity_inr
                    * -100
                )
            else:
                state.current_drawdown_pct = 0.0

        # Publish to Prometheus
        drawdown_pct.labels(strategy="portfolio", account=account_id).set(
            state.current_drawdown_pct
        )

    async def _compute_strategy_metrics(self) -> dict[str, dict[str, float]]:
        """
        Compute current metrics for all strategies.

        Reads daily returns from Redis cache (written by Position Tracker).
        Returns mapping of strategy_id to metric_name to current value.

        Returns:
            e.g., {"S1": {"after_cost_sharpe_60d": 0.42}, ...}
        """
        metrics: dict[str, dict[str, float]] = {}

        for sid in ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
            cached = await self._redis.lrange(f"RETURNS:{sid}:daily", 0, -1)
            if not cached:
                continue

            returns = orjson.loads(cached)
            sid_metrics: dict[str, float] = {}

            # Rolling Sharpe (60-day)
            if len(returns) >= 60:
                recent = returns[-60:]
                mean_r = sum(recent) / len(recent)
                std_r = (
                    sum((r - mean_r) ** 2 for r in recent) / len(recent)
                ) ** 0.5
                if std_r > 0:
                    sid_metrics["after_cost_sharpe_60d"] = (
                        mean_r / std_r * (252 ** 0.5)
                    )

            # Rolling Sharpe (6-month, ~126 trading days)
            if len(returns) >= 126:
                recent = returns[-126:]
                mean_r = sum(recent) / len(recent)
                std_r = (
                    sum((r - mean_r) ** 2 for r in recent) / len(recent)
                ) ** 0.5
                if std_r > 0:
                    sid_metrics["after_cost_sharpe_6m"] = (
                        mean_r / std_r * (252 ** 0.5)
                    )

            # Consecutive negative PnL days
            consec = 0
            for r in reversed(returns):
                if r < 0:
                    consec += 1
                else:
                    break
            sid_metrics["consecutive_negative_pnl_days"] = consec

            # Monthly drawdown
            if len(returns) >= 22:
                recent_month = returns[-22:]
                cumulative = 0.0
                peak = 0.0
                max_dd = 0.0
                for r in recent_month:
                    cumulative += r
                    if cumulative > peak:
                        peak = cumulative
                    dd = cumulative - peak
                    if dd < max_dd:
                        max_dd = dd
                sid_metrics["monthly_drawdown_pct"] = max_dd * 100

            metrics[sid] = sid_metrics

        return metrics

    async def _update_vix_regime(self) -> None:
        """Update VIX from latest tick data and set regime."""
        vix_str = await self._redis.get("TICK:INDIA_VIX:ltp")
        if vix_str is None:
            return

        vix = float(vix_str)
        async with self._risk._shared_lock:
            self._risk._shared_state.vix_current = vix
            if vix <= 20:
                self._risk._shared_state.vix_regime = "NORMAL"
            elif vix <= 25:
                self._risk._shared_state.vix_regime = "ELEVATED"
            else:
                self._risk._shared_state.vix_regime = "EXTREME"

    async def _check_margin_utilization(self, account_id: str) -> None:
        """Warn if margin utilization exceeds 80% for this account."""
        state = self._risk._account_states[account_id]

        if state.margin_available_inr <= 0:
            return

        total_margin = state.margin_used_inr + state.margin_available_inr
        utilization = state.margin_used_inr / total_margin * 100

        if utilization > 80:
            await self._telegram.send(
                "WARNING",
                f"MARGIN WARNING: account {account_id} at "
                f"{utilization:.0f}% margin utilization "
                f"(used: {state.margin_used_inr:.0f}, "
                f"available: {state.margin_available_inr:.0f})",
            )

    def _publish_metrics(self) -> None:
        """Publish current risk state to Prometheus gauges."""
        for account_id, state in self._risk._account_states.items():
            margin_utilized_pct.labels(account=account_id).set(
                state.margin_used_inr
                / max(state.margin_used_inr + state.margin_available_inr, 1)
                * 100
            )
            drawdown_pct.labels(strategy="portfolio", account=account_id).set(
                state.current_drawdown_pct
            )

        vix_gauge.set(self._risk._shared_state.vix_current)

        for sid in self._risk._shared_state.killed_strategies:
            strategy_status.labels(strategy=sid).set(0)
```

---

### State Table

Every piece of state in the risk management layer, showing where it lives and whether it is per-account or shared.

| State | Storage | Per-Account or Shared | Lifecycle | Written By | Read By |
|-------|---------|----------------------|-----------|------------|---------|
| `AccountRiskState` (equity, drawdown, margin) | In-memory + Redis `RISK:state:{account_id}` | Per-Account | Session, restored on startup | PostTradeMonitor | RiskManager, KillEvaluator |
| `SharedRiskState` (VIX, killed set) | In-memory | Shared | Session | PostTradeMonitor | RiskManager (pre-trade checks) |
| `KILLED:{strategy_id}` | Redis | Shared | Until manual DEL | KillEvaluator | RiskManager (check 6) |
| `HALT:global` | Redis | Shared | Until manual DEL | Global kill switch | RiskManager (check 7) |
| `HALT:{account_id}` | Redis | Per-Account | Until manual DEL | KillEvaluator | RiskManager (check 7) |
| `RISK:trade_count:{account_id}:{strategy_id}` | Redis | Per-Account | Reset 08:30 IST daily | OMS (INCR on each fill event) | RiskManager (check 2) |
| `RISK:trade_count:{account_id}:portfolio` | Redis | Per-Account | Reset 08:30 IST daily | OMS (INCR on each fill event) | RiskManager (check 2) |
| `OMS:daily_count:{account_id}` | Redis | Per-Account | Reset 08:30 IST daily | OMS (on placement) | RiskManager (check 9) |
| `POSITION:strategy:{strategy_id}` | Redis | Shared | Set/cleared on position open/close | PositionTracker | RiskManager (check 5) |
| `RETURNS:{sid}:daily` | Redis LIST | Shared | Updated daily (weighted sum across accounts) | PositionTracker | PostTradeMonitor |
| `CACHE:realized_pnl_today:{account_id}` | Redis | Per-Account | Reset daily | PositionTracker | PostTradeMonitor |
| Kill condition states | In-memory (`_kill_states` dict) | Shared | Session | KillEvaluator | KillEvaluator (dedup) |
| Post-order recon timers | In-memory (asyncio.Task) | Per-Account | Created on order, cancelled on fill | PerAccountReconciler | PerAccountReconciler |
| Aggregate exposure report | Computed on demand | Shared (derived) | Transient | AggregateMonitor | Monitoring/logging |

---

### Concurrency Model

Risk checks run with per-account parallelism and shared-state locking. The design allows N accounts to be risk-checked simultaneously while protecting shared state from races.

#### Per-Account Parallelism

When a signal fans out to N accounts (via `MultiAccountFanOut.fan_out_signal`), each account's risk check runs as an independent `asyncio.Task`. These tasks execute concurrently within the same event loop. There is no cross-account contention because:

1. **Rate limiters are per-account.** Account A's margin check uses Account A's rate limiter. Account B's runs in parallel using its own.
2. **Account state writes are per-account locked.** Each account has its own `asyncio.Lock` (`_account_locks[account_id]`). No two coroutines write the same account's state simultaneously.
3. **Broker API calls are per-account.** Each `GET /v2/positions` call uses a different API key and hits a different account on Dhan's backend.

#### Shared State Locking

Shared state (VIX, killed strategies, S1/S3 correlation) is protected by `_shared_lock`:

```python
# Writing shared state (PostTradeMonitor, KillEvaluator)
async with self._risk._shared_lock:
    self._risk._shared_state.vix_current = vix
    self._risk._shared_state.vix_regime = "EXTREME"

# Reading shared state (pre-trade checks)
# No lock needed — reads are atomic for simple Python attributes.
# The worst case is reading a slightly stale VIX value, which is
# acceptable (VIX changes slowly relative to the 10s update cycle).
vix = self._shared_state.vix_current
```

Redis operations (`KILLED:*`, `HALT:*`) are atomic by nature. No application-level locking needed for Redis reads/writes.

#### Race: Kill Condition vs New Signal

A signal may pass the `KILLED:S1` check at time T, but the kill evaluator sets `KILLED:S1` at time T+1ms. The signal proceeds to order placement with the strategy about to be killed. This is acceptable because:

1. The kill condition takes 10+ seconds to materialize (it evaluates on the monitor cycle, not per-tick).
2. The new order will have a stop-loss attached. The worst case is one additional trade in a dying strategy with SL protection.
3. The alternative (holding a global lock across all risk checks) would serialize all accounts and add latency.

#### Concurrent Risk Checks for Same Account

This cannot happen by construction. The `MultiAccountFanOut` creates exactly one task per account per signal. Signals are processed sequentially from the Redis stream. If signal B arrives while signal A is still in the risk gate for Account X, signal B waits for signal A's task to complete (they are awaited sequentially within the fan-out loop). There is no scenario where two signals run risk checks for the same account simultaneously.

---

### Failure Modes

| # | Scenario | Detection | Impact | Handling | Severity |
|---|----------|-----------|--------|----------|----------|
| 1 | Risk check timeout (> 5s) | `asyncio.wait_for` wrapper | Order delayed for one account | Cancel the check, reject the order for this account. Other accounts proceed. Log WARNING. | Medium |
| 2 | Margin API failure (per-account) | `BrokerAPIError` exception | Cannot verify margin for this account | HARD BLOCK this account's order. Other accounts unaffected. Retry on next signal. | High |
| 3 | Ghost position detected | Broker positions API mismatch | Double-position risk | Block order, trigger force_sync_from_broker(account_id), Telegram CRITICAL. Order is rejected for this account. | Critical |
| 4 | Kill condition race with new signal | Signal passes check 6, kill fires immediately after | One extra trade in dying strategy | Acceptable: trade has SL protection. Next signal will see KILLED flag. | Low |
| 5 | Redis failure (read) | `RedisError` exception | Cannot read KILLED/HALT flags, trade counts | Fall back to in-memory state. If in-memory state is stale (process restart), BLOCK all orders until Redis recovers. | Critical |
| 6 | Redis failure (write) | `RedisError` exception on SET | Kill/halt state not persisted | Retry write 3 times with 100ms backoff. If all fail, apply kill in-memory only and Telegram CRITICAL. Risk: kill lost on process restart. | Critical |
| 7 | False positive ghost detection | Broker returns stale position (e.g., intraday SELL shows as position until T+1 settlement) | Legitimate order blocked | Manual override via Redis: `SET RISK:ghost_override:{account_id}:{instrument_id} 1 EX 300`. Operator must verify. Timer-based TTL prevents permanent override. | Medium |
| 8 | Position tracker desync | Local state drifts from broker (missed fills, partial fill race) | Incorrect drawdown calculation, wrong kill decisions | Periodic 5-minute reconciliation catches drift. Force sync resolves it. Drawdown is re-computed from broker positions after sync. | High |
| 9 | VIX data stale | No TICK:INDIA_VIX update for > 60s | VIX regime check uses stale value | If VIX tick age > 60s, treat regime as ELEVATED (conservative). Log WARNING. | Medium |
| 10 | Kill evaluator crash | Unhandled exception in PostTradeMonitor.run() | Kill conditions not evaluated | Top-level try/except logs the error and continues the loop. If the monitor loop itself dies, the process health check (systemd watchdog) restarts the process. All SLs are server-side on the exchange, so risk is bounded. | High |
| 11 | Concurrent force_sync for same account | Two triggers (periodic + post-order timer) fire simultaneously | Redundant API calls, potential state confusion | Per-account lock (`_account_locks[account_id]`) serializes force_sync calls. Second call reads already-corrected state and finds no discrepancy. | Low |
| 12 | Broker rate limit exceeded during risk check | HTTP 429 from Dhan | Ghost check or margin check fails | PriorityRateLimiter prevents this by design (risk checks use RISK priority, which has reserved tokens). If it still happens: treat as API failure, block order. | Medium |
| 13 | Account halt during active fill management | Account halted while OMS is managing an order fill | Order in progress on halted account | Halt only blocks NEW entries. In-progress fills continue to completion (they already have SLs). This is correct behavior: halting mid-fill would leave an unprotected position. | Low |
| 14 | Multiple strategies trigger flatten_immediately simultaneously | S6 kill fires while another flatten is in progress | Duplicate exit orders for same positions | OMS dedup: if an exit order already exists for a position (tracked via instrument_id + account_id + direction), skip. PositionTracker marks position as "flattening" to prevent double exits. | Medium |

---

### Edge Cases

#### Edge Case 1: Account Added Mid-Session

A new account is added to the configuration while the system is running.

**Handling:** The system does NOT support hot-adding accounts. Adding an account requires a process restart. On restart, the new account initializes with fresh `AccountRiskState`, runs startup reconciliation, and joins the fan-out pool. This is a deliberate simplification for v1.

#### Edge Case 2: Account Halted, Then Kill Condition Fires for a Strategy It Holds

Account B is halted (drawdown limit). Then S6 triggers `flatten_immediately`.

**Handling:** `flatten_immediately` overrides the account halt for exit orders only. The `halt_account` flag blocks new entries, but flatten orders use `urgency="URGENT"` which bypasses the halt check. The account remains halted for new entries after the flatten completes.

```python
# In pre_trade_check, the halt check does NOT apply to exit orders
if order.is_exit or order.urgency == "URGENT":
    pass  # skip halt check for exits
else:
    ok, reason = await self.check_not_halted(account)
    if not ok:
        return RiskCheckResult(passed=False, ...)
```

#### Edge Case 3: S1 and S3 Active on Different Accounts

Account A holds S1, Account B wants to enter S3. The correlation check is shared (reads `POSITION:strategy:S1`), so S3 is blocked on ALL accounts, including Account B.

**Handling:** This is intentional. S1 and S3 trade the same underlying (NIFTY) with anti-correlated signals. Even on different accounts, simultaneous S1+S3 positions create a costly hedge across the operator's total capital. The shared block prevents this.

#### Edge Case 4: Ghost Check Returns Empty Positions (Market Not Yet Open)

At 09:15, the ghost check queries `GET /v2/positions` but the broker returns an empty list because no positions exist yet (clean start of day).

**Handling:** Local state should also be empty after startup reconciliation. Empty broker + empty local = no ghosts = check passes. If local state has leftover positions from yesterday (process didn't restart cleanly), this correctly triggers a ghost detection (local has positions broker doesn't), which force-syncs from broker and clears the stale local state.

#### Edge Case 5: VIX Crosses 25 While S1 Order is in Fill Management

S1 passed the VIX check at VIX=24.8. During the fill management loop (which may take 30+ seconds), VIX rises to 25.3.

**Handling:** The pre-trade check is a point-in-time gate. Once passed, the order proceeds through fill management. The VIX regime change will suppress the NEXT S1 signal, not cancel the current one. Cancelling an in-progress fill would leave the SL order orphaned and require complex cleanup. The fill will complete (or timeout) and the position will have SL protection.

#### Edge Case 6: Daily Order Count Reaches 4500 on One Account During Multi-Account Fan-Out

A signal fans out to 3 accounts. Account A (4499 orders) passes the check. Account B (4501 orders) fails. Account C (2000 orders) passes.

**Handling:** Each account's check is independent. Account B is rejected; Accounts A and C proceed. The `AccountOrderResult` for Account B will have `outcome="RISK_REJECTED"` and `error="daily_order_limit"`. The signal log records the divergence.

#### Edge Case 7: Reconciliation Force-Sync Discovers Orphan Position

During the 5-minute reconciliation, the broker shows a position that local state doesn't track AND cannot attribute to any strategy (no matching signal_id, no matching order_id in recent history).

**Handling:** The position is attributed to strategy `"ORPHAN"` and Telegram CRITICAL is sent. The operator must manually investigate. The orphan position will not have a local SL, but the broker may have a server-side SL if it was placed by the OMS (SLs survive process restarts on the exchange). If no SL exists, the operator must manually place one or flatten the position.

#### Edge Case 8: Redis Recovers After Extended Outage

Redis was down for 2 minutes. During this time, the system operated in degraded mode (in-memory state only). Redis comes back.

**Handling:** On Redis recovery, the system writes current in-memory state back to Redis:
- All `AccountRiskState` objects
- Current trade counts (may be stale if trades happened during outage)
- Current `KILLED:*` and `HALT:*` flags

Trade counts are re-read from the OMS's in-memory counters (which were maintained during the outage). Any kills triggered during the outage were applied in-memory and are now persisted to Redis. The system logs a WARNING with the duration of the Redis outage and the state delta.

```python
async def on_redis_recovery(self) -> None:
    """
    Called when Redis connection is re-established after outage.

    Writes current in-memory state to Redis to ensure persistence.
    """
    for account_id, state in self._account_states.items():
        await self._redis.set(
            f"RISK:state:{account_id}",
            state.model_dump_json(),
        )

        if state.halted:
            await self._redis.set(
                f"HALT:{account_id}",
                state.halted_reason or "recovered_from_outage",
            )

    for sid in self._shared_state.killed_strategies:
        await self._redis.set(f"KILLED:{sid}", "recovered_from_outage")

    logger.warning(
        "redis_recovery_state_sync",
        accounts_synced=len(self._account_states),
        killed_strategies=list(self._shared_state.killed_strategies),
    )
```

#### Edge Case 9: Two Accounts Place Orders for Same Instrument Simultaneously

Account A and Account B both receive an S1 signal and both pass risk checks. Both place LIMIT BUY orders for the same NIFTY CE option at the same price.

**Handling:** This is correct behavior. Each account is independent on the exchange. Both orders enter the exchange order book separately. They may fill at different times and different prices. The combined exposure is 2x a single account's position size, which is expected in multi-account mode. The aggregate monitor will report the combined notional, but will not block it.

#### Edge Case 10: Kill Condition Threshold Exactly Met

S1 Sharpe is exactly 0.5 (the threshold). The comparator is `lt` (less than).

**Handling:** `0.5 < 0.5` is `False`, so the kill does NOT trigger. The strategy continues. The kill only fires when Sharpe drops strictly below 0.5. This is intentional: the threshold represents the minimum acceptable performance, not a rounding boundary.

---


## Component 8: Position & PnL Tracker

> **Multi-Account Note:** The Position Tracker is PER-ACCOUNT for position state and SHARED as a process. A single PositionTracker async process manages positions for ALL accounts, but each account's positions are tracked independently with separate DuckDB tables, separate Redis cache keys, and separate reconciliation cycles. Market data (prices for MTM) is shared. Entry prices, lot counts, and PnL differ across accounts because fill prices and allocation sizes diverge.

### Responsibility

- Real-time position state for every account (sole source of truth locally)
- Realized and unrealized PnL per account
- **Sole writer to DuckDB** -- all other components read via Redis cache
- Per-account reconciliation with Dhan positions API
- Greeks computation for option positions (shared market data, per-position results)
- State restoration on startup (per-account broker query + Redis merge)
- Aggregate views across accounts for monitoring and PMS reporting

---

### Position Model

Every position is uniquely identified by `(account_id, security_id, strategy_id)`. Two accounts holding the same instrument for the same strategy are two distinct Position objects with independent entry prices, quantities, and PnL.

```python
from datetime import date
from typing import Literal
import pydantic


class Position(pydantic.BaseModel):
    """
    Single position for one account, one instrument, one strategy.

    Position key: (account_id, security_id, strategy_id).
    Two accounts holding NIFTY 24500 CE for S1 are two Position objects.
    """
    # --- Identity ---
    account_id: str                          # Dhan client ID (e.g., "1000000001")
    strategy_id: str                         # S1..S7
    security_id: str                         # Dhan security ID (from instrument master)
    trading_symbol: str                      # e.g., "NIFTY-24500-CE-2026-04-02"
    underlying: str                          # e.g., "NIFTY50", "BANKNIFTY"
    instrument_type: Literal[
        "CE", "PE", "FUT", "EQ"
    ]

    # --- Position state ---
    direction: Literal["LONG", "SHORT"]
    quantity: int                            # signed: +ve for long, -ve for short
    avg_entry_price: float                   # VWAP of all entry fills
    current_price: float                     # last MTM price (bid/ask/LTP per table)
    unrealized_pnl: float                    # (current_price - avg_entry_price) * quantity
    realized_pnl: float                      # accumulated realized PnL for this position
    entry_ts: int                            # epoch ms of first entry fill
    last_update_ts: int                      # epoch ms of last price/fill update

    # --- Derivative fields (None for EQ) ---
    strike: float | None = None
    expiry: date | None = None
    option_type: Literal["CE", "PE"] | None = None
    lot_size: int | None = None              # exchange lot size (e.g., 65 for NIFTY)

    # --- Greeks (None for non-options) ---
    delta: float | None = None
    gamma: float | None = None
    theta: float | None = None
    vega: float | None = None
    iv: float | None = None                  # implied volatility used for Greeks

    # --- Order tracking ---
    order_ids: list[str] = []                # all entry order IDs that built this position
    sl_order_id: str | None = None           # current active SL order ID
    sl_type: Literal["DAY", "GTT"] | None = None

    # --- Flags ---
    is_overnight: bool = False               # True for S2, S6, S7 positions held overnight
    is_orphan: bool = False                  # True if found on broker but unknown locally

    @property
    def position_key(self) -> tuple[str, str, str]:
        return (self.account_id, self.security_id, self.strategy_id)

    @property
    def notional_value(self) -> float:
        return abs(self.quantity) * self.current_price

    @property
    def is_option(self) -> bool:
        return self.instrument_type in ("CE", "PE")

    def update_mtm(self, price: float, ts: int) -> None:
        """Update mark-to-market price and recalculate unrealized PnL."""
        self.current_price = price
        if self.direction == "LONG":
            self.unrealized_pnl = (price - self.avg_entry_price) * abs(self.quantity)
        else:
            self.unrealized_pnl = (self.avg_entry_price - price) * abs(self.quantity)
        self.last_update_ts = ts
```

---

### TradeRecord Model

Every fill generates a TradeRecord written to DuckDB. This is the immutable audit trail of executions per account.

```python
class TradeRecord(pydantic.BaseModel):
    """
    Immutable record of a single trade execution for one account.
    Written to trades_{account_id} table in DuckDB.
    """
    trade_id: str                            # UUID, globally unique
    account_id: str                          # Dhan client ID
    strategy_id: str                         # S1..S7
    signal_id: str                           # originating signal UUID
    order_id: str                            # Dhan order ID
    exchange_order_id: str | None            # exchange order ID (from Dhan)

    security_id: str                         # Dhan security ID
    trading_symbol: str                      # human-readable symbol
    underlying: str
    instrument_type: str                     # CE, PE, FUT, EQ
    strike: float | None = None
    expiry: date | None = None
    option_type: str | None = None

    transaction_type: Literal["BUY", "SELL"]
    trade_type: Literal[
        "ENTRY",                             # opening a new position
        "EXIT",                              # closing an existing position
        "SL_TRIGGERED",                      # stop-loss triggered fill
        "EOD_FLATTEN",                       # end-of-day flatten
        "EMERGENCY_EXIT",                    # global kill / risk halt
        "PARTIAL_ENTRY",                     # partial fill on entry
        "PARTIAL_EXIT",                      # partial fill on exit
    ]

    quantity: int                            # filled quantity (always positive)
    price: float                             # fill price
    notional_value: float                    # quantity * price

    # --- Cost breakdown (INR) ---
    brokerage: float                         # Dhan: 0 for delivery, 20/order for intraday
    stt: float                               # Securities Transaction Tax
    exchange_fees: float                     # NSE transaction charges
    sebi_fee: float                          # SEBI turnover fee
    gst: float                               # GST on brokerage + exchange fees
    stamp_duty: float                        # stamp duty (buy-side only)
    total_cost: float                        # sum of all costs

    # --- Timestamps ---
    exchange_ts: int                         # exchange fill timestamp (epoch ms)
    local_ts: int                            # our receive timestamp (epoch ms)
    trade_date: date                         # IST date of trade

    # --- Context ---
    entry_price: float | None = None         # avg entry for position (for exit trades)
    realized_pnl: float | None = None        # PnL realized by this trade (exits only)
    spot_at_fill: float | None = None        # underlying spot at time of fill
    iv_at_fill: float | None = None          # IV at time of fill (options only)
```

---

### Per-Account DuckDB Tables

DuckDB does not support concurrent writes from multiple processes. The PositionTracker is the sole writer. All tables are partitioned by account_id via separate table names. This avoids row-level filtering overhead and makes per-account queries trivial.

#### Schema Definitions

```sql
-- ============================================================
-- trades_{account_id}: every fill execution
-- ============================================================
CREATE TABLE IF NOT EXISTS trades_{account_id} (
    trade_id            VARCHAR PRIMARY KEY,
    strategy_id         VARCHAR NOT NULL,
    signal_id           VARCHAR NOT NULL,
    order_id            VARCHAR NOT NULL,
    exchange_order_id   VARCHAR,

    security_id         VARCHAR NOT NULL,
    trading_symbol      VARCHAR NOT NULL,
    underlying          VARCHAR NOT NULL,
    instrument_type     VARCHAR NOT NULL,       -- CE, PE, FUT, EQ
    strike              DOUBLE,
    expiry              DATE,
    option_type         VARCHAR,                -- CE, PE, NULL

    transaction_type    VARCHAR NOT NULL,       -- BUY, SELL
    trade_type          VARCHAR NOT NULL,       -- ENTRY, EXIT, SL_TRIGGERED, ...
    quantity            INTEGER NOT NULL,
    price               DOUBLE NOT NULL,
    notional_value      DOUBLE NOT NULL,

    brokerage           DOUBLE NOT NULL DEFAULT 0,
    stt                 DOUBLE NOT NULL DEFAULT 0,
    exchange_fees       DOUBLE NOT NULL DEFAULT 0,
    sebi_fee            DOUBLE NOT NULL DEFAULT 0,
    gst                 DOUBLE NOT NULL DEFAULT 0,
    stamp_duty          DOUBLE NOT NULL DEFAULT 0,
    total_cost          DOUBLE NOT NULL DEFAULT 0,

    exchange_ts         BIGINT NOT NULL,        -- epoch ms
    local_ts            BIGINT NOT NULL,
    trade_date          DATE NOT NULL,

    entry_price         DOUBLE,
    realized_pnl        DOUBLE,
    spot_at_fill        DOUBLE,
    iv_at_fill          DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_trades_{account_id}_strategy
    ON trades_{account_id} (strategy_id, trade_date);

CREATE INDEX IF NOT EXISTS idx_trades_{account_id}_date
    ON trades_{account_id} (trade_date);

CREATE INDEX IF NOT EXISTS idx_trades_{account_id}_security
    ON trades_{account_id} (security_id, trade_date);


-- ============================================================
-- daily_returns_{account_id}: one row per strategy per day
-- Used by Capital Allocator (ERC) via Redis cache
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_returns_{account_id} (
    trade_date          DATE NOT NULL,
    strategy_id         VARCHAR NOT NULL,
    gross_pnl           DOUBLE NOT NULL,        -- before costs
    total_costs         DOUBLE NOT NULL,
    net_pnl             DOUBLE NOT NULL,        -- after costs
    capital_deployed    DOUBLE NOT NULL,         -- capital allocated that day
    net_return          DOUBLE NOT NULL,         -- net_pnl / capital_deployed
    num_trades          INTEGER NOT NULL,
    num_winners         INTEGER NOT NULL,
    num_losers          INTEGER NOT NULL,
    max_intraday_dd     DOUBLE NOT NULL,         -- worst intraday drawdown
    end_of_day_position VARCHAR NOT NULL,        -- FLAT, LONG, SHORT
    PRIMARY KEY (trade_date, strategy_id)
);

CREATE INDEX IF NOT EXISTS idx_daily_returns_{account_id}_strategy
    ON daily_returns_{account_id} (strategy_id, trade_date);


-- ============================================================
-- realized_pnl_{account_id}: per-position realized PnL
-- One row per closed position (entry + exit matched)
-- ============================================================
CREATE TABLE IF NOT EXISTS realized_pnl_{account_id} (
    position_id         VARCHAR PRIMARY KEY,     -- UUID
    strategy_id         VARCHAR NOT NULL,
    security_id         VARCHAR NOT NULL,
    trading_symbol      VARCHAR NOT NULL,
    underlying          VARCHAR NOT NULL,
    instrument_type     VARCHAR NOT NULL,
    direction           VARCHAR NOT NULL,        -- LONG, SHORT

    entry_price         DOUBLE NOT NULL,
    exit_price          DOUBLE NOT NULL,
    quantity            INTEGER NOT NULL,
    gross_pnl           DOUBLE NOT NULL,
    total_costs         DOUBLE NOT NULL,
    net_pnl             DOUBLE NOT NULL,

    entry_ts            BIGINT NOT NULL,
    exit_ts             BIGINT NOT NULL,
    entry_date          DATE NOT NULL,
    exit_date           DATE NOT NULL,
    holding_period_ms   BIGINT NOT NULL,

    exit_reason         VARCHAR NOT NULL,        -- SIGNAL, SL_TRIGGERED, EOD_FLATTEN, EMERGENCY
    entry_order_ids     VARCHAR NOT NULL,         -- JSON array of order IDs
    exit_order_ids      VARCHAR NOT NULL,          -- JSON array of order IDs

    spot_at_entry       DOUBLE,
    spot_at_exit        DOUBLE,
    iv_at_entry         DOUBLE,
    iv_at_exit          DOUBLE
);

CREATE INDEX IF NOT EXISTS idx_realized_pnl_{account_id}_strategy
    ON realized_pnl_{account_id} (strategy_id, exit_date);


-- ============================================================
-- reconciliation_{account_id}: audit trail of every recon event
-- ============================================================
CREATE TABLE IF NOT EXISTS reconciliation_{account_id} (
    recon_id            VARCHAR PRIMARY KEY,
    recon_type          VARCHAR NOT NULL,        -- QUICK, FULL, POST_ORDER, STARTUP
    recon_ts            BIGINT NOT NULL,
    status              VARCHAR NOT NULL,        -- MATCH, DISCREPANCY, FORCE_SYNCED

    -- discrepancy details (NULL if MATCH)
    security_id         VARCHAR,
    local_qty           INTEGER,
    broker_qty          INTEGER,
    local_avg_price     DOUBLE,
    broker_avg_price    DOUBLE,
    action_taken        VARCHAR,                 -- FORCE_SYNC, ORPHAN_CREATED, NONE
    details             VARCHAR                  -- JSON blob with full context
);


-- ============================================================
-- metadata: schema versioning
-- ============================================================
CREATE TABLE IF NOT EXISTS metadata (
    key                 VARCHAR PRIMARY KEY,
    value               VARCHAR NOT NULL
);

-- Initial version
INSERT OR IGNORE INTO metadata (key, value) VALUES ('schema_version', '1');
```

#### Single-Writer Pattern

```
PositionTracker (sole DuckDB writer, single async process)
  ├── Tables per account: trades_{aid}, daily_returns_{aid},
  │                       realized_pnl_{aid}, reconciliation_{aid}
  ├── Writes: on every fill, on recon, daily EOD summary
  ├── Reads: on startup (restore state), on request (daily returns for ERC)
  └── Caches to Redis:
        CACHE:daily_returns:{account_id}:{strategy_id}  -- JSON list, updated daily
        CACHE:strategy_pnl:{account_id}:{strategy_id}   -- current PnL, updated on fill
        CACHE:account_nav:{account_id}                   -- NAV, updated every MTM cycle
        POSITION:strategy:{strategy_id}                  -- aggregated position (all accounts)
        POSITION:account:{account_id}:{strategy_id}      -- per-account position

Other components (read from Redis cache, NEVER DuckDB directly):
  ├── Capital Allocator reads CACHE:daily_returns:{account_id}:{strategy_id}
  ├── Risk Manager reads CACHE:strategy_pnl:{account_id}:{strategy_id}
  ├── Strategy processes read POSITION:strategy:{strategy_id}
  └── Monitoring reads CACHE:account_nav:{account_id}
```

Why this pattern: DuckDB uses an OS-level file lock for writes. If two processes attempt concurrent writes, one blocks until the other finishes -- or worse, the file corrupts on unclean termination. By funneling all writes through the PositionTracker process, we guarantee sequential write access. Reads from other processes are safe (DuckDB supports concurrent reads) but we avoid even that by caching everything in Redis, keeping DuckDB as a durable store only.

---

### Per-Account Reconciliation

Each account is reconciled independently with its own Dhan positions API response. Account A may show a discrepancy while Account B matches perfectly.

#### Reconciliation Schedule

| When | Type | What | OPS Cost |
|------|------|------|----------|
| 09:16 IST | Full | Startup, after market open. Per account. | 1 per account |
| Every 5 min | Quick | security_id + quantity check. Per account. | 1 per account |
| Post-order timer | Targeted | Timer fires `max_patience_s + 5s` after placement. Per account per order. | 1 per account |
| 15:31 IST | Full | EOD reconciliation. Per account. | 1 per account |

With N accounts, the 5-minute quick recon costs N API calls (one per account), staggered 500ms apart to avoid burst. Total OPS impact at 3 accounts: 3 calls every 5 min = negligible.

#### Dhan Positions API

```python
# GET https://api.dhan.co/v2/positions
# Headers: {"access-token": "{account_access_token}"}
#
# Returns all positions for the authenticated account.

class DhanPosition(pydantic.BaseModel):
    """Single position from Dhan positions API response."""
    dhanClientId: str
    tradingSymbol: str
    securityId: str
    positionType: str                        # "LONG", "SHORT", "CLOSED"
    exchangeSegment: str                     # "NSE_FNO", "NSE_EQ"
    productType: str                         # "INTRADAY", "CNC", "MARGIN"
    buyAvg: float
    sellAvg: float
    netQty: int                              # positive = long, negative = short
    buyQty: int
    sellQty: int
    realizedProfit: float
    unrealizedProfit: float
    multiplier: int                          # lot multiplier (1 for equity)
    drvExpiryDate: str | None
    drvOptionType: str | None
    drvStrikePrice: float | None
```

#### Reconciliation Implementation

```python
class Discrepancy(pydantic.BaseModel):
    """Single discrepancy found during reconciliation."""
    account_id: str
    security_id: str
    trading_symbol: str
    local_qty: int
    broker_qty: int
    local_avg_price: float
    broker_avg_price: float
    discrepancy_type: Literal[
        "QTY_MISMATCH",           # quantities differ
        "MISSING_LOCAL",           # broker has it, we don't
        "MISSING_BROKER",          # we have it, broker doesn't
        "PRICE_MISMATCH",          # qty matches but avg price differs
    ]


async def reconcile_account(
    self,
    account_id: str,
    recon_type: Literal["QUICK", "FULL", "POST_ORDER", "STARTUP"],
) -> list[Discrepancy]:
    """
    Reconcile positions for a single account against Dhan API.

    QUICK: compare security_id + netQty only (fast, catches gross errors)
    FULL: compare security_id + netQty + avg_price (catches price drift)

    Returns list of discrepancies. Empty list = all matched.
    """
    account = self._accounts[account_id]
    rate_limiter = self._account_rate_limiters[account_id]

    # 1. Fetch broker positions for THIS account
    await rate_limiter.acquire("POLL")
    broker_positions: list[DhanPosition] = await self._dhan_get_positions(account)

    # 2. Build broker position map: security_id -> DhanPosition
    broker_map: dict[str, DhanPosition] = {}
    for bp in broker_positions:
        if bp.netQty != 0:  # skip closed positions
            broker_map[bp.securityId] = bp

    # 3. Build local position map for this account
    local_map: dict[str, Position] = {}
    for pos in self._positions.values():
        if pos.account_id == account_id:
            local_map[pos.security_id] = pos

    discrepancies: list[Discrepancy] = []

    # 4. Check all local positions exist on broker with correct qty
    for sec_id, local_pos in local_map.items():
        broker_pos = broker_map.pop(sec_id, None)
        if broker_pos is None:
            discrepancies.append(Discrepancy(
                account_id=account_id,
                security_id=sec_id,
                trading_symbol=local_pos.trading_symbol,
                local_qty=local_pos.quantity,
                broker_qty=0,
                local_avg_price=local_pos.avg_entry_price,
                broker_avg_price=0.0,
                discrepancy_type="MISSING_BROKER",
            ))
            continue

        if local_pos.quantity != broker_pos.netQty:
            discrepancies.append(Discrepancy(
                account_id=account_id,
                security_id=sec_id,
                trading_symbol=local_pos.trading_symbol,
                local_qty=local_pos.quantity,
                broker_qty=broker_pos.netQty,
                local_avg_price=local_pos.avg_entry_price,
                broker_avg_price=(broker_pos.buyAvg
                                  if broker_pos.netQty > 0
                                  else broker_pos.sellAvg),
                discrepancy_type="QTY_MISMATCH",
            ))
        elif recon_type in ("FULL", "STARTUP"):
            # FULL recon also checks avg price
            broker_avg = (broker_pos.buyAvg
                         if broker_pos.netQty > 0
                         else broker_pos.sellAvg)
            if abs(local_pos.avg_entry_price - broker_avg) > 0.05:
                discrepancies.append(Discrepancy(
                    account_id=account_id,
                    security_id=sec_id,
                    trading_symbol=local_pos.trading_symbol,
                    local_qty=local_pos.quantity,
                    broker_qty=broker_pos.netQty,
                    local_avg_price=local_pos.avg_entry_price,
                    broker_avg_price=broker_avg,
                    discrepancy_type="PRICE_MISMATCH",
                ))

    # 5. Check remaining broker positions not in local state
    for sec_id, broker_pos in broker_map.items():
        discrepancies.append(Discrepancy(
            account_id=account_id,
            security_id=sec_id,
            trading_symbol=broker_pos.tradingSymbol,
            local_qty=0,
            broker_qty=broker_pos.netQty,
            local_avg_price=0.0,
            broker_avg_price=(broker_pos.buyAvg
                             if broker_pos.netQty > 0
                             else broker_pos.sellAvg),
            discrepancy_type="MISSING_LOCAL",
        ))

    # 6. Log reconciliation result to DuckDB
    recon_id = str(uuid.uuid4())
    status = "MATCH" if not discrepancies else "DISCREPANCY"
    self._write_recon_record(account_id, recon_id, recon_type, status,
                             discrepancies)

    # 7. If discrepancies found, force sync and alert
    if discrepancies:
        logger.critical("position_discrepancy",
                       account_id=account_id,
                       recon_type=recon_type,
                       discrepancies=[d.model_dump() for d in discrepancies])

        await telegram.send(CRITICAL,
            f"POSITION DISCREPANCY: Account {account_id}\n"
            + "\n".join(
                f"  {d.trading_symbol}: local={d.local_qty} broker={d.broker_qty} "
                f"({d.discrepancy_type})"
                for d in discrepancies
            ))

        await self.force_sync_from_broker(account_id)

    return discrepancies


async def force_sync_from_broker(self, account_id: str) -> None:
    """
    Force-sync local position state to match broker for one account.

    Broker is ALWAYS source of truth. This overwrites local state.

    Steps:
    1. Fetch broker positions for this account
    2. For each broker position:
       a. If local position exists: update qty and avg price
       b. If no local position: create ORPHAN position
    3. For each local position not on broker:
       a. Mark as closed (qty went to zero via unknown fill)
    4. Update Redis cache
    """
    account = self._accounts[account_id]
    rate_limiter = self._account_rate_limiters[account_id]

    await rate_limiter.acquire("POLL")
    broker_positions = await self._dhan_get_positions(account)

    broker_map: dict[str, DhanPosition] = {
        bp.securityId: bp
        for bp in broker_positions
        if bp.netQty != 0
    }

    # --- Sync existing + create orphans ---
    synced_security_ids: set[str] = set()

    for sec_id, broker_pos in broker_map.items():
        synced_security_ids.add(sec_id)

        # Find local position for this account + security
        local_key = None
        local_pos = None
        for key, pos in self._positions.items():
            if pos.account_id == account_id and pos.security_id == sec_id:
                local_key = key
                local_pos = pos
                break

        broker_avg = (broker_pos.buyAvg
                     if broker_pos.netQty > 0
                     else broker_pos.sellAvg)

        if local_pos is not None:
            # Update local to match broker
            local_pos.quantity = broker_pos.netQty
            local_pos.direction = "LONG" if broker_pos.netQty > 0 else "SHORT"
            local_pos.avg_entry_price = broker_avg
            local_pos.last_update_ts = now_ms()
            logger.warning("force_sync_updated",
                          account_id=account_id,
                          security_id=sec_id,
                          new_qty=broker_pos.netQty)
        else:
            # Broker has position we don't know about -> ORPHAN
            orphan = Position(
                account_id=account_id,
                strategy_id="ORPHAN",
                security_id=sec_id,
                trading_symbol=broker_pos.tradingSymbol,
                underlying=self._infer_underlying(broker_pos),
                instrument_type=self._infer_instrument_type(broker_pos),
                direction="LONG" if broker_pos.netQty > 0 else "SHORT",
                quantity=broker_pos.netQty,
                avg_entry_price=broker_avg,
                current_price=broker_avg,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_ts=now_ms(),
                last_update_ts=now_ms(),
                is_orphan=True,
            )
            key = orphan.position_key
            self._positions[key] = orphan

            logger.critical("orphan_position_created",
                          account_id=account_id,
                          security_id=sec_id,
                          trading_symbol=broker_pos.tradingSymbol,
                          qty=broker_pos.netQty)

            await telegram.send(CRITICAL,
                f"ORPHAN POSITION: Account {account_id} "
                f"{broker_pos.tradingSymbol} qty={broker_pos.netQty}. "
                f"Unknown to local state. Manual review required.")

    # --- Remove local positions not on broker ---
    keys_to_remove = []
    for key, pos in self._positions.items():
        if (pos.account_id == account_id
                and pos.security_id not in synced_security_ids):
            logger.warning("force_sync_removed_local",
                          account_id=account_id,
                          security_id=pos.security_id,
                          prev_qty=pos.quantity)
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del self._positions[key]

    # --- Update Redis cache ---
    await self._publish_all_positions(account_id)
```

---

### Per-Account Mark-to-Market

Market data is shared: all accounts see the same bid, ask, and LTP for every instrument. However, MTM PnL differs across accounts because entry prices differ (different fill times, different partial fills).

#### MTM Price Selection

| Instrument Type | Long Position Mark | Short Position Mark | Source | Rationale |
|----------------|-------------------|--------------------|---------|----|
| Options (CE/PE) | Best bid | Best ask | Option chain snapshot | Liquidation mark: what you would receive if closing now |
| Futures | LTP | LTP | `LASTTICK:{symbol}` | Futures have tight spreads; LTP is representative |
| Equity | LTP | LTP | `LASTTICK:{symbol}` | Same reasoning as futures |

**Two PnL views:**

| View | Mark Used | Purpose |
|------|-----------|---------|
| Risk PnL | Liquidation mark (table above) | Conservative. Used by Risk Manager for drawdown, kill conditions. |
| Display PnL | Mid-price: `(bid + ask) / 2` | Less noisy. Used in Telegram summaries and Grafana dashboard. |

#### MTM Cycle

The MTM cycle runs every 5 seconds. It iterates over all accounts' positions using the same price snapshot:

```python
async def mark_to_market(self) -> None:
    """
    Update all positions across all accounts with current prices.
    Runs every 5 seconds during market hours.

    Market data is fetched ONCE (shared), then applied to each
    account's positions independently.
    """
    # 1. Get current prices for all instruments with open positions
    instruments = set()
    for pos in self._positions.values():
        instruments.add(pos.security_id)

    price_snapshot: dict[str, MTMPrice] = {}
    for sec_id in instruments:
        tick = await self._redis.get(f"LASTTICK:{self._sec_to_symbol[sec_id]}")
        if tick is None:
            continue
        tick_data = orjson.loads(tick)
        price_snapshot[sec_id] = MTMPrice(
            ltp=tick_data["ltp"],
            bid=tick_data["bid"],
            ask=tick_data["ask"],
            ts=tick_data["exchange_ts"],
        )

    # 2. Apply prices to each position (per-account, independent PnL)
    ts = now_ms()
    for pos in self._positions.values():
        snap = price_snapshot.get(pos.security_id)
        if snap is None:
            continue

        # Select mark based on instrument type and direction
        if pos.is_option:
            if pos.direction == "LONG":
                risk_mark = snap.bid      # liquidation: sell at bid
                display_mark = (snap.bid + snap.ask) / 2
            else:
                risk_mark = snap.ask      # liquidation: buy back at ask
                display_mark = (snap.bid + snap.ask) / 2
        else:
            risk_mark = snap.ltp
            display_mark = snap.ltp

        pos.update_mtm(risk_mark, ts)

    # 3. Update Redis cache per account
    for account_id in self._account_ids:
        account_positions = [
            p for p in self._positions.values()
            if p.account_id == account_id
        ]

        # Per-account NAV
        total_unrealized = sum(p.unrealized_pnl for p in account_positions)
        total_realized = sum(p.realized_pnl for p in account_positions)
        nav = self._account_capitals[account_id] + total_unrealized + total_realized

        await self._redis.set(
            f"CACHE:account_nav:{account_id}",
            orjson.dumps({"nav": nav, "unrealized": total_unrealized,
                         "realized": total_realized, "ts": ts}),
        )

        # Per-account, per-strategy PnL
        strategy_pnl: dict[str, float] = {}
        for p in account_positions:
            strategy_pnl[p.strategy_id] = (
                strategy_pnl.get(p.strategy_id, 0.0) + p.unrealized_pnl
            )
        for sid, pnl in strategy_pnl.items():
            await self._redis.set(
                f"CACHE:strategy_pnl:{account_id}:{sid}",
                orjson.dumps({"unrealized_pnl": pnl, "ts": ts}),
            )

    # 4. Update Prometheus gauges
    for pos in self._positions.values():
        unrealized_pnl_inr.labels(
            strategy=pos.strategy_id,
            account=pos.account_id,
        ).set(pos.unrealized_pnl)

    # 5. Aggregate position for strategy processes (cross-account)
    for sid in self._strategy_ids:
        agg_qty = sum(
            p.quantity for p in self._positions.values()
            if p.strategy_id == sid
        )
        direction = "FLAT" if agg_qty == 0 else ("LONG" if agg_qty > 0 else "SHORT")
        await self._redis.set(
            f"POSITION:strategy:{sid}",
            orjson.dumps({"direction": direction, "quantity": agg_qty,
                         "ts": ts}),
        )


class MTMPrice(pydantic.BaseModel):
    ltp: float
    bid: float
    ask: float
    ts: int
```

---

### Multi-Day Position Restoration

On startup (or after crash recovery), positions must be restored for each account independently. The broker is authoritative for what positions exist; Redis saved state provides strategy attribution.

```python
async def restore_positions(self) -> None:
    """
    Startup position recovery for all accounts.

    For each account:
    1. Query Dhan positions API -> broker truth
    2. Read Redis saved state -> strategy attribution
    3. Merge: broker authoritative for qty/price
    4. Unknown positions -> ORPHAN

    Called once during system startup, before strategies begin.
    """
    for account_id in self._account_ids:
        await self._restore_account_positions(account_id)

    logger.info("position_restore_complete",
               total_positions=len(self._positions),
               accounts=len(self._account_ids))


async def _restore_account_positions(self, account_id: str) -> None:
    """Restore positions for a single account."""
    account = self._accounts[account_id]
    rate_limiter = self._account_rate_limiters[account_id]

    # 1. Fetch broker positions
    await rate_limiter.acquire("POLL")
    broker_positions = await self._dhan_get_positions(account)
    broker_map: dict[str, DhanPosition] = {
        bp.securityId: bp
        for bp in broker_positions
        if bp.netQty != 0
    }

    # 2. Read Redis saved state for this account
    saved_state: dict[str, dict] = {}
    cursor = 0
    while True:
        cursor, keys = await self._redis.scan(
            cursor,
            match=f"POSITION:saved:{account_id}:*",
            count=100,
        )
        for key in keys:
            data = await self._redis.get(key)
            if data:
                parsed = orjson.loads(data)
                saved_state[parsed["security_id"]] = parsed
        if cursor == 0:
            break

    # 3. Merge: broker positions + Redis strategy attribution
    for sec_id, broker_pos in broker_map.items():
        saved = saved_state.get(sec_id)
        broker_avg = (broker_pos.buyAvg
                     if broker_pos.netQty > 0
                     else broker_pos.sellAvg)

        if saved is not None:
            # Known position: use broker qty/price, Redis strategy info
            pos = Position(
                account_id=account_id,
                strategy_id=saved["strategy_id"],
                security_id=sec_id,
                trading_symbol=broker_pos.tradingSymbol,
                underlying=saved.get("underlying", self._infer_underlying(broker_pos)),
                instrument_type=self._infer_instrument_type(broker_pos),
                direction="LONG" if broker_pos.netQty > 0 else "SHORT",
                quantity=broker_pos.netQty,
                avg_entry_price=broker_avg,
                current_price=broker_avg,
                unrealized_pnl=0.0,
                realized_pnl=saved.get("realized_pnl", 0.0),
                entry_ts=saved.get("entry_ts", now_ms()),
                last_update_ts=now_ms(),
                strike=saved.get("strike"),
                expiry=saved.get("expiry"),
                option_type=saved.get("option_type"),
                lot_size=saved.get("lot_size"),
                order_ids=saved.get("order_ids", []),
                sl_order_id=saved.get("sl_order_id"),
                sl_type=saved.get("sl_type"),
                is_overnight=saved.get("is_overnight", False),
            )
            logger.info("position_restored",
                       account_id=account_id,
                       strategy_id=pos.strategy_id,
                       security_id=sec_id,
                       qty=pos.quantity)

        else:
            # Unknown position on broker -> ORPHAN
            pos = Position(
                account_id=account_id,
                strategy_id="ORPHAN",
                security_id=sec_id,
                trading_symbol=broker_pos.tradingSymbol,
                underlying=self._infer_underlying(broker_pos),
                instrument_type=self._infer_instrument_type(broker_pos),
                direction="LONG" if broker_pos.netQty > 0 else "SHORT",
                quantity=broker_pos.netQty,
                avg_entry_price=broker_avg,
                current_price=broker_avg,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                entry_ts=now_ms(),
                last_update_ts=now_ms(),
                is_orphan=True,
            )
            logger.critical("orphan_on_startup",
                          account_id=account_id,
                          security_id=sec_id,
                          trading_symbol=broker_pos.tradingSymbol,
                          qty=broker_pos.netQty)

            await telegram.send(CRITICAL,
                f"ORPHAN ON STARTUP: Account {account_id} "
                f"{broker_pos.tradingSymbol} qty={broker_pos.netQty}. "
                f"Not in saved state. Manual review required.")

        self._positions[pos.position_key] = pos

    # 4. Check for saved positions NOT on broker (closed overnight by broker)
    for sec_id, saved in saved_state.items():
        if sec_id not in broker_map:
            logger.warning("saved_position_not_on_broker",
                          account_id=account_id,
                          security_id=sec_id,
                          strategy_id=saved["strategy_id"],
                          saved_qty=saved.get("quantity", 0))
            # Position was closed by broker (auto-square, SL triggered overnight)
            # Do NOT recreate it. Log for audit.

    # 5. Publish restored positions to Redis
    await self._publish_all_positions(account_id)


async def save_positions(self) -> None:
    """
    Save all position state to Redis for crash recovery.
    Called during graceful shutdown (Phase 4) and periodically every 60s.
    """
    for pos in self._positions.values():
        key = f"POSITION:saved:{pos.account_id}:{pos.security_id}"
        await self._redis.set(
            key,
            orjson.dumps(pos.model_dump(mode="json")),
            ex=86400 * 3,  # 3-day TTL; stale state is worse than no state
        )
```

---

### Aggregate Views (Monitoring)

Aggregate views combine data across accounts for portfolio-level monitoring. These are computed in the MTM cycle and published to Redis for Grafana dashboards.

```python
async def compute_aggregate_views(self) -> AggregateView:
    """
    Compute portfolio-level aggregates across all accounts.
    Called after every MTM cycle.
    """
    account_navs: dict[str, float] = {}
    total_unrealized = 0.0
    total_realized = 0.0
    total_exposure = 0.0
    strategy_pnl: dict[str, float] = {}  # strategy -> sum across accounts

    for account_id in self._account_ids:
        acct_positions = [
            p for p in self._positions.values()
            if p.account_id == account_id
        ]
        acct_unrealized = sum(p.unrealized_pnl for p in acct_positions)
        acct_realized = sum(p.realized_pnl for p in acct_positions)
        nav = self._account_capitals[account_id] + acct_unrealized + acct_realized
        account_navs[account_id] = nav
        total_unrealized += acct_unrealized
        total_realized += acct_realized
        total_exposure += sum(p.notional_value for p in acct_positions)

        for p in acct_positions:
            strategy_pnl[p.strategy_id] = (
                strategy_pnl.get(p.strategy_id, 0.0)
                + p.unrealized_pnl + p.realized_pnl
            )

    total_aum = sum(account_navs.values())

    view = AggregateView(
        total_aum=total_aum,
        total_unrealized_pnl=total_unrealized,
        total_realized_pnl=total_realized,
        total_exposure=total_exposure,
        account_navs=account_navs,
        strategy_pnl=strategy_pnl,
        ts=now_ms(),
    )

    # Publish to Redis for Grafana
    await self._redis.set("CACHE:aggregate_view",
                          orjson.dumps(view.model_dump()))

    # Update Prometheus
    portfolio_drawdown_pct.set(self._compute_portfolio_drawdown(total_aum))
    for sid, pnl in strategy_pnl.items():
        realized_pnl_inr.labels(strategy=sid).set(pnl)

    return view


class AggregateView(pydantic.BaseModel):
    total_aum: float                         # sum of all account NAVs
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_exposure: float                    # sum of notional across all positions
    account_navs: dict[str, float]           # account_id -> NAV
    strategy_pnl: dict[str, float]           # strategy_id -> total PnL across accounts
    ts: int
```

**Per-account NAV for PMS reporting:** Each account's NAV is tracked independently in `CACHE:account_nav:{account_id}`. This enables per-client reporting where each account owner sees only their own performance, while the system operator sees the aggregate.

---

### Greeks Computation

Greeks are computed per position using Black-76 (for exchange-traded options on futures-settled underlyings). Market data inputs are shared; output Greeks are per-position because strike, expiry, and quantity differ.

```python
async def compute_greeks(self) -> None:
    """
    Compute Greeks for all option positions across all accounts.
    Runs every 30 seconds during market hours.

    Inputs (shared across accounts):
    - Spot price: LASTTICK for underlying
    - IV: from option chain snapshot or BSM inversion
    - Risk-free rate: from config (6.5%, updated quarterly)

    Outputs (per position):
    - delta, gamma, theta, vega stored on Position object
    """
    spot_cache: dict[str, float] = {}  # underlying -> spot
    iv_cache: dict[str, float] = {}    # security_id -> IV

    for pos in self._positions.values():
        if not pos.is_option:
            continue

        # Get spot (cached per underlying, shared across accounts)
        if pos.underlying not in spot_cache:
            tick = await self._redis.get(f"LASTTICK:{pos.underlying}")
            if tick is None:
                continue
            spot_cache[pos.underlying] = orjson.loads(tick)["ltp"]
        spot = spot_cache[pos.underlying]

        # Get IV (cached per security_id, shared across accounts)
        if pos.security_id not in iv_cache:
            iv = await self._get_iv(pos.security_id, spot,
                                     pos.strike, pos.expiry,
                                     pos.option_type)
            if iv is None:
                continue
            iv_cache[pos.security_id] = iv
        iv = iv_cache[pos.security_id]

        # Time to expiry in years
        tte = self._time_to_expiry_years(pos.expiry)
        if tte <= 0:
            # Expired option — set Greeks to terminal values
            pos.delta = 1.0 if pos.option_type == "CE" and spot > pos.strike else 0.0
            pos.gamma = 0.0
            pos.theta = 0.0
            pos.vega = 0.0
            pos.iv = 0.0
            continue

        r = self._config.risk_free_rate  # 0.065

        # Black-76 Greeks
        F = spot * math.exp(r * tte)  # forward price
        d1 = (math.log(F / pos.strike) + 0.5 * iv**2 * tte) / (iv * math.sqrt(tte))
        d2 = d1 - iv * math.sqrt(tte)

        discount = math.exp(-r * tte)
        n_d1 = norm.cdf(d1)
        n_d2 = norm.cdf(d2)
        n_prime_d1 = norm.pdf(d1)

        if pos.option_type == "CE":
            pos.delta = discount * n_d1
            pos.theta = (
                -(spot * n_prime_d1 * iv) / (2 * math.sqrt(tte))
                - r * pos.strike * discount * n_d2
            ) / 365  # per-day theta
        else:  # PE
            pos.delta = discount * (n_d1 - 1)
            pos.theta = (
                -(spot * n_prime_d1 * iv) / (2 * math.sqrt(tte))
                + r * pos.strike * discount * (1 - n_d2)
            ) / 365

        pos.gamma = (discount * n_prime_d1) / (spot * iv * math.sqrt(tte))
        pos.vega = spot * discount * n_prime_d1 * math.sqrt(tte) / 100  # per 1% IV
        pos.iv = iv

        # Scale by quantity for portfolio Greeks
        # (Individual Greeks are per-unit; portfolio aggregation uses qty)

    # Portfolio delta exposure check
    total_delta_lots = sum(
        (p.delta or 0) * p.quantity / (p.lot_size or 1)
        for p in self._positions.values()
        if p.is_option
    )
    delta_exposure.set(total_delta_lots)

    if abs(total_delta_lots) > 50:
        logger.warning("delta_exposure_high",
                      delta_lots=total_delta_lots)
        await telegram.send(WARNING,
            f"DELTA EXPOSURE: Portfolio net delta = {total_delta_lots:.1f} "
            f"NIFTY lot equivalents (threshold: +/-50)")
```

---

### PositionTracker Class

```python
class PositionTracker:
    """
    Central position management for all accounts.

    Single async process. Sole writer to DuckDB. Manages per-account
    positions, reconciliation, MTM, Greeks, and aggregate views.

    Position key: (account_id, security_id, strategy_id)
    """

    def __init__(
        self,
        accounts: list["Account"],
        redis: aioredis.Redis,
        duckdb_path: str,
        config: "TradingConfig",
    ):
        self._accounts: dict[str, "Account"] = {
            a.account_id: a for a in accounts
        }
        self._account_ids: list[str] = [a.account_id for a in accounts]
        self._redis = redis
        self._db = duckdb.connect(duckdb_path)
        self._config = config

        # Positions: (account_id, security_id, strategy_id) -> Position
        self._positions: dict[tuple[str, str, str], Position] = {}

        # Per-account rate limiters (shared with OMS via reference)
        self._account_rate_limiters: dict[str, "PriorityRateLimiter"] = {}

        # Per-account capital (loaded from config, updated daily)
        self._account_capitals: dict[str, float] = {
            a.account_id: a.initial_capital for a in accounts
        }

        # Symbol mapping caches
        self._sec_to_symbol: dict[str, str] = {}

        # Strategy IDs (derived from config)
        self._strategy_ids: list[str] = config.strategy_ids

        # Initialize DuckDB tables for all accounts
        self._init_tables()

    def _init_tables(self) -> None:
        """Create DuckDB tables for each account if they don't exist."""
        for account_id in self._account_ids:
            self._db.execute(f"""
                CREATE TABLE IF NOT EXISTS trades_{account_id} (
                    trade_id VARCHAR PRIMARY KEY,
                    strategy_id VARCHAR NOT NULL,
                    signal_id VARCHAR NOT NULL,
                    order_id VARCHAR NOT NULL,
                    exchange_order_id VARCHAR,
                    security_id VARCHAR NOT NULL,
                    trading_symbol VARCHAR NOT NULL,
                    underlying VARCHAR NOT NULL,
                    instrument_type VARCHAR NOT NULL,
                    strike DOUBLE,
                    expiry DATE,
                    option_type VARCHAR,
                    transaction_type VARCHAR NOT NULL,
                    trade_type VARCHAR NOT NULL,
                    quantity INTEGER NOT NULL,
                    price DOUBLE NOT NULL,
                    notional_value DOUBLE NOT NULL,
                    brokerage DOUBLE NOT NULL DEFAULT 0,
                    stt DOUBLE NOT NULL DEFAULT 0,
                    exchange_fees DOUBLE NOT NULL DEFAULT 0,
                    sebi_fee DOUBLE NOT NULL DEFAULT 0,
                    gst DOUBLE NOT NULL DEFAULT 0,
                    stamp_duty DOUBLE NOT NULL DEFAULT 0,
                    total_cost DOUBLE NOT NULL DEFAULT 0,
                    exchange_ts BIGINT NOT NULL,
                    local_ts BIGINT NOT NULL,
                    trade_date DATE NOT NULL,
                    entry_price DOUBLE,
                    realized_pnl DOUBLE,
                    spot_at_fill DOUBLE,
                    iv_at_fill DOUBLE
                )
            """)
            self._db.execute(f"""
                CREATE TABLE IF NOT EXISTS daily_returns_{account_id} (
                    trade_date DATE NOT NULL,
                    strategy_id VARCHAR NOT NULL,
                    gross_pnl DOUBLE NOT NULL,
                    total_costs DOUBLE NOT NULL,
                    net_pnl DOUBLE NOT NULL,
                    capital_deployed DOUBLE NOT NULL,
                    net_return DOUBLE NOT NULL,
                    num_trades INTEGER NOT NULL,
                    num_winners INTEGER NOT NULL,
                    num_losers INTEGER NOT NULL,
                    max_intraday_dd DOUBLE NOT NULL,
                    end_of_day_position VARCHAR NOT NULL,
                    PRIMARY KEY (trade_date, strategy_id)
                )
            """)
            self._db.execute(f"""
                CREATE TABLE IF NOT EXISTS realized_pnl_{account_id} (
                    position_id VARCHAR PRIMARY KEY,
                    strategy_id VARCHAR NOT NULL,
                    security_id VARCHAR NOT NULL,
                    trading_symbol VARCHAR NOT NULL,
                    underlying VARCHAR NOT NULL,
                    instrument_type VARCHAR NOT NULL,
                    direction VARCHAR NOT NULL,
                    entry_price DOUBLE NOT NULL,
                    exit_price DOUBLE NOT NULL,
                    quantity INTEGER NOT NULL,
                    gross_pnl DOUBLE NOT NULL,
                    total_costs DOUBLE NOT NULL,
                    net_pnl DOUBLE NOT NULL,
                    entry_ts BIGINT NOT NULL,
                    exit_ts BIGINT NOT NULL,
                    entry_date DATE NOT NULL,
                    exit_date DATE NOT NULL,
                    holding_period_ms BIGINT NOT NULL,
                    exit_reason VARCHAR NOT NULL,
                    entry_order_ids VARCHAR NOT NULL,
                    exit_order_ids VARCHAR NOT NULL,
                    spot_at_entry DOUBLE,
                    spot_at_exit DOUBLE,
                    iv_at_entry DOUBLE,
                    iv_at_exit DOUBLE
                )
            """)
            self._db.execute(f"""
                CREATE TABLE IF NOT EXISTS reconciliation_{account_id} (
                    recon_id VARCHAR PRIMARY KEY,
                    recon_type VARCHAR NOT NULL,
                    recon_ts BIGINT NOT NULL,
                    status VARCHAR NOT NULL,
                    security_id VARCHAR,
                    local_qty INTEGER,
                    broker_qty INTEGER,
                    local_avg_price DOUBLE,
                    broker_avg_price DOUBLE,
                    action_taken VARCHAR,
                    details VARCHAR
                )
            """)

    # --- Fill handling ---

    async def on_fill(self, fill: "OrderResult") -> None:
        """
        Process a completed fill from the OMS.

        Creates or updates position for the specific account.
        Writes TradeRecord to DuckDB.
        Updates Redis cache.
        """
        account_id = fill.account_id
        key = (account_id, fill.security_id, fill.strategy_id)

        if fill.trade_type in ("ENTRY", "PARTIAL_ENTRY"):
            if key in self._positions:
                # Add to existing position (scale-in)
                pos = self._positions[key]
                old_notional = pos.avg_entry_price * abs(pos.quantity)
                new_notional = fill.avg_fill_price * fill.filled_qty
                total_qty = abs(pos.quantity) + fill.filled_qty
                pos.avg_entry_price = (old_notional + new_notional) / total_qty
                pos.quantity = total_qty if pos.direction == "LONG" else -total_qty
                pos.order_ids.append(fill.order_id)
                pos.last_update_ts = now_ms()
            else:
                # New position
                pos = Position(
                    account_id=account_id,
                    strategy_id=fill.strategy_id,
                    security_id=fill.security_id,
                    trading_symbol=fill.trading_symbol,
                    underlying=fill.underlying,
                    instrument_type=fill.instrument_type,
                    direction=fill.direction,
                    quantity=(fill.filled_qty if fill.direction == "LONG"
                             else -fill.filled_qty),
                    avg_entry_price=fill.avg_fill_price,
                    current_price=fill.avg_fill_price,
                    unrealized_pnl=0.0,
                    realized_pnl=0.0,
                    entry_ts=now_ms(),
                    last_update_ts=now_ms(),
                    strike=fill.strike,
                    expiry=fill.expiry,
                    option_type=fill.option_type,
                    lot_size=fill.lot_size,
                    order_ids=[fill.order_id],
                    sl_order_id=fill.sl_order_id,
                    sl_type=fill.sl_type,
                    is_overnight=fill.strategy_id in ("S2", "S6", "S7"),
                )
                self._positions[key] = pos

        elif fill.trade_type in ("EXIT", "SL_TRIGGERED", "EOD_FLATTEN",
                                  "EMERGENCY_EXIT", "PARTIAL_EXIT"):
            pos = self._positions.get(key)
            if pos is not None:
                # Calculate realized PnL
                if pos.direction == "LONG":
                    rpnl = (fill.avg_fill_price - pos.avg_entry_price) * fill.filled_qty
                else:
                    rpnl = (pos.avg_entry_price - fill.avg_fill_price) * fill.filled_qty

                remaining = abs(pos.quantity) - fill.filled_qty
                if remaining <= 0:
                    # Position fully closed
                    pos.realized_pnl += rpnl
                    self._write_realized_pnl(pos, fill, rpnl)
                    del self._positions[key]
                else:
                    # Partial exit
                    pos.quantity = remaining if pos.direction == "LONG" else -remaining
                    pos.realized_pnl += rpnl
                    pos.last_update_ts = now_ms()

        # Write trade record to DuckDB
        trade = self._build_trade_record(fill)
        self.write_trade(trade)

        # Update Redis caches
        await self._publish_position_update(account_id, fill.strategy_id)

        # Update Prometheus
        open_positions.labels(
            strategy=fill.strategy_id,
            account=account_id,
        ).set(self._count_positions(account_id, fill.strategy_id))

    async def on_partial_fill(self, state: "OrderState") -> None:
        """Handle partial fill notification. SL qty sync handled by OMS/SLManager."""
        # Position update deferred until manage_order loop completes.
        # This method updates interim state for MTM accuracy.
        key = (state.account_id, state.security_id, state.strategy_id)
        pos = self._positions.get(key)
        if pos is not None:
            pos.last_update_ts = now_ms()

    def get_strategy_position(
        self,
        strategy_id: str,
        account_id: str | None = None,
    ) -> str:
        """
        Get position direction for a strategy.

        If account_id is None, returns aggregate across all accounts.
        """
        positions = [
            p for p in self._positions.values()
            if p.strategy_id == strategy_id
            and (account_id is None or p.account_id == account_id)
        ]
        total_qty = sum(p.quantity for p in positions)
        if total_qty == 0:
            return "FLAT"
        return "LONG" if total_qty > 0 else "SHORT"

    def get_all_open(self, account_id: str | None = None) -> list[Position]:
        """Get all open positions, optionally filtered by account."""
        if account_id is None:
            return list(self._positions.values())
        return [p for p in self._positions.values()
                if p.account_id == account_id]

    def get_overnight_positions(self, account_id: str) -> list[Position]:
        """Get overnight positions for a specific account."""
        return [
            p for p in self._positions.values()
            if p.account_id == account_id and p.is_overnight
        ]

    # --- DuckDB write methods (sole writer) ---

    def write_trade(self, trade: TradeRecord) -> None:
        """Write a single trade record to the account's trades table."""
        table = f"trades_{trade.account_id}"
        self._db.execute(f"""
            INSERT INTO {table} VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            trade.trade_id, trade.strategy_id, trade.signal_id,
            trade.order_id, trade.exchange_order_id,
            trade.security_id, trade.trading_symbol, trade.underlying,
            trade.instrument_type, trade.strike, trade.expiry,
            trade.option_type, trade.transaction_type, trade.trade_type,
            trade.quantity, trade.price, trade.notional_value,
            trade.brokerage, trade.stt, trade.exchange_fees,
            trade.sebi_fee, trade.gst, trade.stamp_duty, trade.total_cost,
            trade.exchange_ts, trade.local_ts, trade.trade_date,
            trade.entry_price, trade.realized_pnl, trade.spot_at_fill,
            trade.iv_at_fill,
        ])

    def write_daily_pnl(
        self,
        account_id: str,
        strategy_id: str,
        d: date,
        summary: "DailySummary",
    ) -> None:
        """Write daily PnL summary. Called at EOD for each strategy per account."""
        table = f"daily_returns_{account_id}"
        self._db.execute(f"""
            INSERT OR REPLACE INTO {table} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            d, strategy_id, summary.gross_pnl, summary.total_costs,
            summary.net_pnl, summary.capital_deployed, summary.net_return,
            summary.num_trades, summary.num_winners, summary.num_losers,
            summary.max_intraday_dd, summary.end_of_day_position,
        ])

        # Update Redis cache
        asyncio.create_task(self._update_daily_returns_cache(
            account_id, strategy_id))

    def get_daily_returns(
        self,
        account_id: str,
        strategy_id: str,
        lookback: int,
    ) -> list[float]:
        """
        Get recent daily net returns for a strategy in an account.
        Used by Capital Allocator for ERC computation.
        """
        table = f"daily_returns_{account_id}"
        result = self._db.execute(f"""
            SELECT net_return FROM {table}
            WHERE strategy_id = ?
            ORDER BY trade_date DESC
            LIMIT ?
        """, [strategy_id, lookback]).fetchall()
        return [row[0] for row in reversed(result)]

    async def _update_daily_returns_cache(
        self,
        account_id: str,
        strategy_id: str,
    ) -> None:
        """Push daily returns to Redis cache for Capital Allocator."""
        returns = self.get_daily_returns(account_id, strategy_id, lookback=252)
        await self._redis.set(
            f"CACHE:daily_returns:{account_id}:{strategy_id}",
            orjson.dumps(returns),
            ex=86400,  # 1-day TTL
        )

        # Also write an aggregated RETURNS:{sid}:daily Redis LIST (shared,
        # not per-account) by summing weighted returns across all accounts.
        # This is the key the Capital Allocator's ERC computation reads.
        await self._update_aggregated_returns(strategy_id)

    async def _publish_position_update(
        self,
        account_id: str,
        strategy_id: str,
    ) -> None:
        """Publish position state to Redis for other components."""
        # Per-account position
        positions = [
            p for p in self._positions.values()
            if p.account_id == account_id and p.strategy_id == strategy_id
        ]
        total_qty = sum(p.quantity for p in positions)
        direction = "FLAT" if total_qty == 0 else ("LONG" if total_qty > 0 else "SHORT")

        await self._redis.set(
            f"POSITION:account:{account_id}:{strategy_id}",
            orjson.dumps({
                "direction": direction,
                "quantity": total_qty,
                "unrealized_pnl": sum(p.unrealized_pnl for p in positions),
                "ts": now_ms(),
            }),
        )

        # Aggregate position (cross-account, for strategy processes)
        all_positions = [
            p for p in self._positions.values()
            if p.strategy_id == strategy_id
        ]
        agg_qty = sum(p.quantity for p in all_positions)
        agg_dir = "FLAT" if agg_qty == 0 else ("LONG" if agg_qty > 0 else "SHORT")

        await self._redis.set(
            f"POSITION:strategy:{strategy_id}",
            orjson.dumps({
                "direction": agg_dir,
                "quantity": agg_qty,
                "ts": now_ms(),
            }),
        )

    async def _publish_all_positions(self, account_id: str) -> None:
        """Publish all positions for an account to Redis."""
        strategy_ids = set(
            p.strategy_id for p in self._positions.values()
            if p.account_id == account_id
        )
        for sid in strategy_ids:
            await self._publish_position_update(account_id, sid)

    # --- Async event loop ---

    async def run(self) -> None:
        """
        Main event loop. Runs concurrently:
        1. MTM every 5s
        2. Greeks every 30s
        3. Quick recon every 5 min per account (staggered)
        4. Position save every 60s
        5. Fill processing (event-driven via Redis subscription)
        """
        await self.restore_positions()

        tasks = [
            asyncio.create_task(self._mtm_loop()),
            asyncio.create_task(self._greeks_loop()),
            asyncio.create_task(self._recon_loop()),
            asyncio.create_task(self._save_loop()),
            asyncio.create_task(self._fill_listener()),
        ]
        await asyncio.gather(*tasks)

    async def _mtm_loop(self) -> None:
        while True:
            try:
                await self.mark_to_market()
                await self.compute_aggregate_views()
            except Exception as e:
                logger.error("mtm_error", error=str(e))
            await asyncio.sleep(5)

    async def _greeks_loop(self) -> None:
        while True:
            try:
                await self.compute_greeks()
            except Exception as e:
                logger.error("greeks_error", error=str(e))
            await asyncio.sleep(30)

    async def _recon_loop(self) -> None:
        while True:
            for i, account_id in enumerate(self._account_ids):
                try:
                    await self.reconcile_account(account_id, "QUICK")
                except Exception as e:
                    logger.error("recon_error",
                               account_id=account_id, error=str(e))
                await asyncio.sleep(0.5)  # stagger 500ms between accounts
            await asyncio.sleep(300)  # 5 minutes

    async def _save_loop(self) -> None:
        while True:
            try:
                await self.save_positions()
            except Exception as e:
                logger.error("save_error", error=str(e))
            await asyncio.sleep(60)

    async def _fill_listener(self) -> None:
        """Listen for fill events from OMS via Redis pub/sub."""
        pubsub = self._redis.pubsub()
        await pubsub.subscribe("CHANNEL:FILLS")
        async for message in pubsub.listen():
            if message["type"] == "message":
                fill_data = orjson.loads(message["data"])
                fill = OrderResult(**fill_data)
                await self.on_fill(fill)
```

---

### State Table

Every piece of state is either per-account or shared. This table is the definitive reference.

| State | Scope | Redis Key Pattern | Updated By | Read By |
|-------|-------|-------------------|------------|---------|
| Position (live) | Per-account | `POSITION:account:{account_id}:{strategy_id}` | PositionTracker | Risk Manager, OMS |
| Position (aggregated) | Shared (cross-account) | `POSITION:strategy:{strategy_id}` | PositionTracker | Strategy processes |
| Position (saved for recovery) | Per-account | `POSITION:saved:{account_id}:{security_id}` | PositionTracker | PositionTracker (startup) |
| Daily returns cache | Per-account | `CACHE:daily_returns:{account_id}:{strategy_id}` | PositionTracker | Capital Allocator |
| Aggregated daily returns | Shared | `RETURNS:{sid}:daily` (Redis LIST) | PositionTracker (weighted sum across all accounts) | Capital Allocator (ERC computation) |
| Strategy PnL cache | Per-account | `CACHE:strategy_pnl:{account_id}:{strategy_id}` | PositionTracker | Risk Manager |
| Account NAV | Per-account | `CACHE:account_nav:{account_id}` | PositionTracker | Monitoring |
| Aggregate view | Shared | `CACHE:aggregate_view` | PositionTracker | Grafana |
| Market data (ticks) | Shared | `LASTTICK:{symbol}` | Data Ingester | PositionTracker (MTM) |
| Option chain | Shared | `CHAIN:{underlying}:{expiry}` | Instrument Resolution (background poller) | PositionTracker (Greeks) |
| DuckDB trades | Per-account | N/A (file, table per account) | PositionTracker | PositionTracker only |
| DuckDB daily returns | Per-account | N/A (file, table per account) | PositionTracker | PositionTracker only |
| DuckDB realized PnL | Per-account | N/A (file, table per account) | PositionTracker | PositionTracker only |
| DuckDB reconciliation | Per-account | N/A (file, table per account) | PositionTracker | PositionTracker only |
| Prometheus metrics | Labels include account_id | N/A (scrape endpoint) | PositionTracker | Grafana |

---

### Failure Modes

| # | Scenario | Detection | Impact | Recovery | Severity |
|---|----------|-----------|--------|----------|----------|
| 1 | DuckDB file locked by stale process | Write timeout (>5s) | Trades not persisted, cache stale | Kill stale process. Replay fills from audit log. | CRITICAL |
| 2 | DuckDB file corrupted | Read/write exception on startup | All historical data lost | Restore from S3 nightly backup. Replay today's fills from audit log. RPO: 1 day. | CRITICAL |
| 3 | Redis unavailable | ConnectionError on cache write | MTM cache stale. Other components read stale data. Positions still tracked in-memory. | Sentinel failover (<1s). If persistent: positions protected by server-side SLs. | HIGH |
| 4 | Dhan positions API returns error | HTTP 4xx/5xx on recon call | Reconciliation skipped for one cycle | Retry next cycle (5 min). If 3 consecutive failures: Telegram CRITICAL. | MEDIUM |
| 5 | Dhan positions API returns stale data | Qty mismatch persists across multiple recon cycles | False discrepancy alerts | After 3 consecutive mismatches, force sync from broker (broker is truth). | MEDIUM |
| 6 | Fill notification lost (WS drop + REST miss) | Post-order timer fires, broker shows TRADED | Local state missing a position. Ghost risk. | force_sync_from_broker catches it. Pre-order broker position check prevents second entry. | CRITICAL |
| 7 | MTM price stale (no tick for >30s) | Tick staleness check in MTM loop | PnL and risk calculations use stale prices | Use last known price. Flag stale instruments. Skip Greeks computation for stale prices. | MEDIUM |
| 8 | Greeks computation error (IV solve fails) | Exception in BSM inversion | Greeks set to None for that position. Delta exposure potentially underestimated. | Use previous Greeks. Log warning. Do not block MTM. | LOW |
| 9 | Account API key expired mid-session | 401 on any Dhan API call for that account | Cannot reconcile, cannot sync that account | Telegram CRITICAL. Other accounts unaffected. Existing SLs still active. | HIGH |
| 10 | Orphan position on startup | Broker shows position not in saved state | Unknown position with no strategy attribution | Create ORPHAN position. Telegram CRITICAL. Manual review. Do not auto-close. | HIGH |
| 11 | Position saved to Redis but DuckDB write failed | DuckDB exception after Redis publish | Redis shows position, DuckDB missing trade record | On next startup: audit log replay fills any DuckDB gaps. | MEDIUM |
| 12 | Concurrent startup (two PositionTracker processes) | DuckDB file lock contention | Second process blocks or corrupts | PID file check on startup. If PID file exists and process alive: abort with error. | CRITICAL |
| 13 | System clock drift >100ms | Chrony alert | Timestamps unreliable for ordering, IV computation skewed | Alert ops. Do not halt trading. Exchange timestamps remain authoritative. | LOW |
| 14 | Memory pressure (too many positions in-memory) | RSS monitoring via Prometheus | OOM kill risk | Position count should never exceed ~50 across all accounts. If >100: investigate. | LOW |

---

### Edge Cases

**1. Same instrument, same strategy, two accounts, different directions.**
Account A is LONG NIFTY 24500 CE for S1 (filled earlier), Account B is SHORT the same (filled after reversal signal, A's exit failed). This is a valid state. Each position tracked independently. Aggregate view for S1 may show FLAT even though individual accounts are not.

**2. Partial fill on one account, full fill on another.**
Signal for S1 goes to accounts A, B, C. A fills 10 lots, B fills 6 lots (partial, remainder abandoned), C fills 0 (rejected). Positions: A has 10 lots, B has 6 lots, C has nothing. Daily returns diverge. Divergence tracked by DivergenceTracker (OMS component) and reflected in per-account DuckDB tables.

**3. Orphan position at startup with negative PnL.**
Broker shows -200k position for Account B that is not in saved state. Position created as ORPHAN. No auto-close (could make losses worse). Manual review via Telegram alert. Operator can assign to a strategy or close manually.

**4. EOD flatten partially completes, then system crashes.**
Account A: 3 of 5 positions flattened. System crashes. On restart: broker shows 2 remaining positions for A. Restored as overnight (they survived session close via Dhan's auto-square timeout). Reconciliation at 09:16 next day picks them up. If they were intraday strategies, operator notified.

**5. DuckDB backup running during EOD write burst.**
S3 backup at 16:30 reads DuckDB while EOD summaries are being written. DuckDB supports concurrent reads, so this is safe. The backup may capture a partial EOD state, but the next backup captures everything. Backup is a snapshot of the file; DuckDB's ACID guarantees mean the snapshot is always consistent.

**6. Account removed from config between sessions.**
On startup, the PositionTracker does not create tables for the removed account. But if broker still shows positions for that account (not yet closed), the account is not queried because it is not in the config. Resolution: operator must close all positions for the removed account before removing it from config. Config validation checks this.

**7. Strategy reassignment (position moves from S1 to S3).**
Not supported automatically. If needed: close position under S1, open under S3. The close and open are separate trades with separate cost impact. Cross-strategy position transfer is not a thing.

**8. Expiry day: option expires worthless while position still open.**
On expiry day, NSE auto-exercises/expires options at settlement. The position disappears from broker at EOD. Next morning reconciliation at 09:16 detects MISSING_BROKER for the expired option. If it expired worthless: realize full loss of premium. If it expired ITM: exchange settlement applies; realized PnL computed from settlement price. The PositionTracker writes a trade record with trade_type "EXPIRY_SETTLEMENT".

**9. Split fill across multiple exchanges.**
NSE only in v1. Not applicable. If BSE added in v2, security_id is exchange-specific, so positions are naturally separate.

**10. Negative avg_entry_price after cost adjustment.**
Not possible. avg_entry_price is the raw fill price. Costs are tracked separately in TradeRecord. PnL computation: `(exit_price - entry_price) * qty - total_costs`. Entry price is never adjusted for costs.

---

### Concurrency Model

The PositionTracker runs as a single async process (`asyncio` event loop) within one OS process. All account operations execute within this single event loop, sequentially for DuckDB writes and concurrently for Redis reads.

```
┌─────────────────────────────────────────────────────────────────┐
│  PositionTracker Process (single asyncio event loop)           │
│                                                                 │
│  Concurrent tasks (asyncio.gather):                             │
│  ├── _mtm_loop          (every 5s, iterates ALL accounts)       │
│  ├── _greeks_loop       (every 30s, iterates ALL accounts)      │
│  ├── _recon_loop        (every 5min, per-account, staggered)    │
│  ├── _save_loop         (every 60s, iterates ALL accounts)      │
│  └── _fill_listener     (event-driven, processes fills FIFO)    │
│                                                                 │
│  DuckDB writes:  SEQUENTIAL (one write at a time)               │
│  Redis reads:    CONCURRENT (asyncio, non-blocking)             │
│  Redis writes:   SEQUENTIAL per key (no race, single writer)    │
│  Dhan API calls: SEQUENTIAL per account (rate limiter)          │
│                  CONCURRENT across accounts (independent OPS)   │
│                                                                 │
│  Fill processing order:                                         │
│  1. fill_listener receives fill event                           │
│  2. on_fill() updates in-memory Position                        │
│  3. write_trade() inserts to DuckDB (sync, blocks event loop    │
│     for ~1ms per insert)                                        │
│  4. _publish_position_update() writes to Redis (async)          │
│  5. Next fill processed only after step 4 completes             │
│                                                                 │
│  Why single process, not one per account:                       │
│  - DuckDB file lock prevents concurrent writers anyway          │
│  - Single process simplifies aggregate view computation         │
│  - N accounts x 5-10 positions = 15-50 positions total          │
│  - MTM loop for 50 positions: <1ms. No parallelism needed.      │
│  - Recon API calls are I/O bound; asyncio handles concurrency   │
└─────────────────────────────────────────────────────────────────┘
```

**DuckDB write latency:** A single INSERT into a DuckDB table takes ~0.5-1ms on NVMe SSD. With peak fill rate of 10 fills/second (across all accounts during EOD flatten), DuckDB writes consume ~10ms/s of event loop time. This leaves >99% of the event loop available for MTM, Greeks, and Redis operations.

**Scaling limit:** At 10+ accounts with 20+ positions each, the MTM loop iterating 200+ positions may take >5ms per cycle. If this becomes a bottleneck, the MTM loop can be moved to a thread pool executor (`loop.run_in_executor`), keeping DuckDB writes on the main thread. This is a v2 optimization; v1 targets 3-5 accounts.

---


## Component 9: Audit & Compliance Logger

### Responsibility

- Structured JSON logging (structlog) of every tradable event across all accounts
- Per-account log isolation for PMS client reporting and regulatory queries
- 5-year S3 retention per SEBI record-keeping requirements for portfolio managers
- Signal context capture for complete decision reproduction (spot price, VIX, indicators, bar data, strategy parameters)
- DuckDB-backed local query layer for fast intraday and historical analysis
- Concurrent, non-blocking log writes that never stall the trading hot path
- Multi-account event routing: every event tagged with `account_id`, stored in per-account log partitions

---

### AuditEvent Model

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import Any
import uuid
import datetime
import os


class EventType(str, Enum):
    """All audit event types.

    Signals:
        SIGNAL_GENERATED    — strategy produced a tradable signal
        SIGNAL_REJECTED     — signal failed pre-trade risk checks
        SIGNAL_DEDUPED      — duplicate signal suppressed within dedup window

    Orders:
        ORDER_PLACED        — order submitted to broker API
        ORDER_MODIFIED      — order price/qty modified on exchange
        ORDER_CANCELLED     — order cancelled before fill
        ORDER_FILLED        — order fully filled
        ORDER_PARTIAL_FILL  — order partially filled (interim event)
        ORDER_REJECTED      — broker or exchange rejected the order

    Stop-Loss:
        SL_PLACED           — stop-loss order placed on exchange
        SL_MODIFIED         — SL trigger/limit price updated
        SL_TRIGGERED        — SL hit and converted to market order
        SL_CANCELLED        — SL cancelled (position closed or replaced)
        SL_VERIFICATION_FAILED — SL verification loop detected missing/wrong SL

    Positions:
        POSITION_OPENED     — new position created from fill
        POSITION_CLOSED     — position fully exited
        POSITION_RECONCILED — position state synced with broker after discrepancy

    Risk:
        RISK_CHECK_FAILED   — pre-trade or portfolio risk check blocked an action
        KILL_TRIGGERED      — single strategy killed by risk manager
        GLOBAL_KILL         — global kill switch activated, all strategies halted
        MARGIN_CALL         — broker margin call received, forced liquidation may follow

    System:
        SYSTEM_STARTUP      — trading system process started
        SYSTEM_SHUTDOWN     — trading system process stopped (clean or forced)
        WS_DISCONNECT       — WebSocket connection lost (data or order feed)
        WS_RECONNECT        — WebSocket connection re-established

    Account:
        ACCOUNT_SUSPENDED   — account removed from active trading (risk breach or manual)
        ACCOUNT_RESUMED     — suspended account restored to active trading
        ACCOUNT_AUTH_FAILED  — broker authentication failed for this account
        DIVERGENCE_ALERT    — fill divergence detected between accounts for same signal
    """

    # Signals
    SIGNAL_GENERATED = "SIGNAL_GENERATED"
    SIGNAL_REJECTED = "SIGNAL_REJECTED"
    SIGNAL_DEDUPED = "SIGNAL_DEDUPED"

    # Orders
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_MODIFIED = "ORDER_MODIFIED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_PARTIAL_FILL = "ORDER_PARTIAL_FILL"
    ORDER_REJECTED = "ORDER_REJECTED"

    # Stop-Loss
    SL_PLACED = "SL_PLACED"
    SL_MODIFIED = "SL_MODIFIED"
    SL_TRIGGERED = "SL_TRIGGERED"
    SL_CANCELLED = "SL_CANCELLED"
    SL_VERIFICATION_FAILED = "SL_VERIFICATION_FAILED"

    # Positions
    POSITION_OPENED = "POSITION_OPENED"
    POSITION_CLOSED = "POSITION_CLOSED"
    POSITION_RECONCILED = "POSITION_RECONCILED"

    # Risk
    RISK_CHECK_FAILED = "RISK_CHECK_FAILED"
    KILL_TRIGGERED = "KILL_TRIGGERED"
    GLOBAL_KILL = "GLOBAL_KILL"
    MARGIN_CALL = "MARGIN_CALL"

    # System
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    WS_DISCONNECT = "WS_DISCONNECT"
    WS_RECONNECT = "WS_RECONNECT"

    # Account
    ACCOUNT_SUSPENDED = "ACCOUNT_SUSPENDED"
    ACCOUNT_RESUMED = "ACCOUNT_RESUMED"
    ACCOUNT_AUTH_FAILED = "ACCOUNT_AUTH_FAILED"
    DIVERGENCE_ALERT = "DIVERGENCE_ALERT"


class AuditEvent(BaseModel):
    """Immutable record of a single auditable event in the trading system.

    Every event is tagged with an account_id to support multi-account PMS
    operations. System-level events (SYSTEM_STARTUP, SYSTEM_SHUTDOWN) use
    account_id="SYSTEM" to indicate they are not scoped to a single account.

    Fields are ordered by identification, timing, context, and metadata.
    The payload dict carries event-specific data whose schema varies by
    event_type (documented per event type below).
    """

    # --- Identification ---
    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Globally unique event identifier (UUID4)."
    )
    event_type: EventType = Field(
        description="Categorizes the event. Determines payload schema."
    )
    account_id: str = Field(
        description=(
            "Dhan client ID for the account this event belongs to. "
            "Use 'SYSTEM' for events not scoped to a single account "
            "(e.g., SYSTEM_STARTUP, GLOBAL_KILL)."
        )
    )

    # --- Timestamps ---
    timestamp_utc: str = Field(
        description="Event timestamp in ISO 8601 UTC (e.g., '2026-03-23T04:15:30.123456Z')."
    )
    timestamp_ist: str = Field(
        description="Event timestamp in ISO 8601 IST (e.g., '2026-03-23T09:45:30.123456+05:30')."
    )

    # --- Trading context ---
    strategy_id: str | None = Field(
        default=None,
        description="Strategy that produced or is associated with this event (e.g., 'S1_momentum')."
    )
    signal_id: str | None = Field(
        default=None,
        description="Signal UUID that initiated the order chain. Links signal to fills."
    )
    order_id: str | None = Field(
        default=None,
        description="Broker-assigned order ID (Dhan orderId). Null for non-order events."
    )
    instrument_id: str | None = Field(
        default=None,
        description="Dhan securityId for the instrument involved."
    )
    trading_symbol: str | None = Field(
        default=None,
        description="Human-readable trading symbol (e.g., 'NIFTY26MAR25000CE')."
    )

    # --- Payload ---
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Event-specific data. Schema varies by event_type. "
            "Always serializable to JSON. See per-event-type documentation below."
        )
    )

    # --- System metadata ---
    system_version: str = Field(
        description="Git commit SHA of the running system at event creation time."
    )
    hostname: str = Field(
        default_factory=lambda: os.uname().nodename,
        description="Hostname of the machine that generated this event."
    )
    process_id: int = Field(
        default_factory=os.getpid,
        description="OS process ID of the generating process."
    )
    schema_version: int = Field(
        default=2,
        description=(
            "Version of this event schema. Incremented on breaking changes. "
            "v1 = original single-account schema. v2 = multi-account (added account_id)."
        )
    )

    class Config:
        use_enum_values = True
```

---

### Event Types Reference

| Event Type | account_id | Required Fields | Payload Contents |
|---|---|---|---|
| `SIGNAL_GENERATED` | `SYSTEM` | `strategy_id`, `signal_id`, `instrument_id`, `trading_symbol` | Full signal context (see below) |
| `SIGNAL_REJECTED` | `SYSTEM` | `strategy_id`, `signal_id` | `reason`, risk check details |
| `SIGNAL_DEDUPED` | `SYSTEM` | `strategy_id`, `signal_id` | `original_signal_id`, `dedup_window_s` |
| `ORDER_PLACED` | per-account | `strategy_id`, `signal_id`, `order_id`, `instrument_id`, `trading_symbol` | `side`, `qty`, `price`, `order_type`, `product_type` |
| `ORDER_MODIFIED` | per-account | `order_id` | `old_price`, `new_price`, `old_qty`, `new_qty`, `modification_reason` |
| `ORDER_CANCELLED` | per-account | `order_id` | `reason`, `filled_qty`, `remaining_qty` |
| `ORDER_FILLED` | per-account | `order_id`, `signal_id` | `fill_price`, `fill_qty`, `exchange_timestamp`, `latency_ms` |
| `ORDER_PARTIAL_FILL` | per-account | `order_id` | `fill_price`, `fill_qty`, `cumulative_qty`, `remaining_qty` |
| `ORDER_REJECTED` | per-account | `order_id` | `reject_reason`, `broker_message`, `exchange_code` |
| `SL_PLACED` | per-account | `order_id` (SL order) | `parent_order_id`, `trigger_price`, `limit_price`, `qty` |
| `SL_MODIFIED` | per-account | `order_id` | `old_trigger`, `new_trigger`, `reason` |
| `SL_TRIGGERED` | per-account | `order_id` | `trigger_price`, `market_price_at_trigger` |
| `SL_CANCELLED` | per-account | `order_id` | `reason` |
| `SL_VERIFICATION_FAILED` | per-account | `order_id` | `expected_trigger`, `actual_state`, `broker_response` |
| `POSITION_OPENED` | per-account | `strategy_id`, `instrument_id`, `trading_symbol` | `qty`, `avg_entry_price`, `side` |
| `POSITION_CLOSED` | per-account | `strategy_id`, `instrument_id`, `trading_symbol` | `exit_price`, `pnl_gross`, `pnl_net`, `hold_duration_s` |
| `POSITION_RECONCILED` | per-account | `strategy_id`, `instrument_id` | `local_qty`, `broker_qty`, `action_taken` |
| `RISK_CHECK_FAILED` | per-account or `SYSTEM` | `strategy_id` | `check_name`, `threshold`, `actual_value`, `action` |
| `KILL_TRIGGERED` | per-account or `SYSTEM` | `strategy_id` | `reason`, `positions_flattened`, `orders_cancelled` |
| `GLOBAL_KILL` | `SYSTEM` | — | `reason`, `total_positions_flattened`, `total_orders_cancelled` |
| `MARGIN_CALL` | per-account | — | `margin_available`, `margin_required`, `shortfall`, `broker_message` |
| `SYSTEM_STARTUP` | `SYSTEM` | — | `version`, `config_hash`, `accounts_loaded`, `strategies_loaded` |
| `SYSTEM_SHUTDOWN` | `SYSTEM` | — | `reason`, `uptime_s`, `total_events_logged` |
| `WS_DISCONNECT` | per-account or `SYSTEM` | — | `channel` (data/orders), `duration_ms`, `error` |
| `WS_RECONNECT` | per-account or `SYSTEM` | — | `channel`, `reconnect_attempt`, `downtime_ms` |
| `ACCOUNT_SUSPENDED` | per-account | — | `reason`, `open_positions_at_suspension`, `action_taken` |
| `ACCOUNT_RESUMED` | per-account | — | `suspended_duration_s`, `resumed_by` |
| `ACCOUNT_AUTH_FAILED` | per-account | — | `error_code`, `broker_message`, `retry_count` |
| `DIVERGENCE_ALERT` | per-account | `signal_id` | `expected_fill`, `actual_fill`, `other_account_fills`, `divergence_type` |

---

### Signal Context

Every `SIGNAL_GENERATED` event carries a complete decision context in its payload, enabling full reproduction of why the signal was produced. Approximate size: 2KB per signal.

```python
signal_context_payload = {
    # Market state at signal time
    "spot_price": 24250.50,           # NIFTY spot at signal bar close
    "vix": 13.42,                     # India VIX at signal time
    "futures_price": 24268.00,        # near-month futures price
    "basis_pct": 0.072,               # (futures - spot) / spot * 100

    # Bar data (the bar that triggered the signal)
    "bar": {
        "open": 24230.00,
        "high": 24265.00,
        "low": 24218.50,
        "close": 24250.50,
        "volume": 1842000,
        "timestamp_ist": "2026-03-23T10:15:00+05:30",
        "timeframe": "5m"
    },

    # All indicator values computed by the strategy
    "indicators": {
        "rsi_14": 62.3,
        "ema_9": 24242.10,
        "ema_21": 24235.80,
        "atr_14": 85.2,
        "macd_line": 12.4,
        "macd_signal": 10.1,
        "macd_histogram": 2.3,
        "bollinger_upper": 24380.00,
        "bollinger_lower": 24120.00,
        "adx_14": 28.5,
        "volume_sma_20": 1650000,
        "oi_change_pct": 2.1
    },

    # Strategy parameters active at signal time
    "strategy_params": {
        "entry_threshold": 0.6,
        "exit_threshold": 0.4,
        "sl_atr_multiple": 1.5,
        "position_size_pct": 2.0,
        "max_holding_bars": 24,
        "cooldown_bars": 3
    },

    # Signal details
    "direction": "LONG",
    "conviction": 0.73,               # model confidence or rule strength
    "target_instrument": "NIFTY26MAR25000CE",
    "target_strike": 25000,
    "target_expiry": "2026-03-25",
    "option_type": "CE",
    "option_greeks": {
        "delta": 0.42,
        "gamma": 0.0035,
        "theta": -8.5,
        "vega": 12.1,
        "iv": 14.2
    }
}
```

Signal context is written to the audit log only, not to Redis or DuckDB position state. It exists solely for post-hoc analysis and regulatory response.

---

### Log Destinations

#### Local Filesystem

```
/var/log/trading/audit/{account_id}/{date}/events.jsonl
```

Each account gets its own directory tree. System-level events (account_id=`SYSTEM`) are written to `/var/log/trading/audit/SYSTEM/{date}/events.jsonl`. An aggregate log containing all events across all accounts is maintained at `/var/log/trading/audit/ALL/{date}/events.jsonl` for cross-account queries.

- Format: JSON Lines (one JSON object per line, newline-delimited)
- Rotation: new file per calendar date (IST), cut at 00:00 IST
- Retention on disk: 30 days (older files exist only in S3)
- Permissions: `0640`, owned by `trading:trading`

#### S3 (Long-Term Archive)

```
s3://trading-audit-logs/{account_id}/{year}/{month}/{date}/events.jsonl.gz
```

| Path Component | Example | Description |
|---|---|---|
| `account_id` | `1100012345` | Dhan client ID or `SYSTEM` for system events |
| `year` | `2026` | Four-digit year (IST) |
| `month` | `03` | Zero-padded month (IST) |
| `date` | `23` | Zero-padded day (IST) |

Upload schedule:
- **Daily upload at 16:00 IST** after market close and EOD flatten completion
- Compression: gzip (typically 8:1 ratio on JSON)
- Verification: SHA-256 checksum computed before upload, stored as S3 object metadata (`x-amz-meta-sha256`)
- Retry: 3 attempts with exponential backoff (2s, 4s, 8s). On final failure, event logged locally and Telegram CRITICAL sent.

S3 lifecycle policy:

| Phase | Duration | Storage Class | Cost (approx per GB/month) |
|---|---|---|---|
| Hot | 0-90 days | S3 Standard | $0.023 |
| Warm | 90 days - 1 year | S3 Infrequent Access | $0.0125 |
| Cold | 1-5 years | S3 Glacier Flexible Retrieval | $0.0036 |
| Delete | After 5 years | — | — |

Lifecycle rules are applied per-prefix (per-account), so all accounts follow the same retention policy. The 5-year retention satisfies SEBI's record-keeping requirements for registered portfolio managers.

#### Aggregate Log

In addition to per-account logs, an aggregate copy is written to both local and S3:

```
Local:  /var/log/trading/audit/ALL/{date}/events.jsonl
S3:     s3://trading-audit-logs/ALL/{year}/{month}/{date}/events.jsonl.gz
```

The aggregate log contains every event from every account (and SYSTEM events), enabling cross-account analysis without merging individual files. Events in the aggregate log are ordered by `timestamp_utc`.

---

### DuckDB Schema Versioning

Audit events are loaded into DuckDB for fast analytical queries during the trading day and for post-session analysis. The DuckDB database lives at `/var/lib/trading/audit.duckdb`.

#### Schema Version Tracking

```sql
CREATE TABLE IF NOT EXISTS metadata (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Current schema version
INSERT INTO metadata (key, value) VALUES ('schema_version', '2')
ON CONFLICT (key) DO UPDATE SET value = '2', updated_at = CURRENT_TIMESTAMP;
```

Schema version history:

| Version | Date | Changes |
|---|---|---|
| 1 | 2026-01 | Initial schema, single-account |
| 2 | 2026-03 | Added `account_id` column, per-account partitioning, new event types |

Migrations are forward-only Python scripts stored in `live/migrations/`:

```
live/migrations/
├── 001_initial_schema.py
├── 002_add_account_id.py
└── ...
```

Each migration script:
1. Checks current `schema_version` in `metadata`
2. Runs DDL inside a transaction
3. Updates `schema_version` on success
4. Raises on failure (no partial migrations)

#### Events Table

```sql
CREATE TABLE IF NOT EXISTS audit_events (
    event_id        TEXT PRIMARY KEY,
    event_type      TEXT NOT NULL,
    account_id      TEXT NOT NULL,
    timestamp_utc   TIMESTAMP NOT NULL,
    timestamp_ist   TIMESTAMP WITH TIME ZONE NOT NULL,
    strategy_id     TEXT,
    signal_id       TEXT,
    order_id        TEXT,
    instrument_id   TEXT,
    trading_symbol  TEXT,
    payload         TEXT NOT NULL,           -- JSON string
    system_version  TEXT NOT NULL,
    hostname        TEXT NOT NULL,
    process_id      INTEGER NOT NULL,
    schema_version  INTEGER NOT NULL DEFAULT 2
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_events_account_time
    ON audit_events (account_id, timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_events_type_time
    ON audit_events (event_type, timestamp_utc);

CREATE INDEX IF NOT EXISTS idx_events_strategy
    ON audit_events (strategy_id, timestamp_utc)
    WHERE strategy_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_events_signal
    ON audit_events (signal_id)
    WHERE signal_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_events_order
    ON audit_events (order_id)
    WHERE order_id IS NOT NULL;
```

#### Daily Partitioning

DuckDB does not natively partition like Hive, but the logger creates one `.parquet` export per account per day for archival queries:

```
/var/lib/trading/audit/parquet/{account_id}/{date}.parquet
```

These parquet files are queryable directly by DuckDB without loading into the main database, enabling historical analysis across months without bloating the live database.

---

### State Table

The audit logger maintains a lightweight state table in DuckDB for operational tracking:

```sql
CREATE TABLE IF NOT EXISTS audit_state (
    key             TEXT PRIMARY KEY,
    value           TEXT NOT NULL,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

State keys tracked:

| Key | Value Type | Description |
|---|---|---|
| `last_event_id` | UUID string | ID of the most recently written event |
| `last_event_timestamp_utc` | ISO 8601 string | Timestamp of the most recently written event |
| `events_today_count` | integer string | Running count of events written today (IST date) |
| `events_today_count:{account_id}` | integer string | Per-account event count for today |
| `last_s3_upload_date` | ISO date string | Date of the most recent successful S3 upload |
| `last_s3_upload_status` | `SUCCESS` or `FAILED:{reason}` | Result of the last S3 upload attempt |
| `last_s3_upload_accounts` | JSON list of account IDs | Accounts included in the last S3 upload |
| `disk_usage_bytes` | integer string | Total bytes used by local audit logs |
| `schema_version` | integer string | Mirrors the metadata table for quick access |
| `logger_start_time` | ISO 8601 string | When the audit logger process started |
| `pending_s3_uploads` | JSON list | Dates that failed S3 upload and are queued for retry |

State is updated atomically via DuckDB transactions. On startup, the logger reads state to determine if any S3 uploads from previous days are pending retry.

---

### Per-Account Event Filtering for PMS Client Reporting

In a PMS (Portfolio Management Service) multi-account setup, each client account must have isolated access to its own audit trail. The audit logger supports this through per-account partitioning at the storage layer and filtered queries at the application layer.

#### Querying Events for a Specific Client

To retrieve all events for a single account on a given date:

```python
import duckdb

def get_account_events(
    account_id: str,
    date: str,  # ISO date, e.g. "2026-03-23"
    event_types: list[str] | None = None,
    db_path: str = "/var/lib/trading/audit.duckdb",
) -> list[dict]:
    """Query audit events for a specific PMS client account.

    Args:
        account_id: Dhan client ID (e.g., '1100012345').
        date: Trading date in ISO format.
        event_types: Optional filter for specific event types.
            If None, returns all event types.
        db_path: Path to the DuckDB database.

    Returns:
        List of event dicts ordered by timestamp_utc ascending.
    """
    conn = duckdb.connect(db_path, read_only=True)
    query = """
        SELECT * FROM audit_events
        WHERE account_id = ?
          AND CAST(timestamp_utc AS DATE) = CAST(? AS DATE)
    """
    params = [account_id, date]

    if event_types:
        placeholders = ", ".join(["?"] * len(event_types))
        query += f" AND event_type IN ({placeholders})"
        params.extend(event_types)

    query += " ORDER BY timestamp_utc ASC"
    result = conn.execute(query, params).fetchdf()
    conn.close()
    return result.to_dict(orient="records")
```

#### Querying from S3 for Historical Reports

For dates older than 30 days (no longer on local disk):

```python
import boto3
import gzip
import json

def get_account_events_s3(
    account_id: str,
    year: str,
    month: str,
    day: str,
    bucket: str = "trading-audit-logs",
) -> list[dict]:
    """Retrieve audit events for a specific account from S3.

    Handles Glacier retrieval transparently — if the object is in
    Glacier, initiates a restore and raises an exception with the
    estimated availability time.
    """
    s3 = boto3.client("s3")
    key = f"{account_id}/{year}/{month}/{day}/events.jsonl.gz"

    response = s3.get_object(Bucket=bucket, Key=key)
    compressed = response["Body"].read()
    raw = gzip.decompress(compressed)

    events = []
    for line in raw.decode("utf-8").strip().split("\n"):
        if line:
            events.append(json.loads(line))
    return events
```

#### PMS Client Report Generation

A daily report for each PMS client is generated at 16:30 IST (after S3 upload completes) containing:

1. All orders placed and their outcomes (filled, rejected, cancelled)
2. All positions opened and closed with PnL
3. Any risk events (RISK_CHECK_FAILED, KILL_TRIGGERED, MARGIN_CALL)
4. Any account-specific incidents (ACCOUNT_SUSPENDED, DIVERGENCE_ALERT)

Reports exclude SYSTEM-level events and other accounts' data. The report generator queries only the per-account log path, never the aggregate log.

---

### Concurrency Model

The audit logger runs as a dedicated asyncio task within the main trading process. It must never block the order flow or signal generation pipeline.

#### Write Path

```
Signal/Order Event
       │
       ▼
  asyncio.Queue (unbounded)
       │
       ▼
  AuditWriter task (single consumer)
       │
       ├──► structlog formatter → JSON line
       │
       ├──► Local file append (per-account + aggregate)
       │         aiofiles, O_APPEND, buffered (flush every 100 events or 1s)
       │
       └──► DuckDB batch insert (every 500 events or 5s, whichever first)
```

Key design decisions:

1. **Unbounded asyncio.Queue** — producers (strategy orchestrator, OMS, risk manager) call `audit_logger.log(event)` which enqueues without awaiting the write. The queue is unbounded because backpressure on the audit log must never stall trading. In practice, peak throughput is ~200 events/second during EOD flatten, and the writer drains faster than producers fill.

2. **Single writer task** — all file I/O and DuckDB writes happen in one asyncio task. This avoids file locking complexity and ensures event ordering within each file. The writer is the sole owner of all file handles and the DuckDB connection.

3. **Batch DuckDB inserts** — events are buffered in memory and inserted in batches of up to 500, or flushed every 5 seconds. Batch inserts into DuckDB are 50x faster than individual inserts. The batch buffer is flushed on shutdown.

4. **Per-account file handles** — the writer maintains a dict of open file handles keyed by `(account_id, date)`. File handles are opened lazily on first write and closed at date rollover (00:00 IST) or shutdown.

5. **No write-ahead log** — the asyncio.Queue serves as the in-memory buffer. If the process crashes, events in the queue are lost. This is acceptable because:
   - Broker order records are authoritative (not the audit log)
   - Events are typically written within milliseconds of creation
   - The queue depth at any moment is rarely above 10

#### Thread Safety

| Resource | Access Pattern | Protection |
|---|---|---|
| `asyncio.Queue` | Multi-producer, single-consumer | asyncio-native (coroutine-safe) |
| Local log files | Single-writer | No lock needed (sole owner) |
| DuckDB connection | Single-writer | No lock needed (sole owner) |
| State table | Single-writer reads/writes | DuckDB transaction |
| S3 upload | Triggered by writer task | Sequential, no concurrency |

The audit logger does not use threads. All I/O is async (`aiofiles` for local disk, `aiobotocore` for S3). DuckDB operations use the synchronous API wrapped in `asyncio.to_thread()` to avoid blocking the event loop.

---

### Multi-Account Considerations

#### Per-Account Log Isolation

Each account's audit trail is physically separated at both the filesystem and S3 layers. There is no scenario where one client's events are written to another client's log file. The separation is enforced by:

1. **Directory structure** — each `account_id` gets its own directory subtree
2. **Write-time routing** — the writer task routes events by `account_id` to the correct file handle
3. **S3 prefix isolation** — each account's S3 prefix is independent; IAM policies can restrict access per-prefix if needed
4. **Query isolation** — the `get_account_events()` function filters by `account_id` in the WHERE clause; the DuckDB index on `(account_id, timestamp_utc)` makes this efficient

#### Aggregate vs Per-Account Views

| View | Location | Use Case |
|---|---|---|
| Per-account | `/var/log/trading/audit/{account_id}/...` | PMS client reports, regulatory queries for one client, account-specific debugging |
| Aggregate | `/var/log/trading/audit/ALL/...` | Cross-account divergence detection, system-wide health monitoring, capacity analysis |
| SYSTEM | `/var/log/trading/audit/SYSTEM/...` | Infrastructure events (startup, shutdown, WebSocket health) not tied to any account |

The aggregate log is append-only and written simultaneously with the per-account log. It is not constructed by merging per-account logs after the fact.

#### Account Lifecycle Events

When an account is suspended (ACCOUNT_SUSPENDED):
- The event is written to that account's log and the aggregate log
- No further ORDER_PLACED events are accepted for that account
- Existing positions may still generate POSITION_CLOSED events during forced flatten
- The account's log file remains open for these residual events

When an account is resumed (ACCOUNT_RESUMED):
- The event is written to that account's log and the aggregate log
- ORDER_PLACED events are accepted again
- The `suspended_duration_s` in the payload records how long the account was inactive

#### Divergence Detection

The DIVERGENCE_ALERT event is generated by the OMS when the same signal produces materially different outcomes across accounts. Examples:

- Signal S fills as ORDER_FILLED in account A but ORDER_REJECTED in account B
- Fill prices diverge by more than a configurable threshold (default: 0.5%)
- One account's SL triggers while another's does not

The divergence detector compares fill outcomes across accounts for each `signal_id` and emits a DIVERGENCE_ALERT to every affected account's log. The payload includes `other_account_fills` as a list of `{account_id, fill_price, fill_qty, status}` dicts for comparison.

---

### Failure Modes

#### 1. S3 Upload Failure

**Trigger:** Network error, S3 service degradation, IAM credential expiry, or bucket policy change.

**Detection:** `aiobotocore` raises `ClientError` or `EndpointConnectionError` during the 16:00 IST daily upload.

**Response:**
- Retry 3 times with exponential backoff (2s, 4s, 8s)
- On final failure: write the failed date to `pending_s3_uploads` in the state table
- Send Telegram CRITICAL: "S3 audit upload failed for {date}. Local logs retained. Retry queued."
- Next day's upload cycle retries all pending dates before the current date
- Local logs are retained for 30 days regardless of S3 upload status (provides buffer)
- No impact on trading — S3 upload is entirely post-session

#### 2. Local Disk Full

**Trigger:** `/var/log/trading/audit` partition reaches 95% capacity.

**Detection:** The writer task checks disk usage every 60 seconds via `shutil.disk_usage()`.

**Response:**
- At 90% capacity: Telegram WARNING with disk usage stats
- At 95% capacity: Telegram CRITICAL
- At 95% capacity: compress older local logs (gzip files older than 7 days that are not yet compressed)
- At 98% capacity: delete local logs older than 14 days (S3 copies exist for dates that uploaded successfully; for failed uploads, these dates are in `pending_s3_uploads` and will be force-uploaded first)
- Trading continues — disk full does not halt trading. In the extreme case where writes fail, events accumulate in the asyncio.Queue and are retried on the next successful write.

#### 3. structlog Crash or Formatter Error

**Trigger:** A payload value is not JSON-serializable (e.g., a `datetime` object passed instead of a string), or structlog itself raises an unexpected exception.

**Detection:** try/except around the structlog formatting call in the writer task.

**Response:**
- Catch the exception, log it to stderr (bypassing structlog)
- Write a fallback event with `event_type=AUDIT_INTERNAL_ERROR` containing the original event's `event_id`, `event_type`, and the exception traceback as a string in the payload
- The fallback event uses `json.dumps(event.dict(), default=str)` to force-serialize
- Increment a Prometheus counter `audit_format_errors_total`
- If more than 10 format errors occur within 60 seconds: Telegram WARNING

#### 4. Per-Account Log File Corruption

**Trigger:** Partial JSON line written due to process crash mid-write, or filesystem error.

**Detection:** On startup, the writer task validates the last line of each active log file by attempting `json.loads()`. A truncated final line indicates a crash during write.

**Response:**
- Truncated last line is moved to a `.corrupt` sidecar file for forensic review
- The log file is truncated to the last valid newline
- A SYSTEM_STARTUP event payload includes `{"recovered_corrupt_lines": N}` if any files were repaired
- No events are lost from the perspective of upstream producers (they already completed their work); only the in-flight event at crash time may be lost

#### 5. DuckDB Write Failure

**Trigger:** DuckDB file lock contention (if an external process opened the DB), disk I/O error, or schema mismatch after failed migration.

**Detection:** DuckDB raises `duckdb.IOException` or `duckdb.CatalogException` during batch insert.

**Response:**
- Log the error to local file (local file writes are independent of DuckDB)
- Buffer failed events in memory (up to 10,000 events)
- Retry DuckDB connection every 30 seconds
- On successful reconnection, flush the buffer
- If buffer exceeds 10,000 events: drop oldest events from the DuckDB buffer (local file still has them) and Telegram WARNING
- Trading and local file logging continue unaffected

#### 6. Event Queue Overflow (Theoretical)

**Trigger:** Writer task is blocked for an extended period (e.g., DuckDB and local disk both failing simultaneously).

**Detection:** Queue size monitored via Prometheus gauge `audit_queue_depth`. Alert if depth exceeds 5,000.

**Response:**
- The queue is unbounded, so producers are never blocked
- Memory pressure from an unbounded queue is mitigated by the event size (~2KB max) — 5,000 events consume ~10MB
- At 50,000 queued events (~100MB): Telegram CRITICAL, begin dropping SIGNAL_GENERATED payloads (replace full signal context with `{"context": "dropped_due_to_backpressure"}`)
- Order and position events are never dropped

#### 7. Clock Skew

**Trigger:** System clock drifts beyond 100ms from NTP (chrony monitoring detects this).

**Detection:** Compared via `chrony tracking` output parsed on each event batch write.

**Response:**
- Events are still logged with the (possibly skewed) system timestamp
- A `clock_skew_ms` field is added to the payload of every event written during the skew period
- Telegram WARNING sent with the drift magnitude
- Exchange timestamps (in order fill events) remain authoritative for ordering

---


## Component 10: Monitoring & Alerting

### Responsibility

- Expose Prometheus metrics for all trading, risk, and system health observables
- Provide Grafana dashboards for aggregate (all accounts) and per-account visibility
- Deliver Telegram notifications for actionable events, scoped per-account or shared
- Run a self-check watchdog that verifies monitoring infrastructure health independently
- Rate-limit alert delivery to prevent notification storms while allowing CRITICAL bypass
- Track per-signal execution divergence across accounts
- **Telegram is notification-only** — order execution is fully automated, Telegram sends status updates
- **Monitoring failure does NOT halt trading** — the Risk Manager runs independently of monitoring

---

### Prometheus Metrics

All metrics are served via a single `prometheus_client` HTTP server on port `9090`. Scraped by Prometheus every 5 seconds.

#### Naming Convention

All metrics use the prefix `trading_`. Labels are lowercase snake_case. The `account` label is present on every metric that is account-scoped. The `strategy` label identifies the strategy (S1..S7). Metrics without an `account` label are system-wide aggregates or infrastructure metrics.

#### Full Metric Definitions

```python
from prometheus_client import Counter, Gauge, Histogram, Info

# ──────────────────────────────────────────────
# ORDER METRICS
# ──────────────────────────────────────────────

orders_placed_total = Counter(
    "trading_orders_placed_total",
    "Total orders placed",
    ["strategy", "account"]
)

orders_filled_total = Counter(
    "trading_orders_filled_total",
    "Total orders fully filled",
    ["strategy", "account"]
)

orders_partial_filled_total = Counter(
    "trading_orders_partial_filled_total",
    "Total orders partially filled (before cancel-replace or timeout)",
    ["strategy", "account"]
)

orders_rejected_total = Counter(
    "trading_orders_rejected_total",
    "Total orders rejected by broker or exchange",
    ["strategy", "account", "reason"]
)

orders_cancelled_total = Counter(
    "trading_orders_cancelled_total",
    "Total orders cancelled (by system or user)",
    ["strategy", "account", "cancel_reason"]
)

orders_modified_total = Counter(
    "trading_orders_modified_total",
    "Total order modifications (cancel-replace cycles)",
    ["strategy", "account"]
)

order_to_fill_latency_ms = Histogram(
    "trading_order_to_fill_latency_ms",
    "Latency from order placement to fill confirmation",
    ["strategy", "account"],
    buckets=[50, 100, 250, 500, 1000, 2000, 5000, 10000]
)

signal_to_order_latency_ms = Histogram(
    "trading_signal_to_order_latency_ms",
    "Latency from signal emission to order placement on broker API",
    ["strategy", "account"],
    buckets=[10, 25, 50, 100, 250, 500, 1000]
)

# ──────────────────────────────────────────────
# PnL METRICS
# ──────────────────────────────────────────────

realized_pnl_inr = Gauge(
    "trading_realized_pnl_inr",
    "Realized PnL in INR (today, net of costs)",
    ["strategy", "account"]
)

unrealized_pnl_inr = Gauge(
    "trading_unrealized_pnl_inr",
    "Mark-to-market unrealized PnL in INR",
    ["strategy", "account"]
)

total_pnl_inr = Gauge(
    "trading_total_pnl_inr",
    "Total PnL (realized + unrealized) in INR",
    ["strategy", "account"]
)

drawdown_pct = Gauge(
    "trading_drawdown_pct",
    "Current drawdown percentage from peak equity",
    ["strategy", "account"]
)

portfolio_drawdown_pct = Gauge(
    "trading_portfolio_drawdown_pct",
    "Portfolio-level drawdown percentage (all strategies combined)",
    ["account"]
)

aggregate_portfolio_drawdown_pct = Gauge(
    "trading_aggregate_portfolio_drawdown_pct",
    "Aggregate portfolio drawdown across all accounts"
)

transaction_costs_inr = Counter(
    "trading_transaction_costs_inr",
    "Cumulative transaction costs in INR (brokerage + STT + fees + GST + stamp)",
    ["strategy", "account"]
)

# ──────────────────────────────────────────────
# POSITION & RISK METRICS
# ──────────────────────────────────────────────

open_positions = Gauge(
    "trading_open_positions",
    "Number of open position legs",
    ["strategy", "account"]
)

open_positions_total = Gauge(
    "trading_open_positions_total",
    "Total open position legs across all strategies",
    ["account"]
)

delta_exposure = Gauge(
    "trading_delta_exposure",
    "Net delta exposure in lot-equivalent units",
    ["account"]
)

margin_utilized_pct = Gauge(
    "trading_margin_utilized_pct",
    "Margin utilization as percentage of available margin",
    ["account"]
)

margin_utilized_inr = Gauge(
    "trading_margin_utilized_inr",
    "Absolute margin utilized in INR",
    ["account"]
)

margin_available_inr = Gauge(
    "trading_margin_available_inr",
    "Available margin in INR",
    ["account"]
)

sl_active_count = Gauge(
    "trading_sl_active_count",
    "Number of active server-side stop-loss orders",
    ["strategy", "account"]
)

sl_verification_failures = Counter(
    "trading_sl_verification_failures_total",
    "Number of SL verification failures (SL missing or wrong qty on broker)",
    ["strategy", "account"]
)

position_discrepancy_count = Counter(
    "trading_position_discrepancy_total",
    "Number of position discrepancies detected (local vs broker)",
    ["account"]
)

# ──────────────────────────────────────────────
# KILL & HALT METRICS
# ──────────────────────────────────────────────

strategy_status = Gauge(
    "trading_strategy_status",
    "Strategy status: 0=STOPPED, 1=RUNNING, 2=SUPPRESSED, 3=KILLED, 4=CRASHED",
    ["strategy"]
)

strategy_kill_total = Counter(
    "trading_strategy_kill_total",
    "Number of times a strategy was killed (by risk manager or operator)",
    ["strategy", "kill_reason"]
)

global_kill_total = Counter(
    "trading_global_kill_total",
    "Number of global kill events"
)

halt_active = Gauge(
    "trading_halt_active",
    "Whether a halt condition is active: 0=no, 1=yes",
    ["halt_type"]
    # halt_type: global, no_new_entries, strategy:{sid}
)

# ──────────────────────────────────────────────
# SYSTEM & INFRASTRUCTURE METRICS
# ──────────────────────────────────────────────

tick_latency_ms = Histogram(
    "trading_tick_latency_ms",
    "Latency from exchange timestamp to local processing",
    buckets=[5, 10, 25, 50, 100, 250, 500]
)

ws_connected = Gauge(
    "trading_ws_connected",
    "WebSocket connection status: 0=disconnected, 1=connected",
    ["channel", "account"]
    # channel: "data" (shared), "orders" (per-account)
    # account: "shared" for data WS, actual account_id for order WS
)

ws_reconnect_total = Counter(
    "trading_ws_reconnect_total",
    "Number of WebSocket reconnection attempts",
    ["channel", "account"]
)

rate_limit_tokens = Gauge(
    "trading_rate_limit_tokens",
    "Available rate limit tokens in the priority token bucket",
    ["account"]
)

daily_order_count = Gauge(
    "trading_daily_order_count",
    "Number of orders placed today (towards 5000/day broker limit)",
    ["account"]
)

daily_order_count_total = Gauge(
    "trading_daily_order_count_total",
    "Total orders placed today across all accounts"
)

redis_memory_bytes = Gauge(
    "trading_redis_memory_bytes",
    "Redis used memory in bytes"
)

redis_connected_clients = Gauge(
    "trading_redis_connected_clients",
    "Number of connected Redis clients"
)

process_uptime_seconds = Gauge(
    "trading_process_uptime_seconds",
    "Seconds since process start",
    ["process"]
    # process: "orchestrator", "ingester", "oms", "risk_manager", "monitor"
)

clock_drift_ms = Gauge(
    "trading_clock_drift_ms",
    "Clock drift from NTP source in milliseconds"
)

option_chain_latency_ms = Histogram(
    "trading_option_chain_latency_ms",
    "Latency of option chain API calls",
    buckets=[50, 100, 200, 500, 1000, 2000]
)

# ──────────────────────────────────────────────
# EXECUTION QUALITY METRICS
# ──────────────────────────────────────────────

fill_price_vs_signal_price_bps = Histogram(
    "trading_fill_price_vs_signal_price_bps",
    "Slippage: fill price vs signal price in basis points (positive = worse)",
    ["strategy", "account", "direction"],
    buckets=[-50, -25, -10, -5, 0, 5, 10, 25, 50, 100, 200]
)

fill_rate_pct = Gauge(
    "trading_fill_rate_pct",
    "Percentage of orders that fill completely (rolling 1-day window)",
    ["strategy", "account"]
)

cancel_replace_cycles_per_order = Histogram(
    "trading_cancel_replace_cycles_per_order",
    "Number of cancel-replace cycles per order before terminal state",
    ["strategy", "account"],
    buckets=[0, 1, 2, 3, 4, 5, 10]
)

# ──────────────────────────────────────────────
# DIVERGENCE METRICS
# ──────────────────────────────────────────────

signal_fill_divergence = Gauge(
    "trading_signal_fill_divergence",
    "Per-signal divergence: 0=all accounts filled identically, 1=divergence detected",
    ["strategy", "signal_id"]
)

account_pnl_divergence_inr = Gauge(
    "trading_account_pnl_divergence_inr",
    "Max PnL difference between any two accounts for same strategy",
    ["strategy"]
)

account_position_divergence = Gauge(
    "trading_account_position_divergence",
    "Number of position legs that differ between accounts for same strategy",
    ["strategy"]
)

# ──────────────────────────────────────────────
# SYSTEM INFO
# ──────────────────────────────────────────────

system_info = Info(
    "trading_system",
    "System metadata"
)
# Set once at startup:
# system_info.info({
#     "version": "3.0.0",
#     "environment": "live",  # or "paper"
#     "broker": "dhan",
#     "num_accounts": "3",
#     "num_strategies": "7",
# })
```

#### Metric Update Frequency

| Metric Group | Update Trigger | Typical Frequency |
|---|---|---|
| Order counters | On each order event (placed, filled, rejected) | Per-event |
| PnL gauges | On fill + every 30s mark-to-market cycle | 30s + per-fill |
| Position gauges | On fill, on SL trigger, on reconciliation | Per-event |
| Drawdown | On PnL update | 30s |
| Margin | Polled from broker API | 60s |
| System metrics (WS, uptime, Redis) | Background poller | 10s |
| Execution quality | On fill | Per-event |
| Divergence metrics | On fill + end of fill management loop | Per-signal |
| Tick latency | On each tick | Per-tick |

---

### Grafana Dashboards

All dashboards are provisioned via JSON models stored in `config/grafana/dashboards/`. Datasource: Prometheus. Refresh interval: 5s for live dashboards, 30s for risk dashboards.

#### Dashboard 1: Aggregate Trading (All Accounts Combined)

**Purpose:** Single-pane view of overall trading performance across all accounts.

**Variable:** None (hardcoded to `sum by (strategy)` across all accounts).

| Panel | Type | Query / Description |
|---|---|---|
| Total PnL (All Accounts) | Stat | `sum(trading_total_pnl_inr)` — large number, green/red coloring |
| PnL by Strategy | Bar gauge | `sum by (strategy)(trading_total_pnl_inr)` — horizontal bars, one per strategy |
| Aggregate PnL Curve | Time series | `sum(trading_realized_pnl_inr)` over time — intraday equity curve |
| Aggregate Drawdown | Gauge | `trading_aggregate_portfolio_drawdown_pct` — 0-100% with thresholds at 3% (yellow), 5% (red) |
| Total Open Positions | Stat | `sum(trading_open_positions)` — count |
| Total Orders Today | Stat | `trading_daily_order_count_total` |
| Orders Placed (time series) | Time series | `sum(rate(trading_orders_placed_total[5m]))` — orders per second |
| Fill Rate | Gauge | `avg(trading_fill_rate_pct)` — percentage |
| Total Margin Utilized | Stat | `sum(trading_margin_utilized_inr)` |
| Aggregate Transaction Costs | Stat | `sum(trading_transaction_costs_inr)` |
| Strategy Status Matrix | Table | `trading_strategy_status` — one row per strategy, status text |
| Active Halts | Stat | `sum(trading_halt_active)` — 0 = all clear, red if >0 |
| Account PnL Comparison | Bar chart | `sum by (account)(trading_total_pnl_inr)` — one bar per account |
| Position Divergence Summary | Table | `trading_account_position_divergence` — flags strategies with cross-account divergence |

**Layout:** 4-column grid. Top row: PnL stat, drawdown gauge, orders stat, margin stat. Second row: PnL curve (full width). Third row: strategy bars, status matrix. Bottom row: account comparison, divergence table.

---

#### Dashboard 2: Per-Account Trading

**Purpose:** Detailed view of a single account's trading activity.

**Variables:**
- `account_id` — dropdown, populated from `label_values(trading_total_pnl_inr, account)`, default: first account

All queries filter with `{account="$account_id"}`.

| Panel | Type | Query / Description |
|---|---|---|
| Account PnL | Stat | `sum(trading_total_pnl_inr{account="$account_id"})` |
| Realized PnL | Stat | `sum(trading_realized_pnl_inr{account="$account_id"})` |
| Unrealized PnL | Stat | `sum(trading_unrealized_pnl_inr{account="$account_id"})` |
| PnL Curve | Time series | `sum(trading_realized_pnl_inr{account="$account_id"})` over time |
| PnL by Strategy | Bar gauge | `trading_total_pnl_inr{account="$account_id"}` grouped by strategy |
| Drawdown | Gauge | `trading_portfolio_drawdown_pct{account="$account_id"}` — thresholds at 3%/5% |
| Per-Strategy Drawdown | Table | `trading_drawdown_pct{account="$account_id"}` — one row per strategy |
| Open Positions | Table | `trading_open_positions{account="$account_id"}` by strategy |
| Margin Utilization | Gauge | `trading_margin_utilized_pct{account="$account_id"}` — threshold at 80% (yellow), 90% (red) |
| Margin Available | Stat | `trading_margin_available_inr{account="$account_id"}` |
| Orders Today | Stat | `trading_daily_order_count{account="$account_id"}` |
| Order Rate | Time series | `rate(trading_orders_placed_total{account="$account_id"}[5m])` |
| Fills vs Rejections | Stacked bar | `rate(trading_orders_filled_total{account="$account_id"}[5m])` vs `rate(trading_orders_rejected_total{account="$account_id"}[5m])` |
| SL Active | Stat | `sum(trading_sl_active_count{account="$account_id"})` |
| SL Verification Failures | Stat | `sum(trading_sl_verification_failures_total{account="$account_id"})` — red if >0 |
| Transaction Costs | Stat | `sum(trading_transaction_costs_inr{account="$account_id"})` |
| Rate Limit Tokens | Gauge | `trading_rate_limit_tokens{account="$account_id"}` — red when <2 |
| WS Connection Status | Status map | `trading_ws_connected{account="$account_id"}` — green/red per channel |

**Layout:** 4-column grid. Top row: four PnL stats. Second row: PnL curve (3 cols) + drawdown gauge (1 col). Third row: margin gauge, positions table. Fourth row: orders, SL status. Bottom: WS status, rate limit.

---

#### Dashboard 3: Execution Quality (Per Account)

**Purpose:** Measure execution quality and slippage for a specific account.

**Variables:**
- `account_id` — dropdown
- `strategy` — dropdown, default: All

| Panel | Type | Query / Description |
|---|---|---|
| Median Slippage (bps) | Stat | `histogram_quantile(0.5, trading_fill_price_vs_signal_price_bps{account="$account_id"})` |
| P95 Slippage (bps) | Stat | `histogram_quantile(0.95, trading_fill_price_vs_signal_price_bps{account="$account_id"})` |
| Slippage Distribution | Heatmap | `trading_fill_price_vs_signal_price_bps{account="$account_id"}` — heatmap over time |
| Fill Latency P50 | Stat | `histogram_quantile(0.5, trading_order_to_fill_latency_ms{account="$account_id"})` |
| Fill Latency P95 | Stat | `histogram_quantile(0.95, trading_order_to_fill_latency_ms{account="$account_id"})` |
| Fill Latency P99 | Stat | `histogram_quantile(0.99, trading_order_to_fill_latency_ms{account="$account_id"})` |
| Fill Latency Over Time | Time series | P50 and P95 lines overlaid |
| Signal-to-Order Latency | Time series | `histogram_quantile(0.95, trading_signal_to_order_latency_ms{account="$account_id"})` |
| Cancel-Replace Cycles | Histogram | `trading_cancel_replace_cycles_per_order{account="$account_id"}` |
| Fill Rate by Strategy | Bar gauge | `trading_fill_rate_pct{account="$account_id"}` by strategy |
| Orders Modified | Time series | `rate(trading_orders_modified_total{account="$account_id"}[5m])` |
| Rejection Reasons | Pie chart | `sum by (reason)(trading_orders_rejected_total{account="$account_id"})` |
| Cross-Account Slippage Comparison | Bar chart | `histogram_quantile(0.5, sum by (account)(trading_fill_price_vs_signal_price_bps))` — one bar per account (ignores account_id filter) |

---

#### Dashboard 4: System Health (Shared)

**Purpose:** Infrastructure and process health. Not per-account — covers the shared system.

| Panel | Type | Query / Description |
|---|---|---|
| Tick Latency P50 | Stat | `histogram_quantile(0.5, trading_tick_latency_ms)` |
| Tick Latency P99 | Stat | `histogram_quantile(0.99, trading_tick_latency_ms)` |
| Tick Latency Over Time | Time series | P50, P95, P99 lines |
| Data WS Status | Status map | `trading_ws_connected{channel="data"}` — green/red |
| Order WS Status (All Accounts) | Status map | `trading_ws_connected{channel="orders"}` — one row per account |
| WS Reconnections | Time series | `rate(trading_ws_reconnect_total[5m])` by channel and account |
| Redis Memory | Time series | `trading_redis_memory_bytes` — threshold at 80% of maxmemory |
| Redis Connected Clients | Stat | `trading_redis_connected_clients` |
| Process Uptime | Table | `trading_process_uptime_seconds` — one row per process |
| Clock Drift | Gauge | `trading_clock_drift_ms` — red if >100ms |
| Option Chain API Latency | Time series | `histogram_quantile(0.95, trading_option_chain_latency_ms)` |
| Strategy Process Status | Table | `trading_strategy_status` — color-coded by state |
| Prometheus Scrape Health | Stat | `up{job="trading"}` — 1 = healthy |
| CPU / Memory (node exporter) | Time series | Standard node_exporter panels if available |

---

#### Dashboard 5: Risk (Per Account + Aggregate)

**Purpose:** Risk monitoring with both per-account and aggregate views.

**Variables:**
- `account_id` — dropdown with "ALL" option

| Panel | Type | Query / Description |
|---|---|---|
| Aggregate Drawdown | Gauge | `trading_aggregate_portfolio_drawdown_pct` — always shown |
| Per-Account Drawdown | Bar gauge | `trading_portfolio_drawdown_pct` by account — one bar per account |
| Per-Strategy Drawdown | Table | `trading_drawdown_pct{account="$account_id"}` — filtered or summed |
| Margin Utilization (All Accounts) | Bar gauge | `trading_margin_utilized_pct` by account |
| Delta Exposure | Gauge | `trading_delta_exposure{account="$account_id"}` |
| Kill Condition Proximity | Table | Custom panel: for each kill condition, show current value vs threshold, % remaining |
| Strategy Kills Today | Stat | `sum(trading_strategy_kill_total)` |
| Global Kills Today | Stat | `trading_global_kill_total` |
| Active Halts | Table | `trading_halt_active` — lists all active halt flags |
| SL Coverage | Stat | `sum(trading_sl_active_count{account="$account_id"})` vs `sum(trading_open_positions{account="$account_id"})` — ratio should be 1:1 |
| SL Verification Failures | Time series | `rate(trading_sl_verification_failures_total{account="$account_id"}[5m])` |
| Position Discrepancies | Stat | `sum(trading_position_discrepancy_total{account="$account_id"})` — red if >0 |
| Daily Order Count vs Limit | Gauge | `trading_daily_order_count{account="$account_id"}` — threshold at 4500 (yellow), 4900 (red) out of 5000 |
| Divergence: Position Mismatch | Table | `trading_account_position_divergence` by strategy — highlights nonzero values |
| Divergence: PnL Spread | Bar chart | `trading_account_pnl_divergence_inr` by strategy — shows max INR difference between accounts |

---

### Telegram Notifications

Telegram is used for actionable alerts only. The bot sends messages to configured chat IDs. Multi-account support adds per-account scoping for account-specific events and shared alerts for system-wide events.

#### Configuration

```yaml
# config/system.yaml — telegram section
telegram:
  bot_token_env: "TELEGRAM_BOT_TOKEN"  # from .env

  # Per-account chat IDs (account owner gets their own alerts)
  account_chats:
    ACC_001: "-1001234567890"
    ACC_002: "-1001234567891"
    ACC_003: "-1001234567892"

  # Shared chat for system-wide alerts (operator)
  system_chat: "-1009876543210"

  # Rate limiting
  rate_limit_per_account_msg_per_sec: 0.2  # 1 msg / 5s
  critical_bypass: true
```

#### Alert Levels and Scoping

| Level | Scope | Events | Destination |
|---|---|---|---|
| CRITICAL | Per-account | Margin breach (>95%), auth failure, position discrepancy, SL verification failed 3x, WS disconnect >60s for order channel | Account chat + System chat |
| CRITICAL | Shared | Global kill fired, data WS disconnect, exchange silence >30s, monitoring self-check failure, clock drift >100ms | System chat + All account chats |
| WARNING | Per-account | Margin >80%, drawdown approaching kill threshold (>80% of limit), order rejection, daily orders >4500, rate limit tokens exhausted | Account chat |
| WARNING | Shared | Strategy killed (affects all accounts), OPS pressure across accounts | System chat |
| INFO | Per-account | Daily PnL summary, order placed/filled (if verbose mode on) | Account chat |
| INFO | Shared | System start/stop, strategy state changes, config reload | System chat |

#### Alert Format Examples

**CRITICAL — Per-Account Margin Breach:**

```
CRITICAL [ACC_001] Margin Breach
━━━━━━━━━━━━━━━━━━━━━━━━━
Margin utilized: 96.2% (threshold: 95%)
Margin used: 4,81,000 INR
Margin available: 19,000 INR
Action: No new entries allowed
Time: 2026-03-23 11:42:18 IST
```

**CRITICAL — Per-Account Position Discrepancy:**

```
CRITICAL [ACC_002] Position Discrepancy
━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy: S1
Local: NIFTY 25MAR 22500 CE x 50 LONG
Broker: NIFTY 25MAR 22500 CE x 100 LONG
Delta: +50 qty mismatch
Action: Synced to broker state. Review audit log.
Time: 2026-03-23 12:15:33 IST
```

**CRITICAL — Per-Account Auth Failure:**

```
CRITICAL [ACC_003] Authentication Failure
━━━━━━━━━━━━━━━━━━━━━━━━━
Broker API returned 401 Unauthorized
Last successful auth: 2026-03-23 08:31:00 IST
Action: Orders halted for this account. Re-auth required.
Time: 2026-03-23 13:05:12 IST
```

**CRITICAL — Shared Global Kill:**

```
CRITICAL [SYSTEM] Global Kill Activated
━━━━━━━━━━━━━━━━━━━━━━━━━
Trigger: Aggregate portfolio drawdown exceeded 5%
Aggregate drawdown: 5.2%
Affected accounts: ACC_001, ACC_002, ACC_003
Action: All orders cancelled. All positions flattened.
Recovery: Manual — DEL HALT:global + full restart
Time: 2026-03-23 14:22:45 IST
```

**CRITICAL — Shared Exchange Silence:**

```
CRITICAL [SYSTEM] Exchange Silence
━━━━━━━━━━━━━━━━━━━━━━━━━
No ticks received for 30+ seconds
Last tick: 2026-03-23 10:45:12 IST
All instruments silent
Action: No new entries. Existing SLs remain on broker.
Time: 2026-03-23 10:45:42 IST
```

**WARNING — Per-Account Drawdown Approaching:**

```
WARNING [ACC_001] Drawdown Approaching Kill
━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy: S3
Current drawdown: 3.8%
Kill threshold: 5.0%
Proximity: 76% of kill threshold
Time: 2026-03-23 13:10:22 IST
```

**WARNING — Shared Strategy Kill:**

```
WARNING [SYSTEM] Strategy Killed
━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy: S4
Reason: Max daily loss exceeded (-15,200 INR > -15,000 INR limit)
Effect: S4 stopped for all accounts. No new entries.
Existing positions: SLs remain active on broker.
Recovery: DEL KILLED:S4 in Redis after review.
Time: 2026-03-23 11:55:33 IST
```

**INFO — Per-Account Daily PnL Summary:**

```
INFO [ACC_001] Daily PnL Summary
━━━━━━━━━━━━━━━━━━━━━━━━━
Date: 2026-03-23

Strategy    Realized    Unrealized   Trades   Costs
─────────────────────────────────────────────────────
S1          +3,200       0           4        -128
S2          -1,100       0           2        -88
S3          +5,400       +800        6        -216
S4          +1,800       0           3        -108
S5          0            0           0        0
S6          -400         -200        1        -40
S7          +2,100       0           2        -84
─────────────────────────────────────────────────────
TOTAL       +11,000      +600        18       -664
Net:        +10,936 INR

Drawdown:   1.2% (peak: 5,12,000)
Margin:     62% utilized
Orders:     38 placed / 34 filled / 2 rejected / 2 cancelled

Time: 2026-03-23 15:45:00 IST
```

**INFO — Shared System Start:**

```
INFO [SYSTEM] System Live
━━━━━━━━━━━━━━━━━━━━━━━━━
Accounts: 3 (ACC_001, ACC_002, ACC_003)
Strategies: 7 active (S1-S7)
Positions carried: ACC_001=2, ACC_002=2, ACC_003=2
Environment: live
Broker: Dhan
Time: 2026-03-23 09:20:00 IST
```

#### Telegram Alert Sender Implementation

```python
class TelegramAlertSender:
    """Sends scoped alerts to per-account and system chat IDs."""

    def __init__(self, config: TelegramConfig):
        self.bot_token = config.bot_token
        self.account_chats: dict[str, str] = config.account_chats
        self.system_chat: str = config.system_chat
        self._rate_limiters: dict[str, TokenBucket] = {
            account_id: TokenBucket(rate=0.2, capacity=1)
            for account_id in config.account_chats
        }
        self._system_rate_limiter = TokenBucket(rate=0.2, capacity=1)

    async def send_account_alert(
        self, account_id: str, level: AlertLevel, message: str
    ) -> None:
        """Send alert to a specific account's chat."""
        if level == AlertLevel.CRITICAL:
            # CRITICAL bypasses rate limit
            await self._send(self.account_chats[account_id], message)
            # Also send to system chat for operator visibility
            await self._send(self.system_chat, message)
        else:
            limiter = self._rate_limiters[account_id]
            if limiter.try_consume():
                await self._send(self.account_chats[account_id], message)
            else:
                pass  # Dropped — rate limited

    async def send_system_alert(
        self, level: AlertLevel, message: str
    ) -> None:
        """Send alert to system chat. CRITICAL also fans out to all accounts."""
        if level == AlertLevel.CRITICAL:
            await self._send(self.system_chat, message)
            for chat_id in self.account_chats.values():
                await self._send(chat_id, message)
        else:
            if self._system_rate_limiter.try_consume():
                await self._send(self.system_chat, message)

    async def send_daily_summary(self, account_id: str, summary: str) -> None:
        """Send daily PnL summary — not rate limited."""
        await self._send(self.account_chats[account_id], summary)

    async def _send(self, chat_id: str, text: str) -> None:
        """Send message via Telegram Bot API."""
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_notification": False,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=5) as resp:
                if resp.status != 200:
                    logger.error("Telegram send failed",
                                 chat_id=chat_id, status=resp.status)
```

---

### Alert Rate Limiting

| Parameter | Value |
|---|---|
| Rate limit per account | 1 message per 5 seconds |
| Rate limit for system chat | 1 message per 5 seconds |
| CRITICAL level | Bypasses all rate limits |
| Daily summary | Bypasses rate limits (sent once at EOD) |
| Dropped alert behavior | Silently dropped, logged locally at DEBUG level |
| Token bucket capacity | 1 (no burst allowance for non-CRITICAL) |
| Implementation | Per-account `TokenBucket` with `try_consume()` check before send |

When multiple alerts fire simultaneously (common during volatile markets), the rate limiter ensures Telegram does not throttle the bot. CRITICAL alerts are never dropped — they bypass the limiter entirely to ensure margin breaches, position discrepancies, and auth failures always reach the operator within seconds.

---

### Monitoring Self-Check

A separate cron-based watchdog process runs every 60 seconds, independent of the main trading system. Its sole job is verifying that the monitoring stack itself is healthy.

#### Self-Check Sequence

```
Every 60 seconds:
  1. Query Prometheus /api/v1/query?query=up{job="trading"}
     ├─ Success (result=1): Prometheus healthy, metrics being scraped
     ├─ Success (result=0): Scrape target down — CRITICAL
     └─ Failure (HTTP error/timeout): Prometheus unreachable — CRITICAL

  2. Query Grafana /api/health
     ├─ Success (200 + {"database":"ok"}): Grafana healthy
     └─ Failure: Grafana unreachable — WARNING

  3. Check last Telegram send timestamp (stored in Redis MONITOR:last_tg_send)
     ├─ Within 10 minutes: OK (or no alerts needed)
     └─ >10 minutes AND pending alerts in queue: Telegram delivery stalled — CRITICAL

  4. Verify own process health
     └─ Write heartbeat to Redis MONITOR:watchdog_heartbeat with TTL=120s
```

#### Self-Check Alert Path

When the self-check detects a failure, it sends a Telegram CRITICAL alert directly via the Telegram Bot API — it does NOT go through the main alert sender (which may itself be broken). This is a direct HTTP POST to `api.telegram.org`.

```python
class MonitoringSelfCheck:
    """Independent watchdog — runs as a separate cron process."""

    def __init__(self, config: MonitorConfig):
        self.prometheus_url = config.prometheus_url  # http://localhost:9090
        self.grafana_url = config.grafana_url        # http://localhost:3000
        self.telegram_token = config.telegram_bot_token
        self.system_chat = config.system_chat_id

    async def run_check(self) -> None:
        failures: list[str] = []

        # 1. Prometheus
        try:
            async with aiohttp.ClientSession() as s:
                resp = await s.get(
                    f"{self.prometheus_url}/api/v1/query",
                    params={"query": 'up{job="trading"}'},
                    timeout=5
                )
                if resp.status != 200:
                    failures.append("Prometheus HTTP error")
                else:
                    data = await resp.json()
                    results = data.get("data", {}).get("result", [])
                    if not results or results[0]["value"][1] != "1":
                        failures.append("Prometheus scrape target DOWN")
        except Exception as e:
            failures.append(f"Prometheus unreachable: {e}")

        # 2. Grafana
        try:
            async with aiohttp.ClientSession() as s:
                resp = await s.get(
                    f"{self.grafana_url}/api/health", timeout=5
                )
                if resp.status != 200:
                    failures.append("Grafana health check failed")
        except Exception as e:
            failures.append(f"Grafana unreachable: {e}")

        # 3. Telegram delivery check
        try:
            last_send = await redis.get("MONITOR:last_tg_send")
            pending = await redis.llen("MONITOR:tg_pending_queue")
            if last_send:
                age = time.time() - float(last_send)
                if age > 600 and pending > 0:
                    failures.append(
                        f"Telegram delivery stalled: {pending} pending, "
                        f"last send {int(age)}s ago"
                    )
        except Exception:
            pass  # Redis down is handled by main system

        # 4. Report
        if failures:
            msg = (
                "CRITICAL [MONITOR] Self-Check Failed\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                + "\n".join(f"- {f}" for f in failures)
                + f"\nTime: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}"
                + "\nNote: Trading continues — risk manager is independent."
            )
            await self._direct_telegram_send(msg)

        # Write heartbeat
        await redis.set("MONITOR:watchdog_heartbeat", str(time.time()), ex=120)

    async def _direct_telegram_send(self, text: str) -> None:
        """Direct Telegram send — bypasses main alert sender."""
        url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
        async with aiohttp.ClientSession() as s:
            await s.post(url, json={
                "chat_id": self.system_chat,
                "text": text,
            }, timeout=5)
```

**Key design point:** The self-check runs as a completely separate OS process (launched by cron, not by the trading system). If the main trading process crashes, the self-check still runs and detects the missing Prometheus scrape target.

**Trading does NOT halt on monitoring failure.** The risk manager runs in-process with the trading system and enforces all kill conditions independently. Monitoring is observability, not control.

---

### Divergence Monitoring

In a multi-account setup, all accounts receive the same signals from the same strategies. However, execution divergence can occur due to per-account rate limiting, partial fills, margin differences, or broker-side issues. Divergence monitoring detects and surfaces these differences.

#### Divergence Sources

| Source | Description | Expected? |
|---|---|---|
| Fill timing | Different accounts fill at different times due to rate limiter ordering | Yes — minor, harmless |
| Partial fill differences | Account A fills 50/50, Account B fills 30/50 then cancel-replace | Yes — transient |
| Margin rejection | Account A has margin, Account B rejected for insufficient margin | No — indicates capital imbalance |
| Auth failure | One account's token expired mid-session | No — CRITICAL |
| Qty differences | Different accounts have different lot allocations (capital-proportional) | Yes — by design |
| Position divergence | After a failed fill + timeout, one account has position, another does not | No — requires investigation |

#### Per-Signal Divergence Tracking

After every signal is processed across all accounts, the divergence checker runs:

```python
class DivergenceMonitor:
    """Tracks execution divergence across accounts for the same signal."""

    async def check_signal_divergence(
        self, signal_id: str, strategy: str,
        account_results: dict[str, SignalResult]
    ) -> None:
        """
        Called after fill management completes for a signal across all accounts.

        account_results: {account_id: SignalResult} where SignalResult contains:
          - filled: bool
          - fill_qty: int
          - fill_price: float
          - fill_latency_ms: float
          - terminal_state: str (FILLED, CANCELLED, REJECTED, EXPIRED)
        """
        # Check if all accounts reached the same terminal state
        states = {aid: r.terminal_state for aid, r in account_results.items()}
        unique_states = set(states.values())

        if len(unique_states) > 1:
            # Divergence detected
            signal_fill_divergence.labels(
                strategy=strategy, signal_id=signal_id
            ).set(1)

            msg = (
                f"WARNING [SYSTEM] Signal Divergence\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"Signal: {signal_id}\n"
                f"Strategy: {strategy}\n"
            )
            for aid, state in states.items():
                r = account_results[aid]
                msg += f"  {aid}: {state} (qty={r.fill_qty}, price={r.fill_price})\n"
            msg += f"Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}"

            await self.alert_sender.send_system_alert(AlertLevel.WARNING, msg)
        else:
            signal_fill_divergence.labels(
                strategy=strategy, signal_id=signal_id
            ).set(0)

    async def check_position_divergence(self) -> None:
        """
        Periodic check (every 60s): compare open positions across accounts.
        All accounts should have proportionally equivalent positions.
        """
        for strategy in self.strategies:
            positions_by_account: dict[str, list[Position]] = {}
            for account in self.accounts:
                positions_by_account[account.account_id] = (
                    await self.position_tracker.get_positions(
                        strategy=strategy, account=account.account_id
                    )
                )

            # Normalize: extract instrument + direction, ignore qty (qty is proportional)
            normalized: dict[str, set[str]] = {}
            for aid, positions in positions_by_account.items():
                normalized[aid] = {
                    f"{p.instrument}:{p.direction}" for p in positions
                }

            # Compare all accounts against the first
            reference_id = list(normalized.keys())[0]
            reference_set = normalized[reference_id]
            divergent_count = 0

            for aid, pos_set in normalized.items():
                if aid == reference_id:
                    continue
                diff = reference_set.symmetric_difference(pos_set)
                divergent_count += len(diff)

            account_position_divergence.labels(strategy=strategy).set(divergent_count)

    async def check_pnl_divergence(self) -> None:
        """
        Periodic check (every 60s): compare PnL across accounts per strategy.
        Large divergence indicates execution problems.
        """
        for strategy in self.strategies:
            pnls: dict[str, float] = {}
            for account in self.accounts:
                pnl = await self.position_tracker.get_realized_pnl(
                    strategy=strategy, account=account.account_id
                )
                pnls[account.account_id] = pnl

            if len(pnls) >= 2:
                values = list(pnls.values())
                max_diff = max(values) - min(values)
                account_pnl_divergence_inr.labels(strategy=strategy).set(max_diff)
```

#### Divergence Alert Thresholds

| Metric | Threshold | Alert Level |
|---|---|---|
| Signal terminal state mismatch | Any mismatch | WARNING |
| Position set mismatch (instruments differ) | Any mismatch | WARNING (first), CRITICAL (if persists >5 min) |
| PnL divergence per strategy | >5,000 INR between any two accounts | WARNING |
| PnL divergence per strategy | >15,000 INR between any two accounts | CRITICAL |
| Consecutive divergent signals | >3 for same strategy | CRITICAL — potential systematic issue |

---

### State Table

The monitoring system maintains and reads the following state:

| Key | Location | Type | Description | Written By | Read By |
|---|---|---|---|---|---|
| `MONITOR:watchdog_heartbeat` | Redis | String (epoch) | Self-check watchdog last heartbeat, TTL=120s | Self-check cron | Main system (optional) |
| `MONITOR:last_tg_send` | Redis | String (epoch) | Timestamp of last successful Telegram send | Alert sender | Self-check |
| `MONITOR:tg_pending_queue` | Redis | List | Queue of alerts pending delivery (retry buffer) | Alert sender | Self-check |
| `MONITOR:alert_counts:{date}` | Redis | Hash | Per-level alert counts for the day | Alert sender | Dashboard |
| `MONITOR:divergence:{strategy}` | Redis | Hash | Latest divergence state per strategy | Divergence monitor | Dashboard |
| `MONITOR:daily_summary_sent:{account}:{date}` | Redis | String | Flag: daily summary already sent for this account today | Summary sender | Summary scheduler |
| Prometheus metrics | In-memory (prometheus_client) | Gauges/Counters/Histograms | All metrics defined above | Various components | Prometheus scraper |
| Grafana dashboard JSONs | `config/grafana/dashboards/` | Files | Dashboard definitions | Operator (provisioned) | Grafana |
| `config/system.yaml` telegram section | File | YAML | Telegram config (chat IDs, rate limits) | Operator | Alert sender |

---

### Failure Modes

#### Failure Mode 1: Prometheus Scrape Target Unreachable

| Attribute | Detail |
|---|---|
| **Trigger** | Prometheus cannot reach the metrics HTTP server on port 9090 |
| **Detection** | Self-check queries `up{job="trading"}` and gets result=0 or timeout |
| **Impact** | Grafana dashboards go stale. No new metric data. Historical data retained in Prometheus. |
| **Automated Response** | Self-check sends CRITICAL via direct Telegram. No trading impact. |
| **Manual Response** | Check if trading process is alive (`ps aux`). Restart if metrics server thread crashed. |
| **Trading Impact** | None. Risk manager runs in-process, not via Prometheus. |

#### Failure Mode 2: Telegram Bot API Unreachable

| Attribute | Detail |
|---|---|
| **Trigger** | Telegram API returns HTTP errors or network timeout |
| **Detection** | Alert sender logs error. Self-check detects stalled delivery (pending queue grows, last_send ages). |
| **Impact** | No alert delivery. Operator blind to events. Trading continues. |
| **Automated Response** | Alerts queued in `MONITOR:tg_pending_queue` (max 100 entries, FIFO eviction). Retry on next send attempt. |
| **Manual Response** | Check network connectivity. Verify bot token. Check Telegram bot status via @BotFather. |
| **Trading Impact** | None. Alerts are informational. Risk manager enforces all kill conditions independently. |

#### Failure Mode 3: Grafana Down

| Attribute | Detail |
|---|---|
| **Trigger** | Grafana process crashes or becomes unreachable |
| **Detection** | Self-check queries `/api/health` and gets error |
| **Impact** | No dashboard visibility. Prometheus still collecting. Telegram still sending. |
| **Automated Response** | Self-check sends WARNING via Telegram. |
| **Manual Response** | Restart Grafana: `sudo systemctl restart grafana-server`. |
| **Trading Impact** | None. |

#### Failure Mode 4: Metrics Server Thread Crash (Within Trading Process)

| Attribute | Detail |
|---|---|
| **Trigger** | The `prometheus_client` HTTP server thread throws unhandled exception |
| **Detection** | Prometheus scrape fails. Self-check detects within 60s. |
| **Impact** | New metrics still being computed in-memory but not exposed. Dashboards stale. |
| **Automated Response** | Self-check CRITICAL alert. |
| **Manual Response** | Restart trading process (metrics server restarts with it). |
| **Trading Impact** | None — metric computation and metric serving are decoupled. |

#### Failure Mode 5: Alert Storm (Market Crash / Flash Event)

| Attribute | Detail |
|---|---|
| **Trigger** | Multiple accounts hit margin/drawdown thresholds simultaneously. Many WARNINGs + CRITICALs fire. |
| **Detection** | Rate limiter drops non-CRITICAL messages. Operator sees burst of CRITICALs. |
| **Impact** | Non-CRITICAL alerts lost. Operator overwhelmed with CRITICALs. |
| **Automated Response** | Rate limiter enforces 1 msg/5s per account for non-CRITICAL. CRITICALs all delivered. |
| **Manual Response** | Review CRITICAL alerts first. Non-CRITICAL context available in Grafana dashboards and audit log. |
| **Trading Impact** | Risk manager handles all kill conditions. Alert storm does not affect execution. |

#### Failure Mode 6: Divergence Monitor Detects Persistent Position Mismatch

| Attribute | Detail |
|---|---|
| **Trigger** | One account has a position that others do not, and it persists across 5+ check cycles (>5 minutes) |
| **Detection** | `trading_account_position_divergence` stays nonzero. Divergence monitor escalates from WARNING to CRITICAL after 5 minutes. |
| **Impact** | One account has unhedged risk. PnL diverges. |
| **Automated Response** | CRITICAL alert with exact position mismatch details. No automatic position adjustment (too dangerous). |
| **Manual Response** | Review audit log. Determine root cause (missed fill, rejected order, margin issue). Manually close orphaned position if needed. |
| **Trading Impact** | Divergent account may have unintended exposure. Other accounts unaffected. |

#### Failure Mode 7: Self-Check Cron Stops Running

| Attribute | Detail |
|---|---|
| **Trigger** | Cron daemon failure, or self-check script has a fatal error |
| **Detection** | `MONITOR:watchdog_heartbeat` TTL expires (120s). Main system can optionally check this key. |
| **Impact** | No independent monitoring verification. If Prometheus/Grafana/Telegram all fail, nobody detects it. |
| **Automated Response** | Main system logs WARNING if heartbeat key is missing (optional, defense-in-depth). |
| **Manual Response** | Check cron: `crontab -l`. Check self-check logs. Restart cron if needed. |
| **Trading Impact** | None directly. Reduces monitoring coverage depth. |

---

### Appendix: Metric Labels Reference

| Label | Values | Description |
|---|---|---|
| `strategy` | `S1`, `S2`, `S3`, `S4`, `S5`, `S6`, `S7` | Strategy identifier |
| `account` | Account IDs from config (e.g., `ACC_001`, `ACC_002`) | Dhan account identifier |
| `channel` | `data`, `orders` | WebSocket channel type |
| `reason` | `INSUFFICIENT_MARGIN`, `INVALID_PRICE`, `QUANTITY_FREEZE`, `MARKET_CLOSED`, `RATE_LIMITED`, `UNKNOWN` | Order rejection reason |
| `cancel_reason` | `FILL_TIMEOUT`, `EOD_FLATTEN`, `GLOBAL_KILL`, `STRATEGY_KILL`, `OPERATOR`, `SL_REPARENT` | Why an order was cancelled |
| `direction` | `BUY`, `SELL` | Trade direction (for slippage measurement) |
| `kill_reason` | `MAX_DAILY_LOSS`, `MAX_DRAWDOWN`, `MAX_CONSECUTIVE_LOSS`, `OPERATOR`, `GLOBAL` | Why a strategy was killed |
| `halt_type` | `global`, `no_new_entries`, `strategy:S1`..`strategy:S7` | Type of halt flag |
| `process` | `orchestrator`, `ingester`, `oms`, `risk_manager`, `monitor` | System process name |
| `signal_id` | UUID string | Unique signal identifier for divergence tracking |

---


## Component 11: Account Replication Layer

### Responsibility

- Manage multiple Dhan trading accounts under a single system instance
- Fan out strategy signals to all accounts where that strategy is enabled
- Size positions independently per account based on each account's capital, Kelly fraction, and strategy weights
- Maintain per-account auth lifecycle, order management state, and WS connections
- Track cross-account divergence for compliance reporting
- Enforce per-account risk limits (max drawdown, margin) independently
- Support PMS/AIF regulatory reporting with per-client NAV, returns, and trade logs
- **Does NOT** generate signals, resolve instruments, or modify strategy logic
- **Does NOT** auto-sync accounts — each account operates as an independent execution unit after signal fan-out

---

### Account Model

```python
class Account(pydantic.BaseModel):
    """
    Represents one Dhan trading account managed by the system.

    Each account is an independent execution unit with its own capital,
    risk limits, API credentials, and enabled strategies. The system
    treats every account identically — the prop account has no special
    status versus client accounts.

    Attributes:
        account_id: Human-readable identifier. Convention: "prop" for the
            proprietary account, "client_001", "client_002" for managed
            accounts. Used as partition key in DuckDB, Redis, and logs.
        dhan_client_id: Dhan's internal client ID (numeric string, e.g.,
            "1000000001"). Obtained from Dhan dashboard. Required for all
            API calls.
        dhan_access_token: OAuth2 access token for Dhan API. Refreshed
            daily at 08:25 IST via Dhan's token exchange flow. Stored
            in Redis with TTL for mid-session access. Never logged.
        capital: Total capital allocated to this account in INR. Used as
            the denominator for all sizing calculations. Updated daily
            from broker margin API at 08:40 (Phase 4: Position Recovery).
            Stored as Decimal for exact arithmetic — float rounding on
            ₹10Cr produces visible sizing errors.
        enabled_strategies: List of strategy IDs this account participates
            in. Example: ["S1", "S2", "S5"]. A signal from S3 is silently
            skipped for an account that does not list "S3" here.
        strategy_weights: Per-strategy capital allocation weights for this
            account. Keys must be a subset of enabled_strategies. Values
            must sum to <= 1.0. Example: {"S1": 0.20, "S2": 0.30, "S5": 0.15}.
            Remaining capital (1.0 - sum) is unallocated cash buffer.
        kelly_fraction: Fractional Kelly multiplier for this account. Range
            [0.05, 1.0]. Prop account may run 0.25 (quarter-Kelly), a
            conservative client may run 0.10. Applied multiplicatively
            with strategy_weights during sizing.
        max_drawdown_pct: Maximum drawdown from peak NAV before the account
            is suspended. Range [0.01, 0.50]. When breached: no new orders,
            existing positions protected by server-side SLs. Example: 0.10
            means 10% drawdown triggers suspension.
        max_daily_loss_pct: Maximum single-day loss as percentage of capital
            before the account is suspended for the remainder of the session.
            Range [0.005, 0.10]. More sensitive than max_drawdown_pct.
        status: Current operational status. Managed by the Account Health FSM.
            Only ACTIVE accounts receive new orders.
        peak_nav: Highest NAV ever recorded for this account. Used for
            drawdown calculation. Updated at EOD if current NAV exceeds
            previous peak. Persisted in DuckDB.
        margin_available: Latest available margin reported by Dhan margin
            API. Updated on every order placement response and periodically
            (every 60s) via REST poll. Used for pre-flight margin checks.
    """
    account_id: str
    dhan_client_id: str
    dhan_access_token: str
    capital: Decimal
    enabled_strategies: list[str]
    strategy_weights: dict[str, float]
    kelly_fraction: float = Field(ge=0.05, le=1.0)
    max_drawdown_pct: float = Field(ge=0.01, le=0.50)
    max_daily_loss_pct: float = Field(ge=0.005, le=0.10)
    status: Literal["ACTIVE", "SUSPENDED", "AUTH_FAILED", "MARGIN_CALL"] = "ACTIVE"
    peak_nav: Decimal = Decimal("0")
    margin_available: Decimal = Decimal("0")

    model_config = ConfigDict(frozen=False)

    @field_validator("strategy_weights")
    @classmethod
    def weights_within_budget(cls, v: dict[str, float]) -> dict[str, float]:
        total = sum(v.values())
        if total > 1.0 + 1e-9:
            raise ValueError(
                f"strategy_weights sum to {total:.4f}, exceeds 1.0"
            )
        for key, weight in v.items():
            if weight <= 0:
                raise ValueError(
                    f"strategy_weights['{key}'] = {weight}, must be positive"
                )
        return v

    @field_validator("enabled_strategies")
    @classmethod
    def at_least_one_strategy(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("enabled_strategies must not be empty")
        return v

    @model_validator(mode="after")
    def weights_match_enabled(self) -> "Account":
        for strategy_id in self.strategy_weights:
            if strategy_id not in self.enabled_strategies:
                raise ValueError(
                    f"strategy_weights contains '{strategy_id}' which is "
                    f"not in enabled_strategies {self.enabled_strategies}"
                )
        return self

    def current_nav(self, unrealized_pnl: Decimal, realized_pnl: Decimal) -> Decimal:
        """Compute current NAV for this account."""
        return self.capital + unrealized_pnl + realized_pnl

    def drawdown_from_peak(self, current_nav: Decimal) -> float:
        """Compute drawdown as fraction of peak NAV. Returns 0.0 if no peak."""
        if self.peak_nav <= 0:
            return 0.0
        return float((self.peak_nav - current_nav) / self.peak_nav)

    def is_drawdown_breached(self, current_nav: Decimal) -> bool:
        """Check if current drawdown exceeds the account's limit."""
        return self.drawdown_from_peak(current_nav) >= self.max_drawdown_pct
```

---

### Account Health FSM

Each account has an independent health state machine. Transitions are triggered by system events (auth failure, margin breach, operator action) and are logged to DuckDB for audit.

```
                          ┌──────────────────────────┐
                          │         ACTIVE            │
                          │  (receives new orders)    │
                          └─────┬──────┬──────┬──────┘
                                │      │      │
              auth 401 ─────────┘      │      └──── margin < min_margin
              or token expired         │            or daily_loss > max_daily_loss_pct
                                       │
                  ┌────────────────────┘
                  │ drawdown > max_drawdown_pct
                  │
         ┌────────▼────────┐     ┌──────────────────┐
         │   SUSPENDED      │     │   AUTH_FAILED     │
         │  (no new orders, │     │  (no API calls,   │
         │   SLs active)    │     │   SLs still on    │
         └────────┬─────────┘     │   broker server)  │
                  │               └──────┬────────────┘
                  │                      │
    operator reset│      successful re-auth
    + drawdown    │      (manual or auto)
    recovered     │                      │
                  │               ┌──────▼────────────┐
                  └──────────────►│     ACTIVE          │
                                  └────────────────────┘

         ┌──────────────────┐
         │   MARGIN_CALL    │
         │  (margin deficit, │
         │   broker may      │
         │   square off)     │
         └────────┬─────────┘
                  │
    margin restored (deposit / position closed)
                  │
                  ▼
              ACTIVE
```

**Transition rules:**

| From | To | Trigger | Action |
|------|----|---------|--------|
| ACTIVE | SUSPENDED | `drawdown_from_peak >= max_drawdown_pct` | Stop sending new orders. Cancel pending non-SL orders. Telegram CRITICAL. |
| ACTIVE | SUSPENDED | `daily_loss >= max_daily_loss_pct` | Same as above. Resets next trading day at 08:30. |
| ACTIVE | AUTH_FAILED | HTTP 401 from Dhan API and re-auth fails | Stop all API calls for this account. Existing SLs remain on Dhan servers. Telegram CRITICAL. |
| ACTIVE | MARGIN_CALL | Dhan margin API reports `available_margin < 0` | Stop new orders. Telegram CRITICAL with margin deficit amount. |
| SUSPENDED | ACTIVE | Operator sends `/resume {account_id}` via Telegram AND `drawdown_from_peak < max_drawdown_pct * 0.8` | Resume order flow. The 0.8 multiplier prevents oscillation at the boundary. |
| AUTH_FAILED | ACTIVE | Re-auth succeeds (auto-retry every 5 minutes, max 3 retries) | Resume all operations. Telegram INFO. |
| MARGIN_CALL | ACTIVE | `available_margin > 0` on next periodic margin check (60s interval) | Resume order flow. Telegram INFO. |

```python
class AccountHealthFSM:
    """
    Manages health state transitions for a single account.

    All transitions are logged to structlog with account_id, old_status,
    new_status, trigger_reason, and timestamp. State changes are also
    persisted to Redis for cross-process visibility and to DuckDB for
    historical audit.
    """

    def __init__(self, account: Account, redis: Redis, telegram: TelegramNotifier):
        self._account = account
        self._redis = redis
        self._telegram = telegram
        self._daily_loss: Decimal = Decimal("0")
        self._session_start_nav: Decimal = Decimal("0")

    async def check_health(
        self,
        current_nav: Decimal,
        margin_available: Decimal,
    ) -> None:
        """
        Evaluate all health conditions and transition if needed.

        Called:
        - After every fill (PnL changed)
        - Every 60 seconds (periodic margin check)
        - On auth failure detection

        Args:
            current_nav: Account's current NAV (capital + unrealized + realized).
            margin_available: Latest available margin from Dhan margin API.
        """
        old_status = self._account.status

        # Auth failure is handled separately via on_auth_failure()

        # Check margin call
        if margin_available < 0 and old_status == "ACTIVE":
            self._account.status = "MARGIN_CALL"
            self._account.margin_available = margin_available
            await self._on_transition(old_status, "MARGIN_CALL",
                                       f"margin deficit ₹{abs(margin_available):,.0f}")
            return

        # Check margin call recovery
        if margin_available > 0 and old_status == "MARGIN_CALL":
            self._account.status = "ACTIVE"
            self._account.margin_available = margin_available
            await self._on_transition(old_status, "ACTIVE",
                                       "margin restored")
            return

        # Check daily loss limit
        daily_pnl = current_nav - self._session_start_nav
        daily_loss_pct = float(abs(daily_pnl) / self._account.capital) if daily_pnl < 0 else 0.0
        if daily_loss_pct >= self._account.max_daily_loss_pct and old_status == "ACTIVE":
            self._account.status = "SUSPENDED"
            await self._on_transition(old_status, "SUSPENDED",
                                       f"daily loss {daily_loss_pct:.2%} >= "
                                       f"limit {self._account.max_daily_loss_pct:.2%}")
            return

        # Check drawdown limit
        if self._account.is_drawdown_breached(current_nav) and old_status == "ACTIVE":
            self._account.status = "SUSPENDED"
            dd = self._account.drawdown_from_peak(current_nav)
            await self._on_transition(old_status, "SUSPENDED",
                                       f"drawdown {dd:.2%} >= "
                                       f"limit {self._account.max_drawdown_pct:.2%}")
            return

    async def on_auth_failure(self) -> None:
        """Transition to AUTH_FAILED on 401 response and failed re-auth."""
        old_status = self._account.status
        if old_status != "AUTH_FAILED":
            self._account.status = "AUTH_FAILED"
            await self._on_transition(old_status, "AUTH_FAILED",
                                       "Dhan API returned 401 and re-auth failed")

    async def on_auth_recovery(self) -> None:
        """Transition back to ACTIVE after successful re-auth."""
        if self._account.status == "AUTH_FAILED":
            self._account.status = "ACTIVE"
            await self._on_transition("AUTH_FAILED", "ACTIVE",
                                       "re-auth succeeded")

    async def on_operator_resume(self, current_nav: Decimal) -> bool:
        """
        Handle operator /resume command. Returns True if resumed.

        Only resumes if current drawdown is below 80% of the limit
        (hysteresis to prevent oscillation at the boundary).
        """
        if self._account.status != "SUSPENDED":
            return False

        dd = self._account.drawdown_from_peak(current_nav)
        threshold = self._account.max_drawdown_pct * 0.8
        if dd >= threshold:
            await self._telegram.send(
                "WARNING",
                f"Cannot resume {self._account.account_id}: drawdown "
                f"{dd:.2%} still above recovery threshold {threshold:.2%}"
            )
            return False

        self._account.status = "ACTIVE"
        await self._on_transition("SUSPENDED", "ACTIVE", "operator resume")
        return True

    async def _on_transition(
        self, old: str, new: str, reason: str
    ) -> None:
        """Log and notify on every state transition."""
        logger.critical("account_status_transition",
                       account_id=self._account.account_id,
                       old_status=old,
                       new_status=new,
                       reason=reason)

        await self._redis.hset(
            f"ACCOUNT:{self._account.account_id}",
            mapping={"status": new, "status_reason": reason,
                     "status_ts": str(now_ms())},
        )

        level = "CRITICAL" if new != "ACTIVE" else "INFO"
        await self._telegram.send(
            level,
            f"Account {self._account.account_id}: {old} → {new} ({reason})"
        )
```

---

### Account Manager

The `AccountManager` is the top-level orchestrator for multi-account operations. It loads account configuration, creates per-account infrastructure, and provides the fan-out interface for signal processing.

```python
class AccountManager:
    """
    Manages all trading accounts in the system.

    Lifecycle:
    1. Load accounts from config/accounts.yaml at startup
    2. Validate all accounts (Pydantic validation + cross-account checks)
    3. Create per-account instances (DhanClient, RateLimiter, WS, etc.)
    4. Authenticate all accounts in parallel at 08:25
    5. Provide fan_out() method for signal distribution
    6. Monitor account health continuously
    7. Shut down cleanly at EOD

    Single-account mode: When accounts.yaml contains exactly one account,
    the system behaves identically to v2 (no fan-out overhead, no divergence
    tracking, no multi-account logging). The fan_out() method still works —
    it simply iterates over a list of length 1.
    """

    def __init__(self, config_path: Path = Path("config/accounts.yaml")):
        self._config_path = config_path
        self._accounts: dict[str, Account] = {}
        self._account_clients: dict[str, DhanClient] = {}
        self._rate_limiters: dict[str, PriorityRateLimiter] = {}
        self._order_state_stores: dict[str, OrderStateStore] = {}
        self._position_trackers: dict[str, PositionTracker] = {}
        self._ws_connections: dict[str, DhanOrderWS] = {}
        self._demuxers: dict[str, OrderUpdateDemuxer] = {}
        self._health_fsms: dict[str, AccountHealthFSM] = {}
        self._divergence_tracker: DivergenceTracker | None = None

    def load_accounts(self) -> None:
        """
        Load and validate all accounts from YAML config.

        Raises:
            AccountConfigError: If YAML is malformed, accounts have
                duplicate IDs, or Pydantic validation fails.
        """
        with open(self._config_path) as f:
            raw = yaml.safe_load(f)

        accounts_raw = raw.get("accounts", [])
        if not accounts_raw:
            raise AccountConfigError("No accounts defined in accounts.yaml")

        seen_ids: set[str] = set()
        seen_dhan_ids: set[str] = set()

        for entry in accounts_raw:
            account = Account.model_validate(entry)

            if account.account_id in seen_ids:
                raise AccountConfigError(
                    f"Duplicate account_id: {account.account_id}"
                )
            if account.dhan_client_id in seen_dhan_ids:
                raise AccountConfigError(
                    f"Duplicate dhan_client_id: {account.dhan_client_id}"
                )

            seen_ids.add(account.account_id)
            seen_dhan_ids.add(account.dhan_client_id)
            self._accounts[account.account_id] = account

        logger.info("accounts_loaded", count=len(self._accounts),
                    account_ids=list(self._accounts.keys()))

    async def initialize_all(self, redis: Redis, telegram: TelegramNotifier) -> None:
        """
        Create per-account infrastructure for all loaded accounts.

        Creates for each account:
        - DhanClient: HTTP client with account-specific auth headers
        - PriorityRateLimiter: 10 OPS token bucket (independent per account)
        - OrderStateStore: In-memory order state tracking
        - PositionTracker: Position and PnL tracking
        - DhanOrderWS: WebSocket connection for live order updates
        - OrderUpdateDemuxer: WS message router
        - AccountHealthFSM: Health state machine
        """
        for account_id, account in self._accounts.items():
            self._account_clients[account_id] = DhanClient(
                client_id=account.dhan_client_id,
                access_token=account.dhan_access_token,
            )

            self._rate_limiters[account_id] = PriorityRateLimiter(
                ops_limit=10,
                account_id=account_id,
            )

            self._order_state_stores[account_id] = OrderStateStore(
                account_id=account_id,
            )

            self._position_trackers[account_id] = PositionTracker(
                account_id=account_id,
                redis=redis,
            )

            self._ws_connections[account_id] = DhanOrderWS(account=account)

            self._demuxers[account_id] = OrderUpdateDemuxer(account=account)

            self._health_fsms[account_id] = AccountHealthFSM(
                account=account,
                redis=redis,
                telegram=telegram,
            )

        # Divergence tracker is shared across all accounts
        if len(self._accounts) > 1:
            self._divergence_tracker = DivergenceTracker(
                accounts=self._accounts,
                telegram=telegram,
            )

        logger.info("account_infrastructure_initialized",
                    count=len(self._accounts))

    def get_active_accounts_for_strategy(
        self, strategy_id: str
    ) -> list[Account]:
        """
        Return all ACTIVE accounts that have this strategy enabled.

        Used by Signal Router to determine fan-out targets.
        Accounts in SUSPENDED, AUTH_FAILED, or MARGIN_CALL status
        are excluded.
        """
        return [
            acc for acc in self._accounts.values()
            if acc.status == "ACTIVE" and strategy_id in acc.enabled_strategies
        ]

    def get_rate_limiter(self, account_id: str) -> PriorityRateLimiter:
        """Return the per-account rate limiter."""
        return self._rate_limiters[account_id]

    def get_client(self, account_id: str) -> DhanClient:
        """Return the per-account Dhan API client."""
        return self._account_clients[account_id]

    def get_demuxer(self, account_id: str) -> OrderUpdateDemuxer:
        """Return the per-account WS demuxer."""
        return self._demuxers[account_id]

    @property
    def is_single_account(self) -> bool:
        """True when running with exactly one account (v2-compatible mode)."""
        return len(self._accounts) == 1
```

---

### Per-Account Auth Lifecycle

All accounts authenticate in parallel at 08:25 IST (Phase 1 of the daily startup sequence). Each account's token is independent — one account's auth failure does not block others.

```python
class AuthManager:
    """
    Manages daily authentication for all accounts.

    Dhan access tokens are valid for 24 hours from issuance. The system
    refreshes them daily at 08:25, before any market data or order activity.

    Token storage: Redis hash AUTH:dhan:{account_id} with fields:
    - token: the access token string
    - issued_ts: epoch ms when token was obtained
    - expires_ts: epoch ms when token expires (issued_ts + 24h)

    Note: Stored as a simple Redis string key (SET), not a hash.
    Separate keys for token, auth_ts, and status:
    AUTH:dhan:{account_id}:token, AUTH:dhan:{account_id}:auth_ts,
    AUTH:dhan:{account_id}:status. This aligns with the Broker Router's
    pattern.

    The token is ALSO stored in the Account object in memory for fast
    access during API calls. The Redis copy is for:
    - Cross-process visibility (monitoring, CLI tools)
    - Survival across OMS process restarts within the same session
    """

    def __init__(
        self,
        accounts: dict[str, Account],
        redis: Redis,
        telegram: TelegramNotifier,
    ):
        self._accounts = accounts
        self._redis = redis
        self._telegram = telegram
        self._reauth_tasks: dict[str, asyncio.Task] = {}

    async def authenticate_all(self) -> dict[str, bool]:
        """
        Authenticate all accounts in parallel.

        Called at 08:25 IST daily. Uses asyncio.gather with
        return_exceptions=True so that one account's failure does
        not prevent others from authenticating.

        Returns:
            Dict mapping account_id to success boolean.
        """
        tasks = {
            account_id: self._authenticate_single(account)
            for account_id, account in self._accounts.items()
        }

        results = await asyncio.gather(
            *tasks.values(), return_exceptions=True
        )

        outcome: dict[str, bool] = {}
        for account_id, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                logger.critical("auth_failed",
                              account_id=account_id,
                              error=str(result))
                self._accounts[account_id].status = "AUTH_FAILED"
                outcome[account_id] = False
                await self._telegram.send(
                    "CRITICAL",
                    f"Auth FAILED for account {account_id}: {result}"
                )
            else:
                outcome[account_id] = result

        succeeded = sum(1 for v in outcome.values() if v)
        failed = sum(1 for v in outcome.values() if not v)
        logger.info("auth_complete",
                    succeeded=succeeded,
                    failed=failed)

        if failed > 0 and succeeded == 0:
            await self._telegram.send(
                "CRITICAL",
                "ALL accounts failed authentication. System cannot trade."
            )

        return outcome

    async def _authenticate_single(self, account: Account) -> bool:
        """
        Authenticate a single Dhan account.

        Flow:
        1. POST to Dhan token exchange endpoint with API key
        2. Receive access token (valid 24h)
        3. Store in Redis and in-memory Account object
        4. Verify token with a lightweight API call (GET /v2/fundlimit)

        Returns True on success, raises on failure.
        """
        try:
            token = await self._dhan_token_exchange(account)

            # Store in Redis
            redis_key = f"AUTH:dhan:{account.account_id}:token"
            issued_ts = now_ms()
            expires_ts = issued_ts + (24 * 60 * 60 * 1000)  # 24 hours
            await self._redis.hset(redis_key, mapping={
                "token": token,
                "issued_ts": str(issued_ts),
                "expires_ts": str(expires_ts),
            })
            await self._redis.expire(redis_key, 86400)  # TTL 24h

            # Update in-memory Account
            account.dhan_access_token = token

            # Verify with a lightweight call
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.dhan.co/v2/fundlimit",
                    headers={"access-token": token},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        raise AuthError(
                            f"Token verification failed: HTTP {resp.status}"
                        )

            logger.info("auth_succeeded", account_id=account.account_id)
            return True

        except Exception as e:
            logger.critical("auth_single_failed",
                          account_id=account.account_id,
                          error=str(e))
            raise

    async def handle_401_response(self, account: Account) -> bool:
        """
        Handle a 401 response during normal operation.

        Called by DhanClient when any API call returns 401.
        Attempts immediate re-auth. If that fails, starts a
        background retry task (every 5 minutes, max 3 attempts).

        Returns True if re-auth succeeded immediately.
        """
        logger.warning("auth_401_detected",
                      account_id=account.account_id)

        # Attempt immediate re-auth
        try:
            success = await self._authenticate_single(account)
            if success:
                logger.info("reauth_immediate_success",
                          account_id=account.account_id)
                return True
        except Exception:
            pass

        # Start background retry if not already running
        if account.account_id not in self._reauth_tasks:
            self._reauth_tasks[account.account_id] = asyncio.create_task(
                self._reauth_retry_loop(account),
                name=f"reauth_{account.account_id}",
            )

        return False

    async def _reauth_retry_loop(self, account: Account) -> None:
        """
        Background re-auth retry. Runs every 5 minutes, max 3 attempts.

        If all attempts fail, the account remains in AUTH_FAILED state
        and requires manual intervention (operator re-generates the
        API key on Dhan dashboard and updates config).
        """
        for attempt in range(1, 4):
            await asyncio.sleep(300)  # 5 minutes

            try:
                success = await self._authenticate_single(account)
                if success:
                    account.status = "ACTIVE"
                    await self._telegram.send(
                        "INFO",
                        f"Account {account.account_id} re-auth succeeded "
                        f"on attempt {attempt}"
                    )
                    self._reauth_tasks.pop(account.account_id, None)
                    return
            except Exception as e:
                logger.warning("reauth_retry_failed",
                             account_id=account.account_id,
                             attempt=attempt,
                             error=str(e))

        # All retries exhausted
        await self._telegram.send(
            "CRITICAL",
            f"Account {account.account_id} re-auth FAILED after 3 attempts. "
            f"Manual intervention required: re-generate API key on Dhan dashboard."
        )
        self._reauth_tasks.pop(account.account_id, None)

    async def _dhan_token_exchange(self, account: Account) -> str:
        """
        Exchange API key for access token via Dhan's OAuth flow.

        Implementation depends on Dhan's specific auth mechanism.
        This is a placeholder for the actual HTTP call.

        Returns the access token string.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.dhan.co/v2/token",
                json={
                    "client_id": account.dhan_client_id,
                    "api_key": account.dhan_access_token,  # initial token used as API key
                },
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise AuthError(
                        f"Token exchange failed: HTTP {resp.status}, body: {body}"
                    )
                data = await resp.json()
                return data["access_token"]
```

---

### Signal Fan-Out

When a strategy emits a signal, the Signal Router invokes the Account Manager's fan-out logic. The fan-out produces one independent order lifecycle per eligible account.

#### Complete Flow Diagram

```
Signal from Strategy S1:
  │
  ▼
Signal Router receives StrategySignal
  │
  ▼
Signal Deduplicator (SHARED — runs once, not per account)
  │  Checks: has this signal_id been processed before?
  │  If duplicate: discard. Log and return.
  │
  ▼
AccountManager.get_active_accounts_for_strategy("S1")
  │  Returns: [Account A (prop), Account B (client_001), Account C (client_002)]
  │  Excludes: any account where status != ACTIVE or "S1" not in enabled_strategies
  │
  ▼
Instrument Resolution (SHARED — same security_id for all accounts)
  │  On-demand option chain API call (200-400ms)
  │  Produces: instrument_id, limit_price, sl_trigger, sl_limit
  │  This is instrument selection, NOT sizing — independent of account capital
  │
  ▼
For each eligible account (PARALLEL via asyncio.gather):
  │
  ├── Account A (prop, ₹50L capital):
  │     ├── Capital Allocator: ₹50L × 0.20 (S1 weight) × 0.25 (kelly) = ₹2,50,000
  │     ├── Lot Rounding: floor(250000 / (65 × 200)) = floor(19.23) = 19 lots = 1235 qty
  │     ├── Freeze Check: 1235 < 1800 → single order
  │     ├── Risk Check (PER ACCOUNT): margin sufficient? position limits ok?
  │     ├── OMS.place_entry_with_sl(order, account_A)
  │     └── Fill Management Loop (PER ACCOUNT — independent async task)
  │
  ├── Account B (client_001, ₹2Cr capital):
  │     ├── Capital Allocator: ₹2Cr × 0.20 (S1 weight) × 0.25 (kelly) = ₹10,00,000
  │     ├── Lot Rounding: floor(1000000 / (65 × 200)) = floor(76.92) = 76 lots = 4940 qty
  │     ├── Freeze Check: 4940 > 1800 → use Dhan slicing API
  │     ├── Risk Check (PER ACCOUNT): margin sufficient? position limits ok?
  │     ├── OMS.place_entry_with_sl(order, account_B)  ← uses slicing endpoint
  │     └── Fill Management Loop (PER ACCOUNT — tracks child orders independently)
  │
  └── Account C (client_002, ₹10Cr capital):
        ├── Capital Allocator: ₹10Cr × 0.15 (S1 weight) × 0.10 (kelly) = ₹15,00,000
        ├── Lot Rounding: floor(1500000 / (65 × 200)) = floor(115.38) = 115 lots = 7475 qty
        ├── Freeze Check: 7475 > 1800 → use Dhan slicing API
        ├── Risk Check (PER ACCOUNT): margin sufficient? position limits ok?
        ├── OMS.place_entry_with_sl(order, account_C)  ← uses slicing endpoint
        └── Fill Management Loop (PER ACCOUNT — tracks child orders independently)
```

#### Fan-Out Implementation

```python
class SignalFanOut:
    """
    Distributes a single strategy signal to all eligible accounts.

    The fan-out is the bridge between the Signal Router (which processes
    signals once) and the OMS (which processes orders per-account).

    Key design decisions:
    - Instrument resolution is SHARED: all accounts trade the same contract.
      Different accounts don't pick different strikes.
    - Sizing is PER-ACCOUNT: each account computes its own lot count
      based on its capital, Kelly, and strategy weight.
    - Risk checks are PER-ACCOUNT: one account may have margin, another
      may not.
    - Order placement is PER-ACCOUNT: each account uses its own API key,
      rate limiter, and WS connection.
    - Fill management is PER-ACCOUNT: fill loops run independently.
    - Failures are INDEPENDENT: Account A rejecting does not affect
      Account B.
    """

    def __init__(
        self,
        account_manager: AccountManager,
        oms: OMS,
        capital_allocator: CapitalAllocator,
        risk_gate: RiskGate,
        divergence_tracker: DivergenceTracker | None,
    ):
        self._account_manager = account_manager
        self._oms = oms
        self._allocator = capital_allocator
        self._risk_gate = risk_gate
        self._divergence_tracker = divergence_tracker

    async def fan_out(
        self,
        signal: StrategySignal,
        resolved_instrument: ResolvedInstrument,
    ) -> list[FanOutResult]:
        """
        Fan out a resolved signal to all eligible accounts.

        Args:
            signal: The deduplicated strategy signal.
            resolved_instrument: Instrument details from the shared
                resolution step (security_id, limit_price, SL prices).

        Returns:
            List of FanOutResult, one per eligible account, indicating
            whether the order was placed, rejected, or skipped.
        """
        accounts = self._account_manager.get_active_accounts_for_strategy(
            signal.strategy_id
        )

        if not accounts:
            logger.warning("fan_out_no_eligible_accounts",
                         strategy_id=signal.strategy_id,
                         signal_id=signal.signal_id)
            return []

        # Launch all account orders in parallel
        tasks = [
            self._process_for_account(signal, resolved_instrument, account)
            for account in accounts
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        fan_out_results: list[FanOutResult] = []
        for account, result in zip(accounts, results):
            if isinstance(result, Exception):
                logger.error("fan_out_account_error",
                           account_id=account.account_id,
                           signal_id=signal.signal_id,
                           error=str(result))
                fan_out_results.append(FanOutResult(
                    account_id=account.account_id,
                    signal_id=signal.signal_id,
                    status="ERROR",
                    reason=str(result),
                    quantity=0,
                    lots=0,
                ))
            else:
                fan_out_results.append(result)

        # Record divergence
        if self._divergence_tracker and len(fan_out_results) > 1:
            await self._divergence_tracker.record_signal_outcome(
                signal.signal_id, fan_out_results
            )

        return fan_out_results

    async def _process_for_account(
        self,
        signal: StrategySignal,
        resolved_instrument: ResolvedInstrument,
        account: Account,
    ) -> FanOutResult:
        """
        Process a signal for a single account: size, risk-check, place.

        This is the per-account pipeline that runs independently
        for each account in parallel.
        """
        # 1. Compute position size for this account
        sizing = self._allocator.compute_size(
            account=account,
            strategy_id=signal.strategy_id,
            premium=resolved_instrument.limit_price,
            lot_size=resolved_instrument.lot_size,
        )

        if sizing.lots == 0:
            logger.info("fan_out_skip_insufficient_capital",
                       account_id=account.account_id,
                       signal_id=signal.signal_id,
                       strategy_id=signal.strategy_id)
            return FanOutResult(
                account_id=account.account_id,
                signal_id=signal.signal_id,
                status="SKIPPED",
                reason="insufficient capital for 1 lot",
                quantity=0,
                lots=0,
            )

        # 2. Build the account-specific resolved order
        order = ResolvedOrder(
            signal=signal,
            instrument_id=resolved_instrument.security_id,
            trading_symbol=resolved_instrument.trading_symbol,
            exchange_segment=resolved_instrument.exchange_segment,
            transaction_type=resolved_instrument.transaction_type,
            product_type=resolved_instrument.product_type,
            quantity=sizing.quantity,
            limit_price=resolved_instrument.limit_price,
            tick_size=resolved_instrument.tick_size,
            lot_size=resolved_instrument.lot_size,
            freeze_qty=resolved_instrument.freeze_qty,
            spread_bps=resolved_instrument.spread_bps,
            cost_estimate=sizing.cost_estimate,
            sl_trigger_price=resolved_instrument.sl_trigger_price,
            sl_limit_price=resolved_instrument.sl_limit_price,
            sl_validity=resolved_instrument.sl_validity,
        )

        # 3. Per-account risk check
        risk_result = await self._risk_gate.check(order, account)
        if not risk_result.approved:
            logger.warning("fan_out_risk_rejected",
                         account_id=account.account_id,
                         signal_id=signal.signal_id,
                         reason=risk_result.reason)
            return FanOutResult(
                account_id=account.account_id,
                signal_id=signal.signal_id,
                status="RISK_REJECTED",
                reason=risk_result.reason,
                quantity=order.quantity,
                lots=sizing.lots,
            )

        # 4. Place order via OMS (per-account API key, rate limiter, WS)
        entry_id, sl_id = await self._oms.place_entry_with_sl(order, account)

        if entry_id is None:
            return FanOutResult(
                account_id=account.account_id,
                signal_id=signal.signal_id,
                status="PLACEMENT_FAILED",
                reason="entry or SL placement rejected by broker",
                quantity=order.quantity,
                lots=sizing.lots,
            )

        return FanOutResult(
            account_id=account.account_id,
            signal_id=signal.signal_id,
            status="PLACED",
            reason="",
            quantity=order.quantity,
            lots=sizing.lots,
            entry_order_id=entry_id,
            sl_order_id=sl_id,
        )


class FanOutResult(pydantic.BaseModel):
    """
    Result of processing a signal for one account.

    Recorded by DivergenceTracker and used for compliance reporting.
    """
    account_id: str
    signal_id: str
    status: Literal["PLACED", "SKIPPED", "RISK_REJECTED", "PLACEMENT_FAILED", "ERROR"]
    reason: str
    quantity: int
    lots: int
    entry_order_id: str | None = None
    sl_order_id: str | None = None
    fill_price: float | None = None     # populated after fill
    fill_qty: int | None = None         # populated after fill
    fill_ts: int | None = None          # populated after fill
```

---

### Capital Allocator — Per-Account Sizing

```python
class SizingResult(pydantic.BaseModel):
    """Output of the per-account sizing calculation."""
    lots: int                          # number of lots (may be 0)
    quantity: int                      # lots × lot_size (in units)
    raw_capital: Decimal               # capital × weight × kelly
    raw_lots: float                    # before floor
    needs_slicing: bool                # quantity > freeze_qty
    cost_estimate: float               # estimated round-trip cost in ₹


class CapitalAllocator:
    """
    Computes position size for a given account and strategy.

    The sizing formula:
        raw_capital = account.capital × strategy_weight × kelly_fraction
        raw_qty = raw_capital / premium_per_unit
        raw_lots = raw_qty / lot_size
        lots = floor(raw_lots)

    Where:
        premium_per_unit = limit_price (the per-unit cost of the option/future)
        lot_size = exchange-mandated lot size (65 for NIFTY options, etc.)

    The result is always rounded DOWN (floor). We never over-allocate.
    """

    def __init__(self, cost_model: CostModel):
        self._cost_model = cost_model

    def compute_size(
        self,
        account: Account,
        strategy_id: str,
        premium: float,
        lot_size: int,
        freeze_qty: int = 1800,
    ) -> SizingResult:
        """
        Compute position size for one account on one signal.

        Args:
            account: The account to size for.
            strategy_id: Which strategy generated the signal. Used to
                look up the strategy_weight in account.strategy_weights.
            premium: Per-unit price of the instrument (option premium
                or futures price per unit).
            lot_size: Exchange lot size (e.g., 65 for NIFTY options,
                75 for NIFTY futures, 15 for BANKNIFTY options).
            freeze_qty: Exchange freeze quantity limit. Orders above
                this use the slicing API.

        Returns:
            SizingResult with lots, quantity, and cost estimate.

        Examples:
            Account A: ₹50L capital, 20% to S1, 0.25 Kelly, premium ₹200, lot 65
                raw_capital = 5000000 × 0.20 × 0.25 = 250000
                raw_qty = 250000 / 200 = 1250
                raw_lots = 1250 / 65 = 19.23
                lots = 19
                quantity = 19 × 65 = 1235
                needs_slicing = 1235 < 1800 → False

            Account B: ₹2Cr capital, 20% to S1, 0.25 Kelly, premium ₹200, lot 65
                raw_capital = 20000000 × 0.20 × 0.25 = 1000000
                raw_qty = 1000000 / 200 = 5000
                raw_lots = 5000 / 65 = 76.92
                lots = 76
                quantity = 76 × 65 = 4940
                needs_slicing = 4940 > 1800 → True (use slicing API)

            Account C: ₹1L capital, 10% to S1, 0.10 Kelly, premium ₹200, lot 65
                raw_capital = 100000 × 0.10 × 0.10 = 1000
                raw_qty = 1000 / 200 = 5
                raw_lots = 5 / 65 = 0.077
                lots = 0 → SKIP this account for this signal
        """
        weight = account.strategy_weights.get(strategy_id, 0.0)
        if weight <= 0:
            return SizingResult(
                lots=0, quantity=0, raw_capital=Decimal("0"),
                raw_lots=0.0, needs_slicing=False, cost_estimate=0.0,
            )

        raw_capital = account.capital * Decimal(str(weight)) * Decimal(str(account.kelly_fraction))
        raw_qty = float(raw_capital) / premium
        raw_lots = raw_qty / lot_size
        lots = int(raw_lots)  # floor via int() truncation

        if lots < 1:
            return SizingResult(
                lots=0, quantity=0, raw_capital=raw_capital,
                raw_lots=raw_lots, needs_slicing=False, cost_estimate=0.0,
            )

        quantity = lots * lot_size
        needs_slicing = quantity > freeze_qty
        cost_estimate = self._cost_model.estimate_round_trip(
            premium=premium,
            quantity=quantity,
            instrument_type="OPTIDX",
        )

        return SizingResult(
            lots=lots,
            quantity=quantity,
            raw_capital=raw_capital,
            raw_lots=raw_lots,
            needs_slicing=needs_slicing,
            cost_estimate=cost_estimate,
        )
```

---

### Lot Rounding Rules

| Rule | Description | Rationale |
|------|-------------|-----------|
| Always floor() | `lots = int(raw_lots)` truncates toward zero | Never over-allocate capital. A floor of 19.99 lots = 19 lots. |
| Minimum: 1 lot | If `floor(raw_lots) == 0`, skip this account for this signal | Cannot trade a fractional lot on NSE. Sub-1-lot accounts are too small for the strategy. |
| Maximum: freeze_qty | If `quantity > freeze_qty`, use Dhan slicing API | Exchange rejects single orders above freeze limit. NIFTY options: 1800 qty (~27 lots). BANKNIFTY options: 900 qty (60 lots). |
| Rounding divergence | Different accounts get different lot counts due to floor() | This is expected and acceptable. Account A with 19 lots and Account B with 76 lots are not proportional to their 1:4 capital ratio (19:76 = 1:4.0 vs theoretical). Close but not exact. |
| No ceiling | There is no maximum lot count per account (beyond freeze_qty slicing) | Capacity limits are handled by the Risk Gate (position size limits, not lot count limits). |
| Premium = limit_price | Sizing uses the limit price from instrument resolution | Not the LTP, not the mid. The limit price is the price we will actually pay. |

**Freeze quantities by instrument (NSE):**

| Instrument | Lot Size | Freeze Qty (units) | Freeze Qty (lots) |
|------------|----------|--------------------|--------------------|
| NIFTY options | 65 | 1800 | 27 |
| BANKNIFTY options | 15 | 900 | 60 |
| FINNIFTY options | 25 | 1800 | 72 |
| NIFTY futures | 75 | 1800 | 24 |
| BANKNIFTY futures | 30 | 900 | 30 |
| Stock options (typical) | varies | 2500-10000 | varies |

---

### Divergence Tracking

```python
class DivergenceRecord(pydantic.BaseModel):
    """
    Records the outcome of a signal across all accounts.

    One record per signal_id. Contains per-account outcomes for
    cross-account comparison.
    """
    signal_id: str
    strategy_id: str
    signal_ts: int
    accounts: dict[str, FanOutResult]  # account_id → result
    divergence_type: list[str]          # ["QUANTITY", "PRICE", "POSITION"]
    max_qty_divergence_pct: float       # (max_lots - min_lots) / max_lots
    max_price_divergence_bps: float     # (max_fill - min_fill) / mid * 10000
    position_divergence: bool           # True if some accounts placed and others didn't


class DivergenceTracker:
    """
    Tracks cross-account divergence for compliance and monitoring.

    Divergence is INFORMATIONAL ONLY. The system does NOT auto-sync
    accounts. Each account is an independent execution unit. Divergence
    is expected due to:
    - Lot rounding (different capital → different lot count)
    - Fill price variation (different fill times → different prices)
    - Risk gate rejections (one account has margin, another doesn't)
    - Auth failures (one account is AUTH_FAILED)

    The tracker computes rolling divergence metrics and alerts when
    divergence exceeds thresholds.

    Divergence types:
    (a) QUANTITY: accounts placed different lot counts (always present
        due to rounding unless capitals are identical)
    (b) PRICE: accounts filled at different prices (always present
        unless all accounts fill simultaneously in the same tick)
    (c) POSITION: some accounts placed orders and others did not
        (due to risk rejection, auth failure, insufficient capital)
    """

    def __init__(
        self,
        accounts: dict[str, Account],
        telegram: TelegramNotifier,
        alert_window_days: int = 5,
        alert_threshold_pct: float = 2.0,
    ):
        self._accounts = accounts
        self._telegram = telegram
        self._alert_window_days = alert_window_days
        self._alert_threshold_pct = alert_threshold_pct
        self._records: deque[DivergenceRecord] = deque(maxlen=10000)
        self._daily_returns: dict[str, list[float]] = defaultdict(list)  # account_id → daily returns

    async def record_signal_outcome(
        self,
        signal_id: str,
        results: list[FanOutResult],
    ) -> None:
        """
        Record the outcome of a signal across all accounts.

        Called by SignalFanOut after all per-account operations complete.
        Computes divergence metrics and stores the record.
        """
        placed = [r for r in results if r.status == "PLACED"]
        not_placed = [r for r in results if r.status != "PLACED"]

        divergence_types: list[str] = []

        # Quantity divergence
        if placed:
            lots_values = [r.lots for r in placed]
            max_lots = max(lots_values)
            min_lots = min(lots_values)
            qty_div_pct = ((max_lots - min_lots) / max_lots * 100) if max_lots > 0 else 0.0
            if min_lots != max_lots:
                divergence_types.append("QUANTITY")
        else:
            qty_div_pct = 0.0

        # Price divergence (only available after fills)
        price_div_bps = 0.0  # Updated when fills arrive

        # Position divergence
        position_div = len(placed) > 0 and len(not_placed) > 0
        if position_div:
            divergence_types.append("POSITION")

        record = DivergenceRecord(
            signal_id=signal_id,
            strategy_id=results[0].signal_id if results else "",
            signal_ts=now_ms(),
            accounts={r.account_id: r for r in results},
            divergence_type=divergence_types,
            max_qty_divergence_pct=qty_div_pct,
            max_price_divergence_bps=price_div_bps,
            position_divergence=position_div,
        )
        self._records.append(record)

        # Log divergence
        if divergence_types:
            logger.info("signal_divergence",
                       signal_id=signal_id,
                       types=divergence_types,
                       qty_div_pct=qty_div_pct,
                       position_div=position_div)

    async def update_fill_prices(
        self,
        signal_id: str,
        account_id: str,
        fill_price: float,
        fill_qty: int,
    ) -> None:
        """
        Update divergence record with actual fill data.

        Called by fill management loop when an order fills.
        Recomputes price divergence with the new data.
        """
        for record in reversed(self._records):
            if record.signal_id == signal_id:
                if account_id in record.accounts:
                    record.accounts[account_id].fill_price = fill_price
                    record.accounts[account_id].fill_qty = fill_qty
                    record.accounts[account_id].fill_ts = now_ms()

                # Recompute price divergence
                prices = [
                    r.fill_price for r in record.accounts.values()
                    if r.fill_price is not None
                ]
                if len(prices) >= 2:
                    mid = sum(prices) / len(prices)
                    if mid > 0:
                        price_div_bps = (max(prices) - min(prices)) / mid * 10000
                        record.max_price_divergence_bps = price_div_bps
                        if "PRICE" not in record.divergence_type:
                            record.divergence_type.append("PRICE")
                break

    async def check_rolling_divergence(self) -> None:
        """
        Check rolling return divergence across accounts.

        Called at EOD. Computes the rolling return divergence metric:
            divergence = max(account_return) - min(account_return)
        over the last alert_window_days days.

        Alerts via Telegram WARNING if divergence > alert_threshold_pct
        over any window.
        """
        if not self._daily_returns:
            return

        account_ids = list(self._daily_returns.keys())
        if len(account_ids) < 2:
            return

        window = self._alert_window_days
        for account_id in account_ids:
            returns = self._daily_returns[account_id]
            if len(returns) < window:
                continue

        # Compute rolling window returns
        min_len = min(len(self._daily_returns[aid]) for aid in account_ids)
        if min_len < window:
            return

        for start in range(min_len - window + 1):
            window_returns = {}
            for aid in account_ids:
                period_return = sum(
                    self._daily_returns[aid][start:start + window]
                )
                window_returns[aid] = period_return

            max_ret = max(window_returns.values())
            min_ret = min(window_returns.values())
            divergence = max_ret - min_ret

            if divergence > self._alert_threshold_pct:
                best_account = max(window_returns, key=window_returns.get)
                worst_account = min(window_returns, key=window_returns.get)
                await self._telegram.send(
                    "WARNING",
                    f"Account divergence alert: {divergence:.2f}% over "
                    f"{window}-day window. Best: {best_account} "
                    f"({max_ret:+.2f}%), Worst: {worst_account} "
                    f"({min_ret:+.2f}%)"
                )
                logger.warning("divergence_alert",
                             divergence_pct=divergence,
                             window_days=window,
                             best_account=best_account,
                             worst_account=worst_account)
                return  # Alert once per EOD check

    def record_daily_return(self, account_id: str, daily_return_pct: float) -> None:
        """Record one day's return for an account. Called at EOD."""
        self._daily_returns[account_id].append(daily_return_pct)
        # Keep 60 days of history (rolling 20 + buffer)
        if len(self._daily_returns[account_id]) > 60:
            self._daily_returns[account_id] = self._daily_returns[account_id][-60:]
```

---

### PMS/AIF Regulatory Reporting

```python
class PerAccountReporter:
    """
    Generates per-client regulatory reporting data for PMS/AIF compliance.

    SEBI PMS Regulation 2020 requires:
    - Monthly client portfolio statement
    - Per-client NAV computation
    - Per-client daily returns
    - Per-client trade-level audit log
    - Disclosure of strategy-wise allocation

    All data is sourced from DuckDB, which already partitions by account_id
    (audit logger writes account_id on every trade). This class provides
    aggregation views.
    """

    def __init__(self, duckdb_conn: duckdb.DuckDBPyConnection):
        self._db = duckdb_conn

    def compute_daily_nav(
        self,
        account_id: str,
        trade_date: date,
    ) -> PerAccountNAV:
        """
        Compute end-of-day NAV for a single account.

        NAV = starting_capital + sum(realized_pnl) + sum(unrealized_pnl)
              - sum(costs) - sum(fees)

        Args:
            account_id: The account to compute NAV for.
            trade_date: The trading date.

        Returns:
            PerAccountNAV with breakdown by strategy.
        """
        result = self._db.execute("""
            SELECT
                SUM(realized_pnl) as total_realized,
                SUM(unrealized_pnl) as total_unrealized,
                SUM(transaction_cost) as total_costs
            FROM trades
            WHERE account_id = ?
              AND trade_date = ?
        """, [account_id, trade_date]).fetchone()

        realized = Decimal(str(result[0] or 0))
        unrealized = Decimal(str(result[1] or 0))
        costs = Decimal(str(result[2] or 0))

        # Strategy-wise breakdown
        strategy_breakdown = self._db.execute("""
            SELECT
                strategy_id,
                SUM(realized_pnl) as realized,
                SUM(unrealized_pnl) as unrealized,
                SUM(transaction_cost) as costs,
                COUNT(*) as trade_count
            FROM trades
            WHERE account_id = ?
              AND trade_date = ?
            GROUP BY strategy_id
        """, [account_id, trade_date]).fetchall()

        return PerAccountNAV(
            account_id=account_id,
            trade_date=trade_date,
            starting_capital=self._get_starting_capital(account_id, trade_date),
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            total_costs=costs,
            nav=self._get_starting_capital(account_id, trade_date) + realized + unrealized - costs,
            strategy_breakdown=[
                StrategyPnL(
                    strategy_id=row[0],
                    realized_pnl=Decimal(str(row[1])),
                    unrealized_pnl=Decimal(str(row[2])),
                    costs=Decimal(str(row[3])),
                    trade_count=row[4],
                )
                for row in strategy_breakdown
            ],
        )

    def generate_monthly_statement(
        self,
        account_id: str,
        year: int,
        month: int,
    ) -> MonthlyStatement:
        """
        Generate SEBI-compliant monthly portfolio statement.

        Includes:
        - Opening NAV (first day of month)
        - Closing NAV (last day of month)
        - Monthly return percentage
        - Strategy-wise PnL breakdown
        - Complete trade log for the month
        - Cost breakdown (brokerage, STT, exchange, SEBI, GST, stamp)
        """
        trades = self._db.execute("""
            SELECT *
            FROM trades
            WHERE account_id = ?
              AND EXTRACT(YEAR FROM trade_date) = ?
              AND EXTRACT(MONTH FROM trade_date) = ?
            ORDER BY trade_ts
        """, [account_id, year, month]).fetchall()

        daily_navs = self._db.execute("""
            SELECT trade_date, nav
            FROM daily_nav
            WHERE account_id = ?
              AND EXTRACT(YEAR FROM trade_date) = ?
              AND EXTRACT(MONTH FROM trade_date) = ?
            ORDER BY trade_date
        """, [account_id, year, month]).fetchall()

        opening_nav = daily_navs[0][1] if daily_navs else Decimal("0")
        closing_nav = daily_navs[-1][1] if daily_navs else Decimal("0")
        monthly_return = (
            float((closing_nav - opening_nav) / opening_nav * 100)
            if opening_nav > 0 else 0.0
        )

        return MonthlyStatement(
            account_id=account_id,
            year=year,
            month=month,
            opening_nav=opening_nav,
            closing_nav=closing_nav,
            monthly_return_pct=monthly_return,
            trade_count=len(trades),
            daily_navs=[(row[0], row[1]) for row in daily_navs],
        )

    def daily_returns_series(
        self,
        account_id: str,
        start_date: date,
        end_date: date,
    ) -> list[tuple[date, float]]:
        """
        Return daily return percentages for a date range.

        Used for:
        - Divergence tracking (compare returns across accounts)
        - Client reporting (daily performance attribution)
        - Risk monitoring (drawdown calculation)
        """
        rows = self._db.execute("""
            SELECT trade_date,
                   (nav - LAG(nav) OVER (ORDER BY trade_date)) / LAG(nav) OVER (ORDER BY trade_date) * 100
                   AS daily_return_pct
            FROM daily_nav
            WHERE account_id = ?
              AND trade_date BETWEEN ? AND ?
            ORDER BY trade_date
        """, [account_id, start_date, end_date]).fetchall()

        return [(row[0], float(row[1] or 0.0)) for row in rows]


class PerAccountNAV(pydantic.BaseModel):
    """Daily NAV snapshot for one account."""
    account_id: str
    trade_date: date
    starting_capital: Decimal
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_costs: Decimal
    nav: Decimal
    strategy_breakdown: list["StrategyPnL"]


class StrategyPnL(pydantic.BaseModel):
    """PnL breakdown for one strategy within one account."""
    strategy_id: str
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    costs: Decimal
    trade_count: int


class MonthlyStatement(pydantic.BaseModel):
    """SEBI PMS monthly portfolio statement."""
    account_id: str
    year: int
    month: int
    opening_nav: Decimal
    closing_nav: Decimal
    monthly_return_pct: float
    trade_count: int
    daily_navs: list[tuple[date, Decimal]]
```

---

### Config Schema

Full YAML configuration with three example accounts demonstrating different capital levels, Kelly fractions, strategy allocations, and risk limits.

```yaml
# config/accounts.yaml
#
# Account configuration for the live trading system.
# Each account is an independent execution unit with its own
# Dhan credentials, capital, strategy allocation, and risk limits.
#
# Rules:
# - account_id must be unique across all accounts
# - dhan_client_id must be unique (one Dhan account = one entry)
# - dhan_access_token is the initial API key (refreshed daily at 08:25)
# - strategy_weights values must sum to <= 1.0
# - strategy_weights keys must be a subset of enabled_strategies
# - kelly_fraction range: [0.05, 1.0]
# - max_drawdown_pct range: [0.01, 0.50]
# - max_daily_loss_pct range: [0.005, 0.10]

accounts:
  # Proprietary account — most aggressive allocation
  - account_id: "prop"
    dhan_client_id: "1000000001"
    dhan_access_token: "${DHAN_TOKEN_PROP}"      # env var interpolation at load time
    capital: "5000000"                             # ₹50 lakhs
    enabled_strategies:
      - "S1"    # Opening Range Breakout
      - "S2"    # Overnight Trend
      - "S3"    # VWAP Mean Reversion
      - "S5"    # Expiry Day (0-DTE)
      - "S6"    # Volatility Premium
    strategy_weights:
      S1: 0.20   # ₹10L to S1
      S2: 0.25   # ₹12.5L to S2
      S3: 0.15   # ₹7.5L to S3
      S5: 0.10   # ₹5L to S5
      S6: 0.15   # ₹7.5L to S6
      # Remaining 15% (₹7.5L) = unallocated cash buffer
    kelly_fraction: 0.25                           # quarter-Kelly (conservative)
    max_drawdown_pct: 0.15                         # 15% drawdown limit
    max_daily_loss_pct: 0.03                       # 3% daily loss limit

  # Client account — conservative allocation, fewer strategies
  - account_id: "client_001"
    dhan_client_id: "1000000042"
    dhan_access_token: "${DHAN_TOKEN_CLIENT001}"
    capital: "20000000"                            # ₹2 crores
    enabled_strategies:
      - "S1"    # Opening Range Breakout
      - "S2"    # Overnight Trend
      - "S6"    # Volatility Premium
    strategy_weights:
      S1: 0.20   # ₹40L to S1
      S2: 0.30   # ₹60L to S2
      S6: 0.20   # ₹40L to S6
      # Remaining 30% (₹60L) = unallocated cash buffer
    kelly_fraction: 0.15                           # conservative Kelly
    max_drawdown_pct: 0.10                         # 10% drawdown limit (tighter)
    max_daily_loss_pct: 0.02                       # 2% daily loss limit (tighter)

  # Large institutional account — most conservative, highest capital
  - account_id: "client_002"
    dhan_client_id: "1000000099"
    dhan_access_token: "${DHAN_TOKEN_CLIENT002}"
    capital: "100000000"                           # ₹10 crores
    enabled_strategies:
      - "S1"    # Opening Range Breakout
      - "S2"    # Overnight Trend
      - "S3"    # VWAP Mean Reversion
      - "S6"    # Volatility Premium
      - "S7"    # Pairs Trading
    strategy_weights:
      S1: 0.15   # ₹1.5Cr to S1
      S2: 0.20   # ₹2Cr to S2
      S3: 0.10   # ₹1Cr to S3
      S6: 0.15   # ₹1.5Cr to S6
      S7: 0.10   # ₹1Cr to S7
      # Remaining 30% (₹3Cr) = unallocated cash buffer
    kelly_fraction: 0.10                           # tenth-Kelly (very conservative)
    max_drawdown_pct: 0.08                         # 8% drawdown limit (strictest)
    max_daily_loss_pct: 0.015                      # 1.5% daily loss limit (strictest)
```

**Sizing comparison for a single S1 signal (NIFTY CE, premium ₹200, lot size 65):**

| Account | Capital | S1 Weight | Kelly | Raw Capital | Raw Lots | Floor Lots | Quantity | Slicing? |
|---------|---------|-----------|-------|-------------|----------|------------|----------|----------|
| prop | ₹50L | 0.20 | 0.25 | ₹2,50,000 | 19.23 | 19 | 1,235 | No |
| client_001 | ₹2Cr | 0.20 | 0.15 | ₹6,00,000 | 46.15 | 46 | 2,990 | Yes (>1800) |
| client_002 | ₹10Cr | 0.15 | 0.10 | ₹15,00,000 | 115.38 | 115 | 7,475 | Yes (>1800) |

---

### State Table

| State Item | Storage | Scope | Lifecycle | Writer | Readers |
|-----------|---------|-------|-----------|--------|---------|
| Account objects | In-memory | Per-account | Loaded at startup, mutable status field | AccountManager, AccountHealthFSM | SignalFanOut, OMS, Risk Gate |
| Account config (YAML) | Disk `config/accounts.yaml` | All accounts | Static for session. Requires restart to change. | Operator (manual edit) | AccountManager (load at startup) |
| Account status | In-memory + Redis `ACCOUNT:{account_id}` | Per-account | Updated on every FSM transition | AccountHealthFSM | Monitoring, Telegram, CLI |
| Auth tokens (in-memory) | In-memory `Account.dhan_access_token` | Per-account | Refreshed daily at 08:25. Updated on re-auth. | AuthManager | DhanClient, DhanOrderWS |
| Auth tokens (Redis) | Redis `AUTH:dhan:{account_id}:token` | Per-account | TTL 24h, refreshed daily | AuthManager | Cross-process tools, CLI |
| Re-auth retry tasks | In-memory `asyncio.Task` | Per-account | Created on 401, destroyed on success or 3 failures | AuthManager | AuthManager (cancellation) |
| Per-account DhanClient | In-memory | Per-account | Session lifetime | AccountManager | OMS (REST API calls) |
| Per-account PriorityRateLimiter | In-memory | Per-account | Session lifetime. 10 OPS bucket per account. | AccountManager | OMS (all API calls for this account) |
| Per-account OrderStateStore | In-memory | Per-account | Orders created/removed during session | OMS | Fill management loops |
| Per-account PositionTracker | In-memory + DuckDB | Per-account | Session + persistent | Fill management → DuckDB writer | Risk Gate, PMS Reporter |
| Per-account WS connection | In-memory (socket) | Per-account | Session (reconnects on drop) | AccountManager | OrderUpdateDemuxer |
| Per-account WS demuxer | In-memory | Per-account | Session lifetime | AccountManager | OMS (order routing) |
| Per-account health FSM | In-memory | Per-account | Session lifetime | AccountHealthFSM | AccountManager (fan-out gating) |
| Divergence records | In-memory deque (10K cap) + DuckDB | Shared across accounts | Rolling 10K signals + persistent | DivergenceTracker | EOD reporter, monitoring |
| Daily returns per account | In-memory (60-day rolling) | Per-account | Rolling 60 days | DivergenceTracker (EOD) | Rolling divergence check |
| Fan-out results | In-memory (transient) | Per-signal | Created during fan-out, consumed by divergence tracker | SignalFanOut | DivergenceTracker |
| Daily NAV series | DuckDB `daily_nav` table | Per-account | Persistent | PerAccountReporter (EOD) | PMS reporting, drawdown calc |
| Peak NAV | In-memory `Account.peak_nav` + DuckDB | Per-account | Updated at EOD if current > peak | AccountHealthFSM (EOD) | Drawdown calculation |
| Margin available | In-memory `Account.margin_available` | Per-account | Updated every 60s via REST poll | Margin monitor task | Risk Gate, AccountHealthFSM |
| Instrument resolution cache | In-memory | SHARED (all accounts) | Per-signal (one resolution per signal) | InstrumentResolver | SignalFanOut (all accounts use same result) |
| Signal dedup set | In-memory set | SHARED (all accounts) | Session | Signal Router | Signal Router (before fan-out) |

---

### Failure Modes

| # | Failure | Detection | Impact | Recovery |
|---|---------|-----------|--------|----------|
| 1 | **Auth failure for one account at 08:25** | `authenticate_all()` returns False for that account | Account enters AUTH_FAILED. No orders placed for it. Other accounts proceed normally. | Auto-retry every 5 min (3 attempts). If all fail: Telegram CRITICAL, operator must re-generate Dhan API key. |
| 2 | **Auth failure for ALL accounts at 08:25** | `authenticate_all()` returns all False | System cannot trade at all. No positions taken. | Telegram CRITICAL: "ALL accounts auth failed." Operator must investigate. System waits in auth-retry loop. |
| 3 | **Mid-session token expiry (single account)** | HTTP 401 from Dhan API during normal operation | All API calls for that account fail. Fill management loops for that account stall. Existing server-side SLs remain active on Dhan servers. | `handle_401_response()` attempts immediate re-auth. If fails: background retry (5 min intervals, 3 max). Account set to AUTH_FAILED during retry. |
| 4 | **Margin call on one account** | Dhan margin API returns `available_margin < 0` during periodic 60s check | Account enters MARGIN_CALL. No new orders. Dhan may auto-square-off positions at their discretion. | Monitor margin every 60s. When margin restored (deposit or position closed by Dhan): auto-resume to ACTIVE. |
| 5 | **Drawdown limit breached** | `AccountHealthFSM.check_health()` detects `drawdown >= max_drawdown_pct` | Account enters SUSPENDED. No new orders. Existing positions with SLs continue. | Operator must send `/resume {account_id}` via Telegram. Only succeeds if drawdown recovered to < 80% of limit (hysteresis). |
| 6 | **Daily loss limit breached** | `AccountHealthFSM.check_health()` detects daily loss exceeds `max_daily_loss_pct` | Account enters SUSPENDED for remainder of session. Existing SLs active. | Auto-resets next trading day at 08:30 (session_start_nav is recomputed). |
| 7 | **Divergence alert (returns >2% over 5 days)** | `DivergenceTracker.check_rolling_divergence()` at EOD | Informational — no automatic action. Telegram WARNING. | Operator investigates cause: fill quality, strategy drift, partial suspension history. |
| 8 | **One account rejected, others placed (position divergence)** | `FanOutResult.status != "PLACED"` for one account while others succeeded | Accounts hold different positions for the same signal. Divergence record created with type POSITION. | Do NOT auto-sync. Log for compliance. If account fails >3 consecutive signals: Telegram CRITICAL suggesting manual review. |
| 9 | **WS disconnect for one account** | `ConnectionClosed` in that account's demuxer | Fill management for that account's orders loses real-time updates. Falls back to REST polling (uses OPS). Other accounts unaffected. | Exponential backoff reconnect: 1s, 2s, 4s, 8s, max 30s. Re-auth WS on reconnect. |
| 10 | **Redis failure** | `aioredis.ConnectionError` | Auth tokens in Redis unavailable. Account status mirrors unavailable. In-memory state continues. New orders can still be placed (in-memory tokens used). Monitoring and CLI tools lose visibility. | Retry via Sentinel. If persistent: OMS enters degraded mode. In-memory state is authoritative — Redis is a mirror. |
| 11 | **Token refresh race condition** | Two processes attempt to refresh the same account's token simultaneously | One token overwrites the other in Redis. One process may use an invalidated token. | Single-writer pattern: only the AuthManager process writes tokens. All other processes read from Redis. The AuthManager holds an asyncio.Lock per account during token refresh. |
| 12 | **Capital exhaustion mid-session** | `CapitalAllocator.compute_size()` returns 0 lots for all strategies | Account cannot participate in any new signals. Existing positions unaffected. | Informational log. Account remains ACTIVE (it may have positions generating realized PnL that restores capacity). |
| 13 | **Account re-enable race condition** | Operator sends `/resume` while a signal fan-out is in progress | Account transitions SUSPENDED → ACTIVE between the `get_active_accounts_for_strategy()` check and order placement. | Not a problem — the account was not in the active list when fan-out started, so it is excluded from THIS signal. It will be included in the NEXT signal. The window is sub-second. |
| 14 | **Config file corruption** | `yaml.safe_load()` raises exception or Pydantic validation fails | System refuses to start. No accounts loaded. | Startup aborts with clear error message. Operator must fix YAML. Previous day's config is in git (version-controlled). |
| 15 | **Duplicate dhan_client_id in config** | `AccountManager.load_accounts()` duplicate check | System refuses to start. | Pydantic validation + explicit uniqueness check catches this before any API calls. |

---

### Edge Cases

#### 1. All accounts disabled for a strategy — signal arrives

A signal from S5 arrives, but the only account with S5 enabled (`prop`) is SUSPENDED. `get_active_accounts_for_strategy("S5")` returns an empty list.

**Handling:** Signal is silently consumed. Logged at WARNING level with `fan_out_no_eligible_accounts`. No orders placed. The signal is NOT retried when the account recovers — market conditions will have changed by then.

#### 2. One account's rate limiter is exhausted, others are free

Account `client_002` (₹10Cr, 115 lots, sliced into 5 child orders) consumes 10+ OPS for entry + SL. Meanwhile, a second signal arrives. `prop` and `client_001` have OPS budget; `client_002` does not.

**Handling:** Each account has its own `PriorityRateLimiter` with an independent 10 OPS bucket. Account `client_002`'s rate limiter will queue the second signal's orders behind the first signal's pending API calls. Other accounts process the second signal immediately. The delay for `client_002` is bounded by the rate limiter queue depth — typically 1-3 seconds for large orders.

#### 3. Account added mid-session (hot-add)

An operator wants to add a new client account during market hours.

**Handling:** Not supported in v1. Adding an account requires: (a) editing `config/accounts.yaml`, (b) restarting the system. The restart takes ~30 seconds (auth + position recovery). v2 may support hot-add via a `/add_account` CLI command that creates the per-account infrastructure without restarting other accounts.

#### 4. Two accounts share the same Dhan client ID (misconfiguration)

Operator accidentally duplicates `dhan_client_id` across two account entries.

**Handling:** Caught at startup by `AccountManager.load_accounts()` uniqueness validation. System refuses to start. Error message: `"Duplicate dhan_client_id: 1000000001"`.

#### 5. Sizing produces 1 lot for prop but 0 lots for client (different Kelly)

`prop` has Kelly 0.25 → 1.2 lots → floor to 1 lot. `client_001` has Kelly 0.10 → 0.48 lots → floor to 0. `client_001` is skipped for this signal.

**Handling:** `CapitalAllocator` returns `SizingResult(lots=0)`. `_process_for_account` returns `FanOutResult(status="SKIPPED", reason="insufficient capital for 1 lot")`. `DivergenceTracker` records a POSITION divergence. This is expected for small-capital accounts with conservative Kelly fractions.

#### 6. Signal fan-out partially completes, then system crashes

Two accounts placed orders, third account's order is in-flight when the OMS process crashes.

**Handling:** On restart (Phase 4: Position Recovery):
- Query each Dhan account for active orders and positions via REST API
- Accounts 1 and 2: positions and SLs are on the broker — recovered and tracked
- Account 3: if the order was accepted by Dhan before the crash, it will appear in the active orders query. If it was never sent, there is no position to recover.
- Divergence is recorded — Account 3 may have a different position than 1 and 2.
- Server-side SLs protect all positions during the crash window.

#### 7. Account status transitions during EOD flatten

EOD flatten begins at 15:20. At 15:16, Account B's drawdown limit is breached (final losing trade pushed it over). Account B transitions SUSPENDED.

**Handling:** EOD flatten checks account status but proceeds regardless — flattening positions is a protective action that applies even to SUSPENDED accounts. SUSPENDED means "no new entries," not "stop managing existing positions." The flatten continues. After flatten, all positions are closed, so the suspension has no further effect until the next trading day.

#### 8. YAML config uses environment variables that are unset

`dhan_access_token: "${DHAN_TOKEN_PROP}"` but `DHAN_TOKEN_PROP` is not set in the environment.

**Handling:** The YAML loader performs env var interpolation before Pydantic validation. If an env var is unset:
- The raw string `"${DHAN_TOKEN_PROP}"` is passed to the Account model
- Auth will fail at 08:25 when this string is used as an API key
- To catch this earlier: the config loader validates that no `${...}` patterns remain after interpolation. If any do: abort startup with a clear error listing the missing env vars.

```python
def _interpolate_env(raw: dict) -> dict:
    """
    Replace ${ENV_VAR} patterns with environment variable values.

    Raises AccountConfigError if any env vars are unset.
    """
    import re
    env_pattern = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")

    def _replace(value: str) -> str:
        missing = []
        def _sub(match: re.Match) -> str:
            var_name = match.group(1)
            val = os.environ.get(var_name)
            if val is None:
                missing.append(var_name)
                return match.group(0)
            return val
        result = env_pattern.sub(_sub, value)
        if missing:
            raise AccountConfigError(
                f"Missing environment variables: {missing}"
            )
        return result

    # Recursively interpolate all string values
    if isinstance(raw, dict):
        return {k: _interpolate_env(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        return [_interpolate_env(v) for v in raw]
    elif isinstance(raw, str):
        return _replace(raw)
    return raw
```

#### 9. Account A fills at ₹200, Account B fills at ₹210 (10s later due to rate limiter)

Same signal, same instrument, but Account B's order was delayed by rate limiter queuing (Account B had pending orders from a prior signal consuming OPS budget).

**Handling:** Fill price divergence is expected and tracked by `DivergenceTracker.update_fill_prices()`. A 5% price difference on a ₹200 option is 250 bps — within the normal expected range for sequential fills. The divergence alert threshold (2% return divergence over 5 days) is designed to catch systematic problems, not individual fill price differences.

#### 10. Operator sends `/resume` for an AUTH_FAILED account

The `/resume` command is for SUSPENDED accounts. Sending it for AUTH_FAILED is a mistake.

**Handling:** `on_operator_resume()` checks `if self._account.status != "SUSPENDED": return False`. AUTH_FAILED accounts must be resolved through the auth re-try mechanism, not the operator resume flow. Telegram responds: "Cannot resume {account_id}: status is AUTH_FAILED, not SUSPENDED. Auth must be resolved first."

---

### Concurrency Model

All multi-account operations run within a single `asyncio` event loop in the OMS process. There are no threads and no multiprocessing for account management — the I/O-bound nature of API calls and WS reads makes async the correct choice.

#### Task Structure

```
asyncio event loop (OMS process)
│
├── Per-account WS reader tasks (N accounts × 1 task each)
│   └── ws_reader_{account_id}
│       Reads from DhanOrderWS, feeds OrderUpdateDemuxer.
│       Lifetime: session. Restarts on disconnect (backoff).
│
├── Per-account margin monitor tasks (N accounts × 1 task each)
│   └── margin_monitor_{account_id}
│       Polls Dhan GET /v2/fundlimit every 60s.
│       Updates account.margin_available.
│       Feeds AccountHealthFSM.check_health().
│
├── Per-account health check tasks (N accounts × 1 task each)
│   └── health_check_{account_id}
│       Runs AccountHealthFSM.check_health() after every fill
│       and on periodic timer (60s).
│
├── Shared signal consumer task (1 task)
│   └── signal_consumer
│       Reads from Redis STREAM:SIGNAL.
│       Calls SignalFanOut.fan_out() for each signal.
│       The fan_out() method spawns parallel per-account tasks
│       via asyncio.gather().
│
├── Per-signal, per-account fill management tasks (dynamic)
│   └── fill_mgmt_{account_id}_{order_id}
│       Created by OMS.place_entry_with_sl().
│       Manages the fill lifecycle for one order on one account.
│       Self-terminates on fill, cancel, or timeout.
│       Multiple fill tasks per account can run concurrently
│       (one per active order).
│
├── Auth re-try tasks (0 to N, dynamic)
│   └── reauth_{account_id}
│       Created on 401 detection.
│       Runs every 5 min, max 3 attempts.
│       Self-terminates on success or exhaustion.
│
├── EOD divergence check task (1 task, runs once at 15:30)
│   └── eod_divergence_check
│       Calls DivergenceTracker.check_rolling_divergence().
│       Records daily returns for each account.
│
└── EOD NAV computation task (1 task, runs once at 15:35)
    └── eod_nav_computation
        Calls PerAccountReporter.compute_daily_nav() for each account.
        Updates peak_nav if current > peak.
        Writes to DuckDB daily_nav table.
```

**Total task count:** For N accounts with M active orders:
- Fixed: N (WS readers) + N (margin monitors) + N (health checks) + 1 (signal consumer) + 1 (EOD divergence) + 1 (EOD NAV) = 3N + 3
- Dynamic: up to M fill management tasks + up to N re-auth tasks
- Example: 3 accounts, 10 active orders = 3(3) + 3 + 10 = 22 tasks
- At scale (10 accounts, 50 active orders): 3(10) + 3 + 50 = 83 tasks

All tasks are lightweight asyncio coroutines — 83 concurrent tasks is trivial for a single event loop.

#### Locking and Synchronization

| Resource | Lock Type | Scope | Contention Pattern |
|----------|-----------|-------|--------------------|
| Account.status | No lock (single-writer: AccountHealthFSM) | Per-account | Only the health FSM writes status. Fan-out reads are stale-safe (excluding a just-suspended account from one signal is acceptable). |
| Account.dhan_access_token | `asyncio.Lock` per account in AuthManager | Per-account | Contention only during re-auth (401 handling). Normal operation: no contention. |
| OrderState | `asyncio.Lock` per order (existing from OMS) | Per-account, per-order | Same as single-account mode — no change. |
| DivergenceTracker._records | No lock (single-writer: fan_out runs sequentially per signal) | Shared | Fan-out completes before the next signal is consumed. No concurrent writes. |
| Rate limiter bucket | `asyncio.Lock` per account (existing from OMS) | Per-account | Each account's rate limiter is independent. No cross-account contention. |

**Single-account mode optimization:** When `AccountManager.is_single_account` is True:
- `fan_out()` skips `asyncio.gather()` wrapping (no overhead for a single-element list)
- `DivergenceTracker` is not instantiated (no divergence to track)
- All state tables and failure modes still apply — the system is identical to v2 with exactly one account

#### Startup Sequence Integration

```
08:25  Phase 1: Auth
       └── AuthManager.authenticate_all()
           └── asyncio.gather(*[auth(account) for account in accounts])
           └── Accounts that fail → AUTH_FAILED (others continue)

08:30  Phase 2: Data
       └── (unchanged — instrument master, WS subscriptions)

08:35  Phase 3: Strategies
       └── (unchanged — strategy processes start)

08:40  Phase 4: Position Recovery
       └── For each account (parallel):
           └── Query Dhan positions API
           └── Query Dhan active orders API
           └── Rebuild PositionTracker state
           └── Verify SL pairings for existing positions
           └── Update account.margin_available from Dhan fundlimit API
           └── Compute session_start_nav for daily loss tracking

08:45  Phase 5: Ready
       └── AccountManager.initialize_all() complete
       └── All per-account tasks started (WS readers, margin monitors, health checks)
       └── Signal consumer task starts reading from STREAM:SIGNAL
       └── System is live
```

---

### Shutdown Sequence

```
15:20  EOD Flatten
       └── For each account (parallel):
           └── OMS.eod_flatten(account) — same as single-account, but per-account

15:25  EOD Overnight SL Conversion
       └── For strategies S2, S6, S7 (multi-day):
           └── Convert DAY SLs to Forever Orders (per-account)

15:30  Market Close
       └── Cancel all pending orders across all accounts
       └── WS connections begin graceful disconnect

15:35  EOD Reporting
       └── DivergenceTracker.check_rolling_divergence()
       └── PerAccountReporter.compute_daily_nav() for each account
       └── Update peak_nav for each account
       └── Record daily_returns for divergence tracking
       └── Telegram summary: per-account PnL, divergence status

15:40  Shutdown
       └── Cancel all asyncio tasks
       └── Close WS connections
       └── Flush DuckDB writes
       └── Process exits cleanly
```

---


# Cross-Cutting Concerns, Testing, Open Questions & Implementation

**Scope:** v3 — multi-account mode. Dhan primary broker. AWS EC2 ap-south-1.

---

## 3. CROSS-CUTTING CONCERNS

### A. Time Management

**Internal:** UTC epoch milliseconds (`int64`) everywhere. Conversion to IST only at display boundaries (Telegram messages, Grafana dashboards, audit log human-readable column).

**Time synchronization:** EC2 instance runs `chrony` synced to AWS NTP (`169.254.169.123`). Alert if clock drift exceeds 100ms. Exchange timestamps in ticks are authoritative for ordering — local timestamps are for latency measurement only.

```bash
# /etc/chrony/chrony.conf (EC2 ap-south-1)
server 169.254.169.123 prefer iburst minpoll 4 maxpoll 4
driftfile /var/lib/chrony/drift
makestep 0.1 3
rtcsync
```

**Session phases:**

| Phase | IST Window | System Behavior |
|-------|-----------|-----------------|
| PRE_OPEN | 09:00 - 09:08 | Ingester active, strategies warming up, no orders |
| AUCTION | 09:08 - 09:15 | Observe only, compute opening indicators |
| CONTINUOUS | 09:15 - 15:30 | Trading active from 09:20 (skip first 5 min of continuous) |
| CLOSING | 15:30 - 15:40 | Monitor only, EOD flatten should be complete |
| CLOSED | 15:40+ | Overnight positions managed (S2, S6, S7) |

```python
class SessionPhase(str, Enum):
    PRE_OPEN   = "PRE_OPEN"
    AUCTION    = "AUCTION"
    CONTINUOUS = "CONTINUOUS"
    CLOSING    = "CLOSING"
    CLOSED     = "CLOSED"

    @classmethod
    def current(cls, now_ist: time) -> "SessionPhase":
        if now_ist < time(9, 8):
            return cls.PRE_OPEN
        if now_ist < time(9, 15):
            return cls.AUCTION
        if now_ist < time(15, 30):
            return cls.CONTINUOUS
        if now_ist < time(15, 40):
            return cls.CLOSING
        return cls.CLOSED
```

**Holiday calendar:** Static JSON `config/nse_holidays_2026.json` + runtime verification:

```python
class HolidayCalendar:
    holidays: set[date]

    async def verify_exchange_open(self) -> bool:
        """Called at 09:10 before strategies start."""
        # 1. Check static calendar
        if today in self.holidays:
            return False
        # 2. Check broker market status API (if available)
        try:
            status = await broker.get_market_status()
            if status != "OPEN":
                return False
        except Exception:
            pass  # API unreliable, continue
        # 3. Check if ticks are flowing (after 09:15)
        # All-instrument-silence detector handles this
        return True

    def next_weekly_expiry(self, d: date) -> date:
        """Next Tuesday. If Tue is holiday, previous trading day."""
        days_ahead = (1 - d.weekday()) % 7  # 1 = Tuesday
        if days_ahead == 0:
            days_ahead = 7
        candidate = d + timedelta(days=days_ahead)
        while candidate in self.holidays:
            candidate -= timedelta(days=1)
            while candidate.weekday() >= 5:  # skip weekends
                candidate -= timedelta(days=1)
        return candidate

    def next_monthly_expiry(self, d: date) -> date:
        """Last Thursday of month. If holiday, previous trading day."""
        # Find last Thursday
        last_day = date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])
        offset = (last_day.weekday() - 3) % 7  # 3 = Thursday
        candidate = last_day - timedelta(days=offset)
        if candidate <= d:
            # Move to next month
            next_month = d.replace(day=28) + timedelta(days=4)
            last_day = date(next_month.year, next_month.month,
                          calendar.monthrange(next_month.year, next_month.month)[1])
            offset = (last_day.weekday() - 3) % 7
            candidate = last_day - timedelta(days=offset)
        while candidate in self.holidays:
            candidate -= timedelta(days=1)
            while candidate.weekday() >= 5:
                candidate -= timedelta(days=1)
        return candidate

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays
```

---

### B. Configuration

#### Directory Layout

```
config/
├── system.yaml             # non-secret system config
├── strategies.yaml         # per-strategy params + fill params
├── risk.yaml               # risk limits, kill conditions
├── accounts.yaml           # multi-account definitions (NEW in v3)
├── broker.yaml             # broker endpoints (not credentials)
├── nse_holidays_2026.json
└── .env                    # secrets (NOT in git)
```

#### system.yaml

```yaml
system:
  trading_env: live            # paper | live
  log_level: INFO
  timezone: Asia/Kolkata
  data_dir: /data/trading
  redis:
    host: 127.0.0.1
    port: 6379
    sentinel:
      master_name: trading-master
      quorum: 2
  duckdb:
    path: /data/trading/live.duckdb
    wal_autocheckpoint: 1000
  s3:
    bucket: trading-prod-ap-south-1
    prefix: live/
    region: ap-south-1
  monitoring:
    prometheus_port: 9090
    grafana_port: 3000
  telegram:
    enabled: true
    rate_limit_per_min: 20
```

#### accounts.yaml (NEW in v3)

Defines all trading accounts managed by the system. Each account has its own Dhan credentials (referenced via `.env` keys), strategy subset, risk limits, and allocation parameters.

```yaml
accounts:
  prop:
    display_name: "Proprietary"
    broker: dhan
    env_key_prefix: DHAN_PROP       # resolves to DHAN_PROP_CLIENT_ID, DHAN_PROP_ACCESS_TOKEN in .env
    capital: 5000000                 # ₹50L
    strategies:
      - S1
      - S2
      - S3
      - S4
      - S5
      - S6
      - S7
    allocation:
      method: erc                    # equal-risk-contribution
      kelly_fraction: 0.5
      max_strategy_weight: 0.25     # no single strategy > 25% of capital
      min_strategy_weight: 0.05     # floor to avoid rounding to zero
    risk:
      max_drawdown_pct: 10.0        # Positive value; code compares as negative (triggers when drawdown < -10%)
      max_daily_loss_pct: 3.0       # daily kill at -3%
      max_single_trade_loss: 100000 # ₹1L per trade
      max_open_positions: 14        # 2 per strategy
      max_notional_exposure: 25000000  # ₹2.5Cr
    overnight:
      enabled: true                  # allow overnight positions (S2, S6, S7)
      max_overnight_positions: 6

  client_001:
    display_name: "Client Alpha"
    broker: dhan
    env_key_prefix: DHAN_C001
    capital: 20000000                # ₹2Cr
    strategies:
      - S1
      - S3
      - S5
    allocation:
      method: erc
      kelly_fraction: 0.3
      max_strategy_weight: 0.40
      min_strategy_weight: 0.10
    risk:
      max_drawdown_pct: 8.0
      max_daily_loss_pct: 2.0
      max_single_trade_loss: 300000  # ₹3L per trade
      max_open_positions: 6
      max_notional_exposure: 80000000  # ₹8Cr
    overnight:
      enabled: false                  # intraday only

  client_002:
    display_name: "Client Beta"
    broker: dhan
    env_key_prefix: DHAN_C002
    capital: 10000000                # ₹1Cr
    strategies:
      - S1
      - S2
      - S6
    allocation:
      method: erc
      kelly_fraction: 0.25
      max_strategy_weight: 0.45
      min_strategy_weight: 0.10
    risk:
      max_drawdown_pct: 6.0
      max_daily_loss_pct: 1.5
      max_single_trade_loss: 100000  # ₹1L per trade
      max_open_positions: 6
      max_notional_exposure: 40000000  # ₹4Cr
    overnight:
      enabled: true
      max_overnight_positions: 4
```

#### Corresponding .env entries

```bash
# Prop account
DHAN_PROP_CLIENT_ID=1100012345
DHAN_PROP_ACCESS_TOKEN=eyJhbGciOi...

# Client Alpha
DHAN_C001_CLIENT_ID=1100067890
DHAN_C001_ACCESS_TOKEN=eyJhbGciOi...

# Client Beta
DHAN_C002_CLIENT_ID=1100099999
DHAN_C002_ACCESS_TOKEN=eyJhbGciOi...

# Telegram
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=-100123456789

# DuckDB encryption (if used)
DUCKDB_ENCRYPTION_KEY=...
```

#### strategies.yaml

```yaml
strategies:
  S1:
    name: ORB
    timeframe: 5m
    enabled: true
    product_type: MIS
    instrument: NIFTY
    params:
      lookback_bars: 3
      breakout_threshold: 0.5
      atr_period: 14
    fill:
      sl_buffer_pct: 0.3
      target_multiple: 2.0
      max_sl_distance_pct: 2.0

  S2:
    name: MeanReversion
    timeframe: 15m
    enabled: true
    product_type: NRML
    instrument: NIFTY
    params:
      zscore_entry: 2.0
      zscore_exit: 0.5
      lookback: 20
    fill:
      sl_buffer_pct: 0.5
      max_sl_distance_pct: 3.0

  S3:
    name: TrendFollower
    timeframe: 15m
    enabled: true
    product_type: MIS
    instrument: BANKNIFTY
    params:
      ema_fast: 9
      ema_slow: 21
      adx_threshold: 25
    fill:
      sl_buffer_pct: 0.3
      target_multiple: 2.5
      max_sl_distance_pct: 2.0

  S4:
    name: VWAP_Reversal
    timeframe: 5m
    enabled: true
    product_type: MIS
    instrument: NIFTY
    params:
      vwap_bands: 2.0
      volume_confirmation: true
    fill:
      sl_buffer_pct: 0.3
      target_multiple: 1.5
      max_sl_distance_pct: 1.5

  S5:
    name: ExpiryPlay
    timeframe: 5m
    enabled: true
    product_type: MIS
    instrument: NIFTY
    params:
      expiry_only: true
      gamma_threshold: 0.05
    fill:
      sl_buffer_pct: 0.5
      max_sl_distance_pct: 5.0

  S6:
    name: SwingMomentum
    timeframe: 1h
    enabled: true
    product_type: NRML
    instrument: NIFTY
    params:
      momentum_period: 10
      rsi_threshold: 60
    fill:
      sl_buffer_pct: 0.5
      target_multiple: 3.0
      max_sl_distance_pct: 4.0

  S7:
    name: VolatilityBreakout
    timeframe: 15m
    enabled: true
    product_type: NRML
    instrument: NIFTY
    params:
      atr_multiple: 1.5
      squeeze_period: 20
    fill:
      sl_buffer_pct: 0.4
      max_sl_distance_pct: 3.0
```

#### Strategy Allocation Block

Each account computes its own allocation weights at startup. The allocation block in `accounts.yaml` controls how capital is distributed across the account's enabled strategies.

```python
@dataclass
class AccountAllocation:
    account_id: str
    strategy_weights: dict[str, float]   # S1 → 0.22, S3 → 0.35, ...
    kelly_fraction: float
    computed_at: int                      # UTC epoch ms

    @classmethod
    async def compute(cls, account: AccountConfig) -> "AccountAllocation":
        """
        Compute ERC weights for this account's strategy subset.
        Uses trailing 60-day realized vol from DuckDB backtest results.
        Weights are then scaled by the account's kelly_fraction.
        """
        strategies = account.strategies
        vols = await duckdb_reader.get_trailing_vols(strategies, lookback=60)
        inv_vol = {s: 1.0 / v for s, v in vols.items() if v > 0}
        total = sum(inv_vol.values())
        raw_weights = {s: w / total for s, w in inv_vol.items()}

        # Clamp to min/max
        weights = {}
        for s, w in raw_weights.items():
            w = max(w, account.allocation.min_strategy_weight)
            w = min(w, account.allocation.max_strategy_weight)
            weights[s] = w

        # Re-normalize after clamping
        total = sum(weights.values())
        weights = {s: w / total for s, w in weights.items()}

        return cls(
            account_id=account.account_id,
            strategy_weights=weights,
            kelly_fraction=account.allocation.kelly_fraction,
            computed_at=utc_epoch_ms(),
        )
```

**Secrets:** `.env` loaded via `python-dotenv`. Contains per-account broker API keys, Telegram bot token, AWS credentials. On EC2, prefer IAM roles for S3 access (no static AWS keys). File permissions: `chmod 600 .env`.

**Hot-reload:** Strategy parameters stored in Redis `CONFIG:strategy:{sid}`. Change flow: edit YAML then `python -m live.cli reload-config` then Redis updated then strategies pick up on next bar close.

Risk limits and account-level config (capital, kelly_fraction, max_drawdown) are NOT hot-reloadable — require restart (deliberate friction for safety-critical params).

**Account-level config reload:** Adding or removing accounts requires a full restart. Changing a strategy's params within an existing account is hot-reloadable. Changing an account's capital or risk limits requires restart.

---

### C. Startup & Shutdown (Multi-Account v3)

#### Startup (08:25 → 09:20 IST)

```
08:25  Phase 1: Infrastructure
  ├─ Verify Redis Sentinel (primary + replica running)
  ├─ Verify DuckDB accessible, schema version matches
  ├─ Verify S3 credentials (or IAM role)
  ├─ Verify static IP attached (AWS metadata)
  ├─ Verify chrony sync (clock drift < 100ms)
  └─ Load config files (system.yaml, accounts.yaml, strategies.yaml, risk.yaml)

08:25  Phase 2: Multi-Account Authentication
  ├─ Load account configs from accounts.yaml
  ├─ For each account in parallel (asyncio.gather):
  │   ├─ Read credentials from .env (DHAN_{prefix}_CLIENT_ID, DHAN_{prefix}_ACCESS_TOKEN)
  │   ├─ Authenticate with Dhan API → access token
  │   ├─ Store token in Redis: AUTH:{account_id}
  │   ├─ Verify: call profile API → confirm client_id matches
  │   ├─ Verify: call fund API → confirm capital >= configured minimum
  │   └─ Log: "Account {account_id} authenticated, available margin ₹{margin}"
  ├─ If ANY account fails auth:
  │   ├─ Mark account DEGRADED
  │   ├─ Telegram WARN: "Account {id} auth failed. Continuing without it."
  │   └─ System continues with remaining accounts (partial start)
  └─ If ALL accounts fail: ABORT startup

08:35  Phase 3: Data Load (shared across accounts)
  ├─ Download broker instrument CSV → parse → build InstrumentMaster
  │   (Tuesday + fallback → disable S5 for all accounts)
  ├─ Load holiday calendar
  └─ Verify today is trading day

08:40  Phase 4: Per-Account Position Recovery
  ├─ For each authenticated account in parallel (asyncio.gather):
  │   ├─ Query Dhan positions API → overnight positions for this account
  │   ├─ Query Dhan orders API → pending/open orders
  │   ├─ Reconcile with Redis saved state: POSITION:{account_id}:*
  │   ├─ For overnight positions: verify SL orders still active (GTT)
  │   ├─ Log discrepancies: broker says X, Redis says Y
  │   ├─ BROKER WINS on conflict (broker is source of truth)
  │   └─ Initialize PositionTracker for this account
  └─ Aggregate: total positions across all accounts

08:50  Phase 5: Data Feed + Per-Account Order WS
  ├─ Connect broker data WS (SHARED — one connection, all instruments)
  ├─ Subscribe instruments
  ├─ Verify ticks flowing (wait up to 60s)
  └─ For each account in parallel (asyncio.gather):
      ├─ Connect Dhan order update WS for this account
      ├─ Verify connection: receive heartbeat
      └─ Store WS handle: WS_ORDER:{account_id} (in-memory handle, not a Redis key)

08:55  Phase 6: Components
  ├─ Initialize OMS (per-account order state, fill management loops)
  ├─ Start option chain poller (background, 5s, shared)
  ├─ For each account: compute ERC weights (AccountAllocation.compute)
  ├─ Start per-account risk monitors
  ├─ Start global risk monitor (aggregate across accounts)
  ├─ Start Prometheus metrics server
  ├─ Start audit logger (per-account partitioned)
  ├─ Start SL lifecycle periodic verification (60s, per account)
  └─ Start monitoring watchdog

09:10  Phase 7: Exchange Verification
  ├─ Holiday calendar check
  ├─ Broker market status check
  └─ If closed: abort, Telegram INFO to all account channels

09:10  Phase 8: Strategy Processes
  ├─ Create Redis consumer groups (XGROUP CREATE ... $)
  ├─ Start S1..S7 (enabled)
  ├─ Each: connect streams, restore state
  ├─ AccountManager registers which strategies serve which accounts
  └─ Strategies SUPPRESSED until 09:20

09:20  Phase 9: Go Live
  ├─ Un-suppress strategies
  ├─ OMS begins accepting orders (per-account routing)
  └─ Telegram INFO: "System live. N accounts, M strategies active, P positions carried."
```

Each phase has a health check and 5-minute timeout. Total startup budget: 55 minutes. Not ready by 09:20 → DEGRADED (manage existing positions only, no new trades, for all accounts).

**Phase timing breakdown:**

| Phase | Duration | Parallelism |
|-------|----------|-------------|
| 1. Infrastructure | 5 min | Sequential |
| 2. Auth | 5 min | N accounts in parallel |
| 3. Data Load | 5 min | Sequential (shared) |
| 4. Position Recovery | 10 min | N accounts in parallel |
| 5. Data Feed + WS | 10 min | 1 shared data WS + N order WS in parallel |
| 6. Components | 5 min | Per-account compute in parallel |
| 7. Exchange Verify | < 1 min | Sequential |
| 8. Strategy Start | 10 min | Strategies start, register per account |
| 9. Go Live | Instant | Flip flag |

**Dependency graph (multi-account):**

```
Redis ──→ Data Ingester ──→ Consumer Groups ──→ Strategies
  │                                                 │
  ├──→ Account Auth (N parallel) ──→ AccountManager ─┤
  │                                       │          │
  └──→ Position Recovery (N parallel) ──→ OMS ←──────┘
         per account                      │
                                   SL Verification
                                    (per account)
```

#### Shutdown

```
Phase 1: Block new entries (immediate)
  └─ For each account: SET HALT:{account_id}:no_new_entries

Phase 2: Flatten intraday (EOD sequence, up to 5 min)
  └─ For each account in parallel:
      ├─ Identify intraday positions (product_type=MIS)
      ├─ Cancel open orders for this account
      └─ Place exit orders (IOC, batch)

Phase 3: Convert DAY SLs to Forever/GTT for overnight positions (per account)
  └─ For each account in parallel:
      ├─ Identify overnight positions (product_type=NRML, strategy in S2/S6/S7)
      ├─ For each SL order:
      │   ├─ Cancel existing DAY SL
      │   ├─ Place Forever/GTT SL at same trigger
      │   ├─ Verify new SL is active via REST
      │   └─ If GTT placement fails: Telegram CRITICAL, log for manual review
      └─ If account.overnight.enabled == false: flatten all (should have none)

Phase 4: Save state to Redis (30s)
  └─ For each account:
      ├─ Persist POSITION:{account_id}:* → final positions
      ├─ Persist ORDER:{account_id}:* → pending SLs
      └─ Persist ALLOC:erc_weights → current allocation weights (shared, not per-account)

Phase 5: Disconnect WS, logout broker (per account, SEBI: daily logout)
  └─ For each account in parallel:
      ├─ Close order update WS
      └─ Call Dhan logout API (invalidate token)
  └─ Close shared data WS

Phase 6: Flush WAL + Parquet + audit to S3 (30s)
  ├─ Flush DuckDB WAL
  ├─ Write Parquet files (per account partitioned)
  └─ Upload to S3: s3://{bucket}/live/{date}/{account_id}/

Phase 7: SIGTERM strategy processes → 5s → SIGKILL

Phase 8: Telegram INFO + exit
  └─ "System shutdown complete. N accounts logged out.
      Overnight positions: {summary per account}"
```

**Shutdown timing budget:** 10 minutes max. If any account's SL conversion hangs beyond 3 minutes, force-proceed and alert. Overnight positions without confirmed SLs are CRITICAL-level alerts requiring manual intervention.

---

### D. State Recovery (Multi-Account)

**Redis Sentinel** protects against Redis process crashes (<1s failover). Same-host limitation acknowledged — EC2 failure kills both primary and replica. Protection in that case: server-side SLs + broker auto-square (per account).

#### Per-Account Redis Key Layout

```
AUTH:{account_id}                    → broker access token
POSITION:{account_id}:{strategy}    → position state JSON
ORDER:{account_id}:{order_id}       → order state JSON
ALLOC:erc_weights                   → allocation weights JSON (shared, not per-account)
CONFIG:strategy:{sid}               → strategy params (shared)
HALT:{account_id}:no_new_entries    → per-account halt flag
HALT:global                         → global halt (all accounts)
KILLED:{account_id}:{sid}           → per-account strategy kill
KILLED:global:{sid}                 → global strategy kill
LASTTICK:{instrument}               → latest tick (shared)
```

#### Recovery Table (Multi-Account)

| Lost State (Redis crash) | Recovery | Scope |
|--------------------------|----------|-------|
| LASTTICK | Auto-rebuilt from ticks within seconds | Shared |
| POSITION:{account_id} | Rebuilt from broker positions API per account | Per account |
| ORDER:{account_id} | Query broker orders API per account | Per account |
| AUTH:{account_id} | Re-authenticate with broker (credentials from .env) | Per account |
| CONFIG | Reload from YAML files | Shared |
| ALLOC:erc_weights | Recompute ERC from DuckDB | Shared |
| HALT flags | Default to halted (safe). Manual un-halt after review. | Per account / global |
| KILLED flags | Default to killed (safe). Manual reset after review. | Per account / global |

**Per-account recovery sequence:**

```python
async def recover_account(account_id: str) -> RecoveryResult:
    """
    Full state recovery for a single account.
    Called on startup or after Redis failover.
    """
    result = RecoveryResult(account_id=account_id)

    # 1. Re-auth if token missing
    token = await redis.get(f"AUTH:{account_id}")
    if not token:
        token = await broker.authenticate(account_id)
        await redis.set(f"AUTH:{account_id}", token, ex=86400)
        result.re_authenticated = True

    # 2. Rebuild positions from broker
    broker_positions = await broker.get_positions(account_id)
    redis_positions = await redis.hgetall(f"POSITION:{account_id}:*")

    for pos in broker_positions:
        key = f"POSITION:{account_id}:{pos.strategy_tag}"
        await redis.set(key, pos.to_json())

    # 3. Reconcile — log discrepancies
    result.discrepancies = reconcile(broker_positions, redis_positions)
    if result.discrepancies:
        await telegram.send(
            f"WARN: Account {account_id} position discrepancy: "
            f"{len(result.discrepancies)} mismatches. Broker state used."
        )

    # 4. Rebuild orders
    broker_orders = await broker.get_orders(account_id)
    for order in broker_orders:
        if order.status in ("PENDING", "OPEN", "TRIGGER_PENDING"):
            await redis.set(f"ORDER:{account_id}:{order.order_id}", order.to_json())

    # 5. Recompute allocation
    alloc = await AccountAllocation.compute(accounts[account_id])
    await redis.set("ALLOC:erc_weights", alloc.to_json())

    return result
```

**OMS degraded mode** covers the gap: fill management continues via broker WS + REST per account. New entries blocked. Server-side SLs active. Degraded mode is per-account — one account's recovery does not block others.

**DuckDB:** RDB-style backup to S3 nightly at 16:30 IST. RPO: 1 day (trading data is reconstructible from broker records). RTO: <5 minutes (restore from S3 + replay today's fills from audit log).

---

### E. DuckDB Architecture

#### Single-Writer Pattern

DuckDB does not support concurrent writes. All writes go through a single `DuckDBWriter` asyncio task that consumes from an internal queue.

```python
class DuckDBWriter:
    """
    Single-writer task for DuckDB.
    All components enqueue writes; this task drains and commits.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.queue: asyncio.Queue[WriteOp] = asyncio.Queue(maxsize=10_000)
        self.conn: duckdb.DuckDBPyConnection | None = None

    async def start(self):
        self.conn = duckdb.connect(self.db_path)
        self._apply_migrations()
        asyncio.create_task(self._drain_loop())

    async def _drain_loop(self):
        """Drain queue in batches. Commit every 100 ops or 1 second."""
        batch: list[WriteOp] = []
        while True:
            try:
                op = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                batch.append(op)
                # Drain remaining without blocking
                while not self.queue.empty() and len(batch) < 100:
                    batch.append(self.queue.get_nowait())
            except asyncio.TimeoutError:
                pass

            if batch:
                await self._execute_batch(batch)
                batch.clear()

    async def _execute_batch(self, batch: list[WriteOp]):
        try:
            self.conn.begin()
            for op in batch:
                self.conn.execute(op.sql, op.params)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            logger.error(f"DuckDB batch write failed: {e}")
            for op in batch:
                op.error_callback(e)

    async def enqueue(self, sql: str, params: tuple = (), callback=None):
        await self.queue.put(WriteOp(sql=sql, params=params, callback=callback))
```

#### Per-Account Tables

```sql
-- Trades table: partitioned by account
CREATE TABLE IF NOT EXISTS trades (
    trade_id        VARCHAR PRIMARY KEY,
    account_id      VARCHAR NOT NULL,
    strategy_id     VARCHAR NOT NULL,
    instrument      VARCHAR NOT NULL,
    direction       VARCHAR NOT NULL,  -- BUY / SELL
    quantity         INTEGER NOT NULL,
    entry_price     DOUBLE NOT NULL,
    exit_price      DOUBLE,
    entry_time      BIGINT NOT NULL,   -- UTC epoch ms
    exit_time       BIGINT,
    pnl_gross       DOUBLE,
    pnl_net         DOUBLE,            -- after costs
    costs_total     DOUBLE,
    sl_price        DOUBLE,
    sl_order_id     VARCHAR,
    status          VARCHAR NOT NULL,  -- OPEN / CLOSED / CANCELLED
    created_at      BIGINT NOT NULL
);

CREATE INDEX idx_trades_account ON trades(account_id);
CREATE INDEX idx_trades_strategy ON trades(account_id, strategy_id);
CREATE INDEX idx_trades_date ON trades(entry_time);

-- Orders table: per-account audit trail
CREATE TABLE IF NOT EXISTS orders (
    order_id        VARCHAR PRIMARY KEY,
    account_id      VARCHAR NOT NULL,
    broker_order_id VARCHAR,
    strategy_id     VARCHAR NOT NULL,
    instrument      VARCHAR NOT NULL,
    order_type      VARCHAR NOT NULL,
    direction       VARCHAR NOT NULL,
    quantity         INTEGER NOT NULL,
    price           DOUBLE,
    trigger_price   DOUBLE,
    status          VARCHAR NOT NULL,
    placed_at       BIGINT NOT NULL,
    filled_at       BIGINT,
    fill_price      DOUBLE,
    fill_quantity    INTEGER,
    reject_reason   VARCHAR,
    created_at      BIGINT NOT NULL
);

CREATE INDEX idx_orders_account ON orders(account_id);

-- Daily PnL: per-account daily summary
CREATE TABLE IF NOT EXISTS daily_pnl (
    date            DATE NOT NULL,
    account_id      VARCHAR NOT NULL,
    strategy_id     VARCHAR NOT NULL,
    gross_pnl       DOUBLE NOT NULL,
    net_pnl         DOUBLE NOT NULL,
    costs           DOUBLE NOT NULL,
    num_trades      INTEGER NOT NULL,
    win_trades      INTEGER NOT NULL,
    max_drawdown    DOUBLE,
    PRIMARY KEY (date, account_id, strategy_id)
);

-- Account snapshots: end-of-day capital state
CREATE TABLE IF NOT EXISTS account_snapshots (
    date            DATE NOT NULL,
    account_id      VARCHAR NOT NULL,
    capital_start   DOUBLE NOT NULL,
    capital_end     DOUBLE NOT NULL,
    total_pnl       DOUBLE NOT NULL,
    drawdown_pct    DOUBLE NOT NULL,
    positions_held  INTEGER NOT NULL,
    PRIMARY KEY (date, account_id)
);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version         INTEGER PRIMARY KEY,
    applied_at      BIGINT NOT NULL,
    description     VARCHAR NOT NULL
);
```

#### Schema Versioning

```python
MIGRATIONS = [
    Migration(
        version=1,
        description="Initial schema: trades, orders, daily_pnl",
        up="""
            CREATE TABLE IF NOT EXISTS trades (...);
            CREATE TABLE IF NOT EXISTS orders (...);
            CREATE TABLE IF NOT EXISTS daily_pnl (...);
            CREATE TABLE IF NOT EXISTS schema_version (...);
        """,
    ),
    Migration(
        version=2,
        description="Add account_snapshots table for multi-account tracking",
        up="""
            CREATE TABLE IF NOT EXISTS account_snapshots (...);
        """,
    ),
    Migration(
        version=3,
        description="Add account_id index on trades",
        up="""
            CREATE INDEX IF NOT EXISTS idx_trades_account ON trades(account_id);
        """,
    ),
]


class SchemaMigrator:
    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def current_version(self) -> int:
        try:
            result = self.conn.execute(
                "SELECT MAX(version) FROM schema_version"
            ).fetchone()
            return result[0] or 0
        except Exception:
            return 0

    def migrate(self):
        current = self.current_version()
        for migration in MIGRATIONS:
            if migration.version > current:
                logger.info(f"Applying migration v{migration.version}: {migration.description}")
                self.conn.execute(migration.up)
                self.conn.execute(
                    "INSERT INTO schema_version VALUES (?, ?, ?)",
                    [migration.version, utc_epoch_ms(), migration.description],
                )
                self.conn.commit()
                logger.info(f"Migration v{migration.version} applied")
```

**Migration rules:**
- Migrations are append-only. Never modify an existing migration.
- All migrations must be idempotent (`IF NOT EXISTS`, `IF NOT EXISTS`).
- Test migrations on a copy of production DuckDB before deploying.
- Backup DuckDB to S3 before applying migrations.
- Migrations run during Phase 1 of startup, before any trading components start.

---

## 4. TESTING & SECURITY

### Testing Strategy

| Level | What | How |
|-------|------|-----|
| Unit | Pydantic models, bar builder, cost model, strike resolution, AccountConfig validation | pytest, property-based tests (hypothesis) for edge cases |
| Integration | Signal → resolve → risk gate → OMS mock (single account) | pytest with mock broker adapter |
| Multi-account integration | Same signal → fan-out to N accounts → verify independent lot sizing, independent SLs | pytest with N mock broker adapters |
| Replay | WAL/Parquet replay through full pipeline | Replay framework: read historical ticks, feed through strategies, compare signals |
| Paper trading (per account) | Full system, real market data, mock execution, per-account paper credentials | `TRADING_ENV=paper` → OMS logs orders but does not call broker API, per account |
| Chaos | Redis kill mid-session, WS drop mid-order, duplicate WS messages, single account auth failure | Scripted chaos tests run weekly during paper trading phase |
| Multi-account chaos | Kill one account's order WS while others are live, auth expiry for one account mid-session | Verify isolation: one account's failure does not affect others |
| OMS state machine | All order status transitions, partial fills, race conditions | Property tests: fuzz OrderState transitions, verify invariants |
| Account divergence | Simulate lot rounding producing different position counts across accounts | Verify divergence metrics fire, alerts sent |

**Paper vs prod separation:** Environment flag `TRADING_ENV=paper|live` controls:
- `paper`: OMS simulates fills (random delay 50-500ms, 90% fill rate). No real API calls. Uses paper broker credentials. All accounts use simulated execution.
- `live`: Real broker API. Real money. Real consequences.

Both share the same code path — only the `BrokerAdapter` implementation differs. Per-account paper trading is supported: set individual accounts to paper mode while others remain live (useful for onboarding new client accounts).

```yaml
# Per-account paper mode override
accounts:
  client_003:
    trading_env: paper    # overrides global TRADING_ENV for this account only
    # ... rest of config
```

### Multi-Account Test Scenarios

| Scenario | Expected Behavior | Test Method |
|----------|-------------------|-------------|
| S1 fires signal, 3 accounts subscribe | 3 independent orders placed, lot sizes differ by capital | Integration test |
| Account "client_001" auth expires mid-day | client_001 enters degraded mode, prop + client_002 unaffected | Chaos test |
| Global kill fires | All accounts flatten, all orders cancelled | Integration test |
| Per-account kill fires for "prop" | Only prop flattens, clients unaffected | Integration test |
| Lot rounding gives 0 lots for small account | Order skipped, metric incremented, no error | Unit test |
| Position recovery disagrees with broker | Broker wins, discrepancy logged per account | Integration test |
| Two accounts fill at different prices | Independent PnL tracking, no cross-contamination | Integration test |

### Security

| Concern | Approach |
|---------|----------|
| Secrets | `.env` file, `chmod 600`. Per-account credentials keyed by prefix (DHAN_PROP_, DHAN_C001_, etc). IAM roles for S3. |
| Credential isolation | Each account's token stored separately in Redis (AUTH:{account_id}). One account's token compromise does not expose others. |
| Network | EC2 security group: inbound SSH (IP-restricted) + Grafana (IP-restricted) only. Redis on localhost only. |
| Broker API | TLS only. No TLS pinning (broker certs rotate). Validate cert chain. Per-account TLS sessions. |
| Authentication | Broker tokens: 24hr validity, rotated daily per account. Stored in Redis (localhost only). |
| Access control | `reload-config` and `kill` CLI require SSH to EC2. Telegram `/kill` requires bot auth. Per-account kills require account_id parameter. |
| Account separation | No cross-account order routing. OMS validates account_id on every order before submission. |
| Logging | No secrets in logs. Audit log contains order details but not API tokens. Logs are per-account partitioned. |
| Multi-account audit | SEBI 5-year trail per account. Each account's trades stored with account_id in DuckDB. |

### Runbooks

| Scenario | Runbook |
|----------|---------|
| System won't start | Check Redis, broker auth (all accounts), instrument CSV. Logs in `/var/log/trading/`. |
| Single account auth failure | Telegram WARN. Account enters DEGRADED. Other accounts unaffected. Re-auth: `python -m live.cli auth --account {id}`. |
| WS disconnected mid-session | Auto-reconnect. If persistent: check broker status page. Manual: restart data ingester. |
| Account order WS disconnected | Auto-reconnect for that account. Other accounts' order feeds unaffected. |
| Position discrepancy | Telegram CRITICAL with account_id. Auto-sync from broker for that account. Review audit log. |
| Kill condition fires (per account) | Telegram INFO. Strategy stops new entries for that account. Review: `python -m live.cli strategy-status --account {id}`. Resume: `DEL KILLED:{account_id}:{sid}`. |
| Kill condition fires (global) | All accounts affected. Review: `python -m live.cli status`. Resume: `DEL HALT:global` + restart. |
| Global kill fired | Everything cancelled + flattened across all accounts. Resume: full restart after review. |
| Account divergence alert | Check position counts. If divergence > threshold, investigate lot rounding or partial fills. |
| Post-incident | Reconcile audit log with broker contract notes PER ACCOUNT. File incident report. |

---

## 5. OPEN QUESTIONS & MUST-PROTOTYPE

### Must Validate During Paper Trading

1. **Broker modify qty semantics.** Place 130, partial 65, modify with qty=130. Does remaining = 65 or 130? System-breaking if wrong.
2. **Broker option chain response schema.** Verify Greeks are present. If not, validate local BS computation.
3. **GTT/Forever order support.** Verify overnight SL via GTT works as expected. Verify AMO as fallback. Test per-account GTT limits.
4. **SL trigger behavior when system is offline.** Confirm SL-Limit triggers and fills on broker infra without our WS connected.
5. **Redis Streams throughput.** Measure XADD/XREADGROUP with 7 consumer groups at 500+ ticks/s burst.
6. **Option chain API P99 latency.** If >500ms, the on-demand call needs a tighter timeout or hybrid approach (use 1s-old cache + LTP delta sanity check).
7. **0-DTE spread behavior.** Measure actual NIFTY weekly option spreads on 4 consecutive Tuesdays.
8. **End-to-end latency.** Signal fire → order on exchange. Target: P99 < 1s. Measure with N accounts (expect N x single-account latency due to serial broker API calls).
9. **Multi-account fan-out latency.** Signal → N account orders placed. Measure total time for 3, 5, 10 accounts. If >2s for 10 accounts, consider parallel broker API calls.
10. **Per-account Dhan OPS consumption.** Verify: do N accounts each get their own 10 OPS limit, or is it shared across the same API key? If shared, this is a binding constraint.
11. **Per-account WS connection limits.** Verify Dhan allows N simultaneous order-update WS connections from the same IP.
12. **Lot rounding at small capital.** For a ₹10L account running S5 (NIFTY weekly options), verify lot sizing does not round to 0 for all reasonable signals. Minimum viable capital per strategy.

### Multi-Account Specific Questions

| Question | Impact | Resolution Path |
|----------|--------|-----------------|
| How much position divergence across accounts is acceptable? | If S1 gives 3 lots to prop and 1 lot to client_001, timing differences may cause one to fill and one to reject. Is this OK? | Define divergence threshold per strategy. Alert if fill rate differs >20% across accounts for same signal. |
| Should a strategy kill apply to all accounts or per-account? | Per-account kill means prop could keep trading S1 while client_001 is killed. Global kill means one account's problem stops everyone. | Default: per-account kills for risk limits, global kills for system-level issues (WS down, data feed dead). |
| Per-account vs shared rate limiter? | If each account has its own OPS limit, rate limiting should be per-account. If shared, need global rate limiter with per-account fairness. | Prototype during paper trading. Dhan docs unclear — test empirically. |
| Account onboarding without restart? | Adding a new account mid-day requires auth + position recovery + WS connect. Safe to do live? | v3.0: restart required. v3.1: hot-add with safety checks (no trades for 5 min after add). |
| Per-account Telegram channels? | Prop owner vs client notifications may need separation. | v3.0: single channel with account_id tags. v3.1: per-account channels configurable. |

### Architecture Decisions and Rationale (v3)

| Decision | Why |
|----------|-----|
| Single broker (Dhan) | Eliminates cross-broker symbol mapping, split auth, position tracking complexity. All accounts on same broker. |
| Redis Streams (not pub/sub) | Persistent, backpressure-aware, consumer groups recover after disconnect |
| On-demand option chain at signal | 200ms << 5-10s repricing from stale data |
| Mandatory server-side SL | Only protection during system outage. Per-account SLs. |
| SL as first-class lifecycle object | SL assumed correct = SL wrong. Verify constantly. Per account. |
| Priority rate limiter | Exit orders must never be starved. Per-account queues with global coordination. |
| asyncio.Lock (not threading.Lock) | threading.Lock blocks entire event loop |
| DuckDB single-writer | DuckDB does not support concurrent writes. All accounts write through one queue. |
| Same-host Redis Sentinel | Process-level HA. Host-level HA via SL + broker auto-square per account. |
| Signal dedup in router | Defense-in-depth against strategy bugs. Dedup is per-strategy, fan-out is per-account. |
| Position-aware allocation | Prevents double positions. Checked per-account. |
| Post-modify REST verification | Brokers silently reject or partially apply modifications. Per-account verification. |
| Tick WAL | Parquet buffer survives process crash. Shared across accounts (data feed is shared). |
| Shared strategies, per-account execution | Strategies compute signals once. AccountManager fans out to N accounts with independent sizing. Avoids N copies of strategy state. |
| Per-account risk monitors | One account breaching max_drawdown must not affect others. Isolation is non-negotiable. |
| Broker source of truth for positions | On any discrepancy between Redis and broker, broker wins. Per-account reconciliation. |

### Scaling Triggers (v3)

| Trigger | Action |
|---------|--------|
| 10 OPS per-account binding constraint | Switch to Upstox (50 OPS) or register algos for higher limits |
| Total OPS across N accounts saturates | Stagger order placement across accounts (50ms gaps) or parallel API calls |
| WS instrument limit reached | Add second WS connection or switch to broker with higher limit |
| Single EC2 CPU saturated (7 strategies + N accounts + Redis) | Split: Redis to managed service, strategies to second instance |
| 5+ accounts | Dedicated order management process per account group |
| 10+ accounts | Evaluate PMS/AIF registration. Separate EC2 per account group for isolation. |
| ₹5Cr+ aggregate capital | Redis on separate instance. Consider hot-standby EC2. |
| ₹25Cr+ aggregate capital | Multi-AZ deployment. True cross-host HA. Per-account EC2 instances. |

---

## 6. IMPLEMENTATION PHASES

```
Phase 1 (Week 1-2): Skeleton
  - Pydantic models for all interfaces (including AccountConfig)
  - Redis Sentinel setup
  - BrokerAdapter (auth + place + modify + cancel + status + SL)
  - Broker WS connection (ticks + order updates)
  - InstrumentMaster loader

Phase 2 (Week 3-4): Core Loop
  - On-demand option chain resolution
  - OMS with fill management + per-order locks
  - SL lifecycle manager
  - Priority rate limiter
  - Position tracker (DuckDB single-writer)
  - DuckDB schema + migration framework

Phase 3 (Week 5-6): Strategy Integration
  - LiveStrategy base + BarBuilder
  - Redis Streams consumer groups
  - Port S1 (ORB) as first live strategy
  - Signal dedup + position-aware allocator
  - Pre-trade risk gate (incl. broker position check)
  - Audit logger

Phase 4 (Week 7-8): Safety
  - Post-order reconciliation
  - SL periodic verification
  - EOD flatten (batch + parallel + IOC)
  - Overnight SL conversion (GTT/AMO)
  - OMS degraded mode
  - Global kill switch
  - Kill conditions

Phase 5 (Week 9-10): Monitoring
  - Prometheus + Grafana dashboards
  - Telegram notifications
  - Exchange open verification
  - Monitoring self-check

Phase 6 (Week 11-12): All Strategies + Hardening
  - Port S2-S7
  - ERC allocation + Kelly ramp
  - Paper trading (full system, single account, 2+ weeks)
  - Chaos testing (single account)
  - Runbook validation

Phase 7 (Week 13-15): Multi-Account Layer (NEW)
  - AccountManager class: load accounts.yaml, manage lifecycle
  - Per-account BrokerAdapter instances (auth, tokens, WS)
  - Signal fan-out: strategy signal → AccountManager → per-account lot sizing
  - Per-account position tracking in DuckDB (account_id column)
  - Per-account risk monitors (max_drawdown, daily_loss per account)
  - Per-account Redis key namespacing (POSITION:{account_id}:*)
  - Per-account SL lifecycle (independent SL orders per account)
  - Per-account EOD flatten + overnight SL conversion
  - Startup sequence update: parallel auth, parallel position recovery
  - Shutdown sequence update: per-account logout
  - CLI updates: --account flag for all commands
  - Telegram updates: account_id in all messages

Phase 8 (Week 16-18): Multi-Account Hardening (NEW)
  - Paper trading with 3 accounts simultaneously (2+ weeks)
  - Chaos testing with N accounts:
    ├─ Kill one account's order WS, verify others unaffected
    ├─ Expire one account's auth token mid-session
    ├─ Redis failover with N accounts' state
    ├─ Simultaneous signal to N accounts, verify all fill independently
    └─ Network partition simulation (one account's API unreachable)
  - Divergence monitoring:
    ├─ Track fill rates per account per strategy
    ├─ Alert if position count diverges across accounts for same strategy
    ├─ Dashboard: side-by-side account PnL comparison
    └─ Automated divergence report (daily)
  - Per-account Grafana dashboards
  - Per-account audit trail verification (SEBI 5-year)
  - Load testing: simulate 10 accounts, measure latency
  - Runbook validation with multi-account scenarios
  - Account onboarding/offboarding runbook
```

**Phase dependencies:**

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6
                                                        │
                                                        ▼
                                                    Phase 7 → Phase 8
```

Phases 1-6 deliver a fully functional single-account system. Phase 7 adds the multi-account layer on top. Phase 8 hardens it. This ordering means the single-account system is production-ready before multi-account complexity is introduced.

**Go-live criteria per phase:**

| Phase | Gate |
|-------|------|
| Phase 6 complete | Single-account paper trading for 2 weeks with zero CRITICAL alerts |
| Phase 6 → Live | Single-account live with ₹5L capital for 2 weeks |
| Phase 7 complete | Multi-account paper trading for 2 weeks, divergence < 5% |
| Phase 8 → Live | Multi-account live with prop account (₹50L) + 1 client (₹10L) for 2 weeks |

---

## 7. REGULATORY NOTES

### General SEBI Compliance

This architecture addresses known SEBI algo trading requirements:

| Requirement | Implementation |
|-------------|----------------|
| Static IP | AWS Elastic IP attached to EC2 instance |
| Limit orders only | OMS enforces LIMIT order type, rejects MARKET orders |
| Daily logout | Shutdown Phase 5: per-account Dhan logout API call |
| 5-year audit trail | DuckDB + S3 Parquet archives, per-account partitioned |
| Server in India | AWS ap-south-1 (Mumbai) |
| Order-level audit | Every order logged with timestamp, account, strategy, instrument, price, quantity |

### Multi-Account (PMS/AIF) Considerations

Operating multiple client accounts through a single algorithmic system has regulatory implications beyond standard algo trading rules.

| Concern | Guidance |
|---------|----------|
| PMS registration | Managing third-party funds with discretionary authority requires SEBI PMS registration (minimum AUM ₹50Cr, net worth ₹5Cr). If managing fewer than 3 clients informally, consult compliance counsel on whether PMS applies. |
| AIF classification | If pooling funds from multiple investors, AIF Category III registration may be required. The multi-account architecture (separate accounts, no pooling) may avoid this — confirm with counsel. |
| Best execution | SEBI requires best execution for client orders. Since all accounts trade the same instruments at the same time, document that signal-to-order latency is uniform across accounts (no preferential ordering). |
| Fair allocation | When a signal fires and multiple accounts subscribe, lot allocation must be fair and pre-determined. The ERC + kelly_fraction method provides a documented, deterministic allocation. No manual override of allocation during trading hours. |
| Front-running | Prop account must not systematically trade before client accounts on the same signal. Implementation: fan-out to all accounts simultaneously via asyncio.gather. Audit log timestamps verify no systematic delay. |
| Account segregation | Each account must have its own Dhan trading account with separate credentials. No co-mingling of funds. DuckDB stores per-account records. |
| Client reporting | Per-account PnL, trade log, and position reports must be available. DuckDB queries filtered by account_id. |
| Risk disclosure | Each client account's risk parameters (max_drawdown, kelly_fraction) must be agreed upon and documented before trading begins. Stored in accounts.yaml. |

**Order of execution fairness:**

```python
async def fan_out_signal(signal: Signal, accounts: list[AccountConfig]):
    """
    Fan out a signal to all subscribed accounts simultaneously.
    asyncio.gather ensures no account is systematically first.
    """
    tasks = []
    for account in accounts:
        if signal.strategy_id in account.strategies:
            tasks.append(
                place_order_for_account(signal, account)
            )
    # All orders dispatched in parallel — no preferential ordering
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Log timing for audit
    for account, result in zip(accounts, results):
        audit_log.record(
            event="SIGNAL_FANOUT",
            signal_id=signal.signal_id,
            account_id=account.account_id,
            dispatched_at=utc_epoch_ms(),
            result=str(result),
        )
```

**Actual regulatory obligations depend on entity classification, number of clients, and broker arrangements. Confirm with compliance counsel before deploying real capital or managing third-party funds. The architecture provides the technical infrastructure for compliance but does not constitute legal advice.**

---

*This document specifies cross-cutting concerns, testing strategy, open questions, and implementation phases for the v3 multi-account live trading system. Every account is independently managed with its own credentials, risk limits, and position tracking. Server-side stop-losses ensure no position in any account is ever unprotected. The system is designed for 3 accounts at ₹3.5Cr aggregate and scales to 10+ accounts at ₹25Cr+ with the triggers documented above.*

---

