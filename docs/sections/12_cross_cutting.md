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
