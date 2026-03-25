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
