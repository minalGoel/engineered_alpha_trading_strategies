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
