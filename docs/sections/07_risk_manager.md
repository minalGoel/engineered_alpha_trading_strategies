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
