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
