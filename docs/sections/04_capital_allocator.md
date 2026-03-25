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
