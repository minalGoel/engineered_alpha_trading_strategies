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
