"""Base interface for hand-implemented strategies.

Every strategy file must export a `Strategy` class inheriting from `BaseStrategy`.
The `compute()` method takes a Polars DataFrame of OHLCV data and returns a
`StrategySignals` dataclass whose fields map directly to what the Numba state
machine in `pipeline.state_machine` expects.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from dataclasses import dataclass, field


@dataclass
class StrategySignals:
    """Output of a strategy's compute() method.

    Arrays must all have length == len(df).
    Scalar exit parameters map directly to run_state_machine() arguments.
    """
    # ── Entry/exit boolean masks (length n) ──
    long_entry: np.ndarray        # bool — True on bars where BUY fires
    short_entry: np.ndarray       # bool — True on bars where SHORT fires
    signal_exit_long: np.ndarray  # bool — signal-based SELL
    signal_exit_short: np.ndarray # bool — signal-based COVER

    # ── Indicator arrays for the state machine (length n) ──
    atr_arr: np.ndarray           # float64 — ATR values (zeros if unused)
    target_indicator: np.ndarray  # float64 — VWAP/EMA for indicator-touch targets (zeros if unused)

    # ── Scalar exit parameters ──
    stop_loss_pct: float = 0.0        # decimal (0.003 = 0.3%)
    stop_loss_atr_mult: float = 0.0   # ATR multiplier for stops (0 = not used)
    target_pct: float = 0.0           # decimal (0 = no % target)
    target_atr_mult: float = 0.0      # ATR multiplier for targets (0 = not used)
    use_target_indicator: bool = False # True if target is VWAP/EMA touch
    trailing_stop_pct: float = 0.0    # decimal (0 = no trailing)
    trailing_activate_pct: float = 0.0
    breakeven_pct: float = 0.0
    time_stop_bars: int = 0           # 0 = use DEFAULT_MAX_HOLD_BARS


@dataclass
class TunableParam:
    """A parameter that Optuna can optimize."""
    name: str       # e.g., "rsi_threshold"
    default: float  # e.g., 30.0
    low: float      # e.g., 15.0
    high: float     # e.g., 45.0


class BaseStrategy:
    """Every strategy implements this."""

    name: str = ""
    is_long_only: bool = False
    session_start: int = 555    # 09:15 IST in minutes from midnight
    session_end: int = 915      # 15:15 IST in minutes from midnight
    max_trades_per_day: int = 10
    unparseable: bool = False
    assumptions: list[str] = []

    # Pipeline-level attributes (used by phases.py / optimizer.py)
    timeframe: str = "1min"
    capital_per_trade: float = 100000.0
    needs_index: bool = True  # safe default — loads index data
    max_lookback: int = 200   # warmup bars needed for indicators

    def tunable_params(self) -> list[TunableParam]:
        """Return list of parameters Optuna can tune.

        Only numeric THRESHOLDS, not indicator periods.
        """
        return []

    def compute(
        self,
        df: pl.DataFrame,
        params: dict[str, float],
    ) -> StrategySignals:
        """Compute indicators and return entry/exit signals.

        Args:
            df: Polars DataFrame with columns:
                datetime, open, high, low, close, volume, vix,
                index_close, day_id, time_minutes
            params: Current parameter values (defaults or Optuna suggestions).

        Returns:
            StrategySignals whose arrays are all length len(df).
        """
        raise NotImplementedError
