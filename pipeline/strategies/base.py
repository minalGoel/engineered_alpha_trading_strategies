"""Base interface for hand-implemented strategies.

Every strategy file must export a `Strategy` class inheriting from `BaseStrategy`.
The `compute()` method takes spot, option, and VIX DataFrames and returns an
`OptionSignals` dataclass whose fields map to what the Numba state machine expects.

This pipeline trades NIFTY and BANKNIFTY index options at 5-second frequency.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from dataclasses import dataclass, field


@dataclass
class OptionSignals:
    """Output of a strategy's compute() method for option trading.

    Arrays must all have length == len(spot_df).
    """
    # ── Entry signals (length n) ──
    buy_ce: np.ndarray          # bool — buy call entry (bullish)
    buy_pe: np.ndarray          # bool — buy put entry (bearish)

    # ── Exit signals (length n) ──
    sell_ce: np.ndarray         # bool — signal-based call exit
    sell_pe: np.ndarray         # bool — signal-based put exit

    # ── Per-bar stop/target in option premium points (length n) ──
    stop_points: np.ndarray     # float64 — stop in premium points (0.0 = no stop)
    target_points: np.ndarray   # float64 — target in premium points (0.0 = no target)

    # ── Strike selection (length n) ──
    strike_offset: np.ndarray   # int32 — offset from ATM (0=ATM, 1=ATM+1step, -1=ATM-1step)

    # ── Scalar parameters ──
    time_stop_bars: int = 24          # max bars before time stop (1 bar=5s, 24=120s)
    max_trades_per_day: int = 10


@dataclass
class TunableParam:
    """A parameter that Optuna can optimize."""
    name: str       # e.g., "momentum_threshold"
    default: float  # e.g., 0.05
    low: float      # e.g., 0.01
    high: float     # e.g., 0.15


class BaseStrategy:
    """Every strategy implements this."""

    name: str = ""
    underlying: str = "BOTH"          # "NIFTY", "BANKNIFTY", or "BOTH"
    session_start_minutes: int = 560  # 09:20 IST in minutes from midnight
    session_end_minutes: int = 925    # 15:25 IST in minutes from midnight
    max_trades_per_day: int = 10
    unparseable: bool = False
    assumptions: list[str] = []

    # Pipeline-level attributes
    timeframe: str = "5s"
    max_lookback: int = 120           # warmup bars needed (120 × 5s = 10 min)

    def tunable_params(self) -> list[TunableParam]:
        """Return list of parameters Optuna can tune.

        Only numeric THRESHOLDS, not indicator periods.
        """
        return []

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        """Compute indicators and return entry/exit signals.

        Args:
            spot_df: Polars DataFrame with columns:
                datetime, open, high, low, close, volume,
                day_id, time_minutes, atm_strike
                (NIFTY50 or BANKNIFTY index 5-second bars, IST timestamps)
            option_df: Polars DataFrame with columns:
                symbol, underlying, session_date, expiry, strike, option_type,
                datetime, open, high, low, close, volume, open_interest
                (option chain for the underlying, 5-second bars)
            vix_df: Polars DataFrame with columns:
                datetime, open, high, low, close, volume, day_id, time_minutes
                (India VIX 5-second bars)
            params: Current parameter values (defaults or Optuna suggestions).

        Returns:
            OptionSignals whose arrays are all length len(spot_df).
        """
        raise NotImplementedError
