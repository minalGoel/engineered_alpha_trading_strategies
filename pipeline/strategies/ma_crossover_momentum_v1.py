"""
ma_crossover_momentum_v1 — EMA Crossover Momentum on NIFTY (5-second bars)

Converted from: trading_strategies/unique_strategies_all/Strategy_17.json
Original: EMA(9/21) crossover momentum on top-50 FnO stocks, 1-min bars, 10-35 min hold.

Mechanism:
On NIFTY, when the 2-minute EMA crosses above the 10-minute EMA with positive 1-minute
spot return, multiple VWAP-benchmarked and systematic momentum algorithms simultaneously
detect a directional regime shift and start building positions in parallel. This synchronous
institutional order flow creates a self-reinforcing burst lasting 30-120 seconds. The
freshness filter (crossover within last 5 minutes) prevents entry into stale trends; only
the early acceleration phase is targeted.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(close: np.ndarray, period: int) -> np.ndarray:
    """Compute exponential moving average."""
    alpha = 2.0 / (period + 1)
    result = np.empty(len(close))
    result[0] = close[0]
    for i in range(1, len(close)):
        result[i] = alpha * close[i] + (1.0 - alpha) * result[i - 1]
    return result


def _crossover_age(fast: np.ndarray, slow: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    For each bar, return the number of bars since the LAST upward crossover (fast > slow
    while prev bar had fast <= slow) and the LAST downward crossover.
    Returns (age_up, age_down) arrays of dtype int32, values 9999 if no crossover seen yet.
    """
    n = len(fast)
    age_up = np.full(n, 9999, dtype=np.int32)
    age_down = np.full(n, 9999, dtype=np.int32)

    last_up = -9999
    last_down = -9999

    for i in range(1, n):
        was_bullish = fast[i - 1] > slow[i - 1]
        is_bullish = fast[i] > slow[i]

        if is_bullish and not was_bullish:
            last_up = i
        if not is_bullish and was_bullish:
            last_down = i

        if last_up >= 0:
            age_up[i] = i - last_up
        if last_down >= 0:
            age_down[i] = i - last_down

    return age_up, age_down


class Strategy(BaseStrategy):
    name = "ma_crossover_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 120            # 10-min warmup (slow EMA period)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ret_threshold", 0.0003, 0.0001, 0.0010),
            TunableParam("vix_max", 20.0, 15.0, 25.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Parameters ──────────────────────────────────────────────────────────
        ret_threshold = params.get("ret_threshold", 0.0003)
        vix_max = params.get("vix_max", 20.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)
        crossover_freshness = 60  # bars = 5 minutes

        # ── Spot close (forward-fill NaN before converting) ──────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── VIX filter ───────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── EMA indicators ───────────────────────────────────────────────────
        # EMA(24) = 2-min fast: detects short-term momentum shifts
        # EMA(120) = 10-min slow: provides trend context (NOT 9×12 blind scale)
        fast_period = 24
        slow_period = 120

        ema_fast = _ema(close, fast_period)
        ema_slow = _ema(close, slow_period)

        # ── 1-minute return (12 bars × 5s = 60s) ────────────────────────────
        # Replaces original rel_volume (unreliable on NIFTY index at 5s)
        ret_12 = np.zeros(n)
        ret_12[12:] = (close[12:] - close[:-12]) / np.where(close[:-12] != 0, close[:-12], 1.0)

        # ── Crossover freshness ──────────────────────────────────────────────
        # Only trade within 5 minutes of a fresh crossover (early momentum phase)
        age_up, age_down = _crossover_age(ema_fast, ema_slow)

        # ── Masks ────────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < vix_max
        warmed_up = np.arange(n) >= slow_period

        ema_bullish = ema_fast > ema_slow
        ema_bearish = ema_fast < ema_slow
        fresh_up = age_up <= crossover_freshness
        fresh_down = age_down <= crossover_freshness

        buy_ce = (
            in_session
            & vix_ok
            & warmed_up
            & ema_bullish
            & fresh_up
            & (ret_12 > ret_threshold)
        )

        buy_pe = (
            in_session
            & vix_ok
            & warmed_up
            & ema_bearish
            & fresh_down
            & (ret_12 < -ret_threshold)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
