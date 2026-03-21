"""European Open Momentum Continuation — NIFTY index options, 5-second bars.

When London opens at 13:30 IST, European institutional desks route directional
orders into NIFTY's ADR/GDR-linked heavyweights (HDFC Bank, RELIANCE, ICICI Bank,
INFOSYS, TCS), which together constitute ~38% of NIFTY's weight. This creates a
measurable directional impulse on the NIFTY index for 30-120 seconds. We enter in
the direction of the impulse when 30-minute pre-European-open momentum and session
VWAP alignment both confirm the directional bias.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_session_vwap(
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    volume_arr: np.ndarray,
    day_id_arr: np.ndarray,
) -> np.ndarray:
    """Session VWAP resetting at each day boundary."""
    n = len(close_arr)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_vol = 0.0
    prev_day = -999999
    for i in range(n):
        if day_id_arr[i] != prev_day:
            cum_pv = 0.0
            cum_vol = 0.0
            prev_day = day_id_arr[i]
        typical = (high_arr[i] + low_arr[i] + close_arr[i]) / 3.0
        vol = volume_arr[i]
        cum_pv += typical * vol
        cum_vol += vol
        vwap[i] = cum_pv / cum_vol if cum_vol > 0.0 else close_arr[i]
    return vwap


class Strategy(BaseStrategy):
    """European Open Momentum Continuation on NIFTY index options."""

    name = "european_open_momentum_continuation"
    underlying = "NIFTY"
    # VWAP requires data from 09:20 IST; entries restricted to EU open window [810, 855)
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 855     # 14:15 IST — EU open window closes; no entries after this
    max_trades_per_day = 5
    # 360 bars (30 min context window) + 24 bars (2 min burst) + buffer
    max_lookback = 400

    # European open window constants (IST minutes from midnight)
    _EU_OPEN_START = 810   # 13:30 IST
    _EU_OPEN_END = 855     # 14:15 IST

    def tunable_params(self) -> list[TunableParam]:
        return [
            # 30-min momentum threshold: ~0.3% pre-European-open drift confirms direction
            TunableParam("momentum_threshold", 0.003, 0.001, 0.008),
            # 2-min burst threshold: ~0.03% acceleration confirms active European flow
            TunableParam("burst_threshold", 0.0003, 0.0001, 0.002),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # --- Extract arrays (forward-fill NaN before converting) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        momentum_threshold = params.get("momentum_threshold", 0.003)
        burst_threshold = params.get("burst_threshold", 0.0003)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # --- Indicator 1: Session VWAP (cumulative, resets daily) ---
        vwap = _compute_session_vwap(open_, high, low, close, volume, day_id)

        # --- Indicator 2: 30-minute momentum (360 bars × 5s) ---
        # Context filter: same 30-minute window as original equity strategy.
        # Measures directional bias heading into European open; 360 = 30 min at 5s.
        mom_360 = np.zeros(n)
        for i in range(360, n):
            ref = close[i - 360]
            if ref > 0.0:
                mom_360[i] = (close[i] - ref) / ref

        # --- Indicator 3: 2-minute burst momentum (24 bars × 5s) ---
        # Trade trigger: detects the immediate acceleration at 13:30 IST.
        # Not in original (1-min bars couldn't isolate a 2-min sub-bar acceleration).
        mom_24 = np.zeros(n)
        for i in range(24, n):
            ref = close[i - 24]
            if ref > 0.0:
                mom_24[i] = (close[i] - ref) / ref

        # --- European open session filter: 13:30-14:15 IST ---
        in_eu_window = (time_min >= self._EU_OPEN_START) & (time_min < self._EU_OPEN_END)

        # --- Signal generation ---
        # buy_ce: bullish — above VWAP, 30-min uptrend, 2-min upside burst at EU open
        buy_ce = (
            in_eu_window
            & (close > vwap)
            & (mom_360 > momentum_threshold)
            & (mom_24 > burst_threshold)
        )

        # buy_pe: bearish — below VWAP, 30-min downtrend, 2-min downside burst at EU open
        buy_pe = (
            in_eu_window
            & (close < vwap)
            & (mom_360 < -momentum_threshold)
            & (mom_24 < -burst_threshold)
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
