"""Mean Reversion Hull MA — 5-second NIFTY index options strategy.

When NIFTY price deviates significantly from its 3-minute Hull Moving Average
while the HMA slope is flat (ranging regime), VWAP-benchmarked algos and
market makers push price back toward the HMA within 30-90 seconds.

Converted from: trading_strategies/unique_strategies_all/Strategy_353.json
Original: HMA(50) mean reversion on 1-min F&O stocks, 10-40 min hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _wma(arr: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average — weight = position index (1..period)."""
    n = len(arr)
    result = np.full(n, np.nan)
    weights = np.arange(1, period + 1, dtype=np.float64)
    weight_sum = weights.sum()
    for i in range(period - 1, n):
        result[i] = np.dot(arr[i - period + 1 : i + 1], weights) / weight_sum
    return result


def _hma(arr: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average = WMA(2*WMA(n/2) - WMA(n), floor(sqrt(n)))."""
    half = max(period // 2, 1)
    sqrtn = max(int(np.sqrt(period)), 1)
    wma_half = _wma(arr, half)
    wma_full = _wma(arr, period)

    # 2*WMA(n/2) - WMA(n) — NaN where either component is NaN
    raw = 2.0 * wma_half - wma_full

    # Forward-fill NaN in raw before outer WMA
    for i in range(1, len(raw)):
        if np.isnan(raw[i]):
            raw[i] = raw[i - 1] if not np.isnan(raw[i - 1]) else 0.0

    return _wma(raw, sqrtn)


class Strategy(BaseStrategy):
    name = "mean_reversion_hull_ma"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min opening noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup (covers HMA(36) + 12-bar slope)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Fractional deviation from HMA required to trigger entry
            # 0.0012 ≈ 0.12% ≈ 26 pts on NIFTY at 22000
            TunableParam("dev_threshold", 0.0012, 0.0006, 0.003),
            # HMA slope (% per 60s) below which market is considered flat/ranging
            TunableParam("slope_threshold", 0.0003, 0.0001, 0.001),
            # Stop in option premium points
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            # Target in option premium points
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Raw data ───────────────────────────────────────────────────────────
        close = (
            spot_df["close"]
            .fill_null(strategy="forward")
            .to_numpy()
            .astype(np.float64)
        )
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        dev_threshold = float(params.get("dev_threshold", 0.0012))
        slope_threshold = float(params.get("slope_threshold", 0.0003))
        stop_pts = float(params.get("stop_pts", 3.0))
        target_pts = float(params.get("target_pts", 6.0))

        # ── HMA(36) — 3-minute Hull Moving Average ────────────────────────────
        # 36 bars × 5s = 3 minutes. Compressed from original 50-min HMA:
        # we hold 15-90s so the equilibrium anchor must be at the same timescale.
        hma = _hma(close, 36)

        # Forward-fill any remaining NaN in HMA
        for i in range(1, n):
            if np.isnan(hma[i]):
                hma[i] = hma[i - 1] if not np.isnan(hma[i - 1]) else close[i]
        # Fill leading NaN with close
        if np.isnan(hma[0]):
            hma[0] = close[0]
        for i in range(1, n):
            if np.isnan(hma[i]):
                hma[i] = hma[i - 1]

        # ── Deviation from HMA ────────────────────────────────────────────────
        # Signed fractional distance: positive = price above HMA (overbought)
        dist = np.zeros(n, dtype=np.float64)
        nonzero = close > 0
        dist[nonzero] = (close[nonzero] - hma[nonzero]) / close[nonzero]

        # ── HMA slope over 12 bars (60 seconds) ──────────────────────────────
        # Flat slope identifies ranging regime. If HMA is trending, deviation
        # is justified by order flow and reversion thesis breaks.
        hma_slope = np.zeros(n, dtype=np.float64)
        for i in range(12, n):
            if hma[i - 12] > 0:
                hma_slope[i] = (hma[i] - hma[i - 12]) / hma[i - 12]

        # ── Signal masks ──────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # HMA is flat — ranging regime filter
        hma_flat = np.abs(hma_slope) < slope_threshold

        # Price below flat HMA → NIFTY oversold relative to recent equilibrium
        # → expect reversion up → buy CE
        buy_ce = in_session & hma_flat & (dist < -dev_threshold)

        # Price above flat HMA → NIFTY overbought relative to recent equilibrium
        # → expect reversion down → buy PE
        buy_pe = in_session & hma_flat & (dist > dev_threshold)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts, dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
