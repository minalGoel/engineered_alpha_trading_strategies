"""changepoint_detection_v1 — CUSUM Regime-Shift Strategy for NIFTY 5-second options.

Mechanism:
    On NIFTY, large institutional TWAP/VWAP execution and macro news events generate
    persistent cumulative return deviations that build bar-by-bar before any lagging
    indicator confirms a trend. CUSUM control charts accumulate these micro-deviations
    against a 5-minute rolling baseline: when the positive CUSUM first crosses a
    volatility-scaled threshold, the index has entered a new buying regime. We enter
    within the first 15-30 seconds of a new regime — before momentum strategies trigger.

Converted from: trading_strategies/unique_strategies_all/Strategy_330.json
Original: CUSUM on 1-min equity bars, hold 15-60 min.
Adapted: CUSUM on 5s NIFTY index, hold 15-120s. Rolling window compressed from
         30-min to 5-min to match the shorter hold horizon.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "changepoint_detection_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — need 60-bar warmup after open
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 120            # 10-min warmup (2x the 60-bar rolling window)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # CUSUM allowable slack: k = k_mult * rolling_std
            TunableParam("cusum_k_multiplier", 0.5, 0.2, 1.0),
            # CUSUM detection threshold: h = h_mult * rolling_std
            TunableParam("cusum_h_multiplier", 3.0, 1.5, 6.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
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

        # ── Extract arrays (forward-fill NaN in Polars before numpy) ──────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        k_mult = float(params.get("cusum_k_multiplier", 0.5))
        h_mult = float(params.get("cusum_h_multiplier", 3.0))
        stop_pts = float(params.get("stop_pts", 4.0))
        target_pts = float(params.get("target_pts", 6.0))

        ROLL = 60  # 5-minute rolling window for baseline statistics

        # ── 1-bar returns ─────────────────────────────────────────────────────
        ret = np.zeros(n, dtype=np.float64)
        denom = np.where(close[:-1] != 0.0, close[:-1], 1.0)
        ret[1:] = (close[1:] - close[:-1]) / denom

        # ── Rolling mean and std of returns (5-minute window) ─────────────────
        roll_mean = np.zeros(n, dtype=np.float64)
        roll_std = np.full(n, 1e-7, dtype=np.float64)
        for i in range(ROLL, n):
            window = ret[i - ROLL:i]
            roll_mean[i] = np.mean(window)
            s = np.std(window)
            roll_std[i] = s if s > 1e-7 else 1e-7

        # ── CUSUM accumulation with session-level reset ────────────────────────
        # cusum_pos: accumulates positive excess returns above allowable slack k
        # cusum_neg: accumulates negative excess returns below allowable slack k
        cusum_pos = np.zeros(n, dtype=np.float64)
        cusum_neg = np.zeros(n, dtype=np.float64)

        for i in range(1, n):
            if day_id[i] != day_id[i - 1]:
                # New trading day — reset CUSUM to avoid prior-session contamination
                cusum_pos[i] = 0.0
                cusum_neg[i] = 0.0
            else:
                k = k_mult * roll_std[i]
                excess = ret[i] - roll_mean[i]
                cusum_pos[i] = max(0.0, cusum_pos[i - 1] + excess - k)
                cusum_neg[i] = min(0.0, cusum_neg[i - 1] + excess + k)

        # ── Detection threshold (volatility-scaled per bar) ────────────────────
        threshold = h_mult * roll_std  # shape (n,)

        # ── "Just crossed" detector — fire only on the crossing bar ───────────
        # buy_ce: cusum_pos exceeded threshold this bar but NOT last bar
        crossed_pos = np.zeros(n, dtype=bool)
        crossed_neg = np.zeros(n, dtype=bool)
        for i in range(1, n):
            crossed_pos[i] = (
                cusum_pos[i] > threshold[i]
                and cusum_pos[i - 1] <= threshold[i - 1]
            )
            crossed_neg[i] = (
                abs(cusum_neg[i]) > threshold[i]
                and abs(cusum_neg[i - 1]) <= threshold[i - 1]
            )

        # ── Session and warmup filters ─────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed_up = np.arange(n) >= ROLL

        buy_ce = in_session & warmed_up & crossed_pos
        buy_pe = in_session & warmed_up & crossed_neg

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts, dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
