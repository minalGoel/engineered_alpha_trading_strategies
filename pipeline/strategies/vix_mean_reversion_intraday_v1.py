"""
vix_mean_reversion_intraday_v1 — India VIX intraday mean reversion on NIFTY options.

Thesis: When India VIX deviates > 3% from its session-open value, options market makers'
cumulative delta-hedge imbalance builds up. The moment VIX begins reverting (2 consecutive
declining 5s bars), those MMs start unwinding their NIFTY futures hedges, generating a
15-30 second bounce. Uses day-open VIX anchor (not rolling window) — differentiates from
vix_mean_reversion_v1 which trades the fast 10-min spike.

Session: 10:00–14:30 IST (let VIX stabilize 45 min post-open; no late-day entries)
Hold: 15–120 seconds (time_stop_bars=24)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average via direct recurrence."""
    alpha = 2.0 / (period + 1)
    result = np.empty(len(arr), dtype=np.float64)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
    return result


class Strategy(BaseStrategy):
    name = "vix_mean_reversion_intraday_v1"
    underlying = "NIFTY"
    # Entries start at 10:00 IST (600 min) — 45 min after open, matching original
    session_start_minutes = 600   # 10:00 IST
    session_end_minutes = 870     # 14:30 IST
    max_trades_per_day = 4
    # EMA(360) needs 360 bars warmup = 30 min; add buffer for day-open anchor
    max_lookback = 420            # 35 min warmup

    def tunable_params(self) -> list[TunableParam]:
        return [
            # VIX % deviation from day-open required to arm the signal
            TunableParam("vix_spike_threshold", 3.0, 1.5, 5.0),
            # NIFTY % move from day-open required to confirm spot direction
            TunableParam("nifty_move_threshold", 0.3, 0.1, 0.6),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        vix_spike_threshold = params.get("vix_spike_threshold", 3.0)
        nifty_move_threshold = params.get("nifty_move_threshold", 0.3)

        # ── Spot data ────────────────────────────────────────────────────────────
        spot_close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── VIX aligned to spot bars via backward asof join ───────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── VIX day-open anchor ───────────────────────────────────────────────
        # For each bar, the day-open VIX = VIX at the very first bar of that day.
        # Institutional hedgers benchmark to morning levels; this anchor captures
        # cumulative intraday fear, not just recent oscillations.
        vix_day_open = np.full(n, 15.0)
        current_day = -1
        current_day_vix_open = 15.0
        for i in range(n):
            if day_id[i] != current_day:
                current_day = day_id[i]
                current_day_vix_open = vix_close[i]
            vix_day_open[i] = current_day_vix_open

        # ── VIX % change from day-open ────────────────────────────────────────
        # Primary trigger: cumulative intraday VIX deviation
        vix_change_pct = np.where(
            vix_day_open > 0,
            (vix_close - vix_day_open) / vix_day_open * 100.0,
            0.0,
        )

        # ── VIX EMA(360) — 30-min intraday trend line ────────────────────────
        # Context filter: is VIX above/below its session trend?
        # 360 bars × 5s = 30 min = same window as original's EMA(30) on 1-min bars.
        vix_ema = _ema(vix_close, 360)

        # ── 2-bar VIX reversal signals ────────────────────────────────────────
        # Declining: 2 consecutive 5s VIX drops = 10 seconds of confirmed deceleration
        vix_declining_2 = np.zeros(n, dtype=bool)
        vix_rising_2 = np.zeros(n, dtype=bool)
        for i in range(2, n):
            vix_declining_2[i] = (
                vix_close[i] < vix_close[i - 1]
                and vix_close[i - 1] < vix_close[i - 2]
            )
            vix_rising_2[i] = (
                vix_close[i] > vix_close[i - 1]
                and vix_close[i - 1] > vix_close[i - 2]
            )

        # ── NIFTY % return from day-open ──────────────────────────────────────
        # Coherence filter: spot index must confirm the VIX-implied direction
        spot_day_open = np.full(n, 0.0)
        current_day = -1
        current_day_spot_open = 0.0
        for i in range(n):
            if day_id[i] != current_day:
                current_day = day_id[i]
                current_day_spot_open = spot_close[i] if spot_close[i] > 0 else 1.0
            spot_day_open[i] = current_day_spot_open

        nifty_open_return_pct = np.where(
            spot_day_open > 0,
            (spot_close - spot_day_open) / spot_day_open * 100.0,
            0.0,
        )

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        # Buy CE: VIX spiked from open, above EMA (room to revert), NIFTY is down,
        #         and VIX has started declining (MM delta-hedge unwind beginning)
        buy_ce = (
            in_session
            & (vix_change_pct > vix_spike_threshold)
            & (vix_close > vix_ema)
            & (nifty_open_return_pct < -nifty_move_threshold)
            & vix_declining_2
        )

        # Buy PE: VIX crushed from open (complacency), below EMA, NIFTY is up,
        #         and VIX has started rising (re-hedging demand returning)
        buy_pe = (
            in_session
            & (vix_change_pct < -vix_spike_threshold)
            & (vix_close < vix_ema)
            & (nifty_open_return_pct > nifty_move_threshold)
            & vix_rising_2
        )

        # No simultaneous signals — buy_ce and buy_pe are mutually exclusive by
        # construction (vix_change_pct cannot be both > threshold and < -threshold)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 4 option pts = ~8 NIFTY spot pts at delta ~0.5
            # If NIFTY moves 8 more pts against us after VIX began reverting,
            # the thesis is invalidated for this bar.
            stop_points=np.full(n, 4.0),
            # Target: 7 option pts = ~14 NIFTY spot pts
            # Captures ~50% of the typical 20-35 spot pt recovery following
            # a 3%-from-open VIX spike reversion (1:1.75 R:R).
            target_points=np.full(n, 7.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds maximum hold
            max_trades_per_day=self.max_trades_per_day,
        )
