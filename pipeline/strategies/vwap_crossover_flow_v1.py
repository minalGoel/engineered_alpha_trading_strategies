"""
vwap_crossover_flow_v1 — VWAP Crossover Institutional Flow

Mechanism:
    On NIFTY, when the index spends 2.5+ minutes (30+ bars at 5s) consistently
    below session VWAP, VWAP-benchmarked institutional algorithms accumulate a
    growing execution deficit. A subsequent crossover back above VWAP with 3x+
    volume surge fingerprints multiple TWAP/VWAP algos crossing their internal
    trigger simultaneously. The cross bar closing in the top 25% of its range
    confirms clean absorption. Residual order queues sustain momentum 30-60s.

Adapted from: trading_strategies/unique_strategies_all/Strategy_51.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_crossover_flow_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip pre-open VWAP noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 360            # 30 min warmup: enough for vol_sma(240) + consec counter

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("min_bars_one_side", 30.0, 15.0, 60.0),
            TunableParam("vol_surge_threshold", 3.0, 2.0, 5.0),
            TunableParam("vix_max", 22.0, 16.0, 28.0),
            TunableParam("range_pos_threshold", 0.75, 0.60, 0.90),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill first) ──────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .fill_null(0)
            .cast(pl.Float64)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        min_bars = int(params.get("min_bars_one_side", 30.0))
        vol_surge_thresh = float(params.get("vol_surge_threshold", 3.0))
        vix_max = float(params.get("vix_max", 22.0))
        range_pos_thresh = float(params.get("range_pos_threshold", 0.75))

        # ── VIX ───────────────────────────────────────────────────────────────
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

        # ── Session VWAP (cumulative, reset per day) ──────────────────────────
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            d = int(day_id[i])
            if d != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = d
            tp = (high[i] + low[i] + close[i]) / 3.0
            cum_tp_vol += tp * volume[i]
            cum_vol += volume[i]
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0 else close[i]

        # ── Volume SMA(240) = 20-minute rolling baseline ───────────────────────
        # Same absolute time window as original (20 1-min bars = 20 min)
        vol_sma = np.zeros(n)
        for i in range(n):
            start = max(0, i - 240)
            window = volume[start:i + 1]
            m = np.mean(window) if len(window) > 0 else 1.0
            vol_sma[i] = m if m > 0 else 1.0

        vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0.0)

        # ── Consecutive bars below/above VWAP (state machine) ─────────────────
        # Counts run of consecutive bars on the SAME side of VWAP within a day.
        consec_below = np.zeros(n, dtype=np.int32)
        consec_above = np.zeros(n, dtype=np.int32)
        prev_day = -1
        cb = 0
        ca = 0
        for i in range(n):
            d = int(day_id[i])
            if d != prev_day:
                cb = 0
                ca = 0
                prev_day = d
            if close[i] < vwap[i]:
                cb += 1
                ca = 0
            elif close[i] > vwap[i]:
                ca += 1
                cb = 0
            # close == vwap: keep existing counts
            consec_below[i] = cb
            consec_above[i] = ca

        # ── VWAP crossover detection ───────────────────────────────────────────
        # At the crossover bar, consec counters have already reset.
        # We need the PREVIOUS bar's count to check the pre-cross persistence.
        cross_up = np.zeros(n, dtype=bool)
        cross_down = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if day_id[i] == day_id[i - 1]:
                cross_up[i] = (close[i] > vwap[i]) and (close[i - 1] < vwap[i - 1])
                cross_down[i] = (close[i] < vwap[i]) and (close[i - 1] > vwap[i - 1])

        # Prior bar consecutive counts (check persistence before the cross)
        prev_consec_below = np.zeros(n, dtype=np.int32)
        prev_consec_above = np.zeros(n, dtype=np.int32)
        prev_consec_below[1:] = consec_below[:-1]
        prev_consec_above[1:] = consec_above[:-1]

        # ── Bar range position (where close sits in bar's high-low range) ──────
        bar_range = high - low
        range_pos = np.where(bar_range > 0, (close - low) / bar_range, 0.5)

        # ── Session filter ─────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        vix_ok = vix_close < vix_max

        # ── Entry signals ──────────────────────────────────────────────────────
        # Buy CE: VWAP cross up, 2.5+ min below VWAP, 3x volume surge, strong close
        buy_ce = (
            in_session
            & vix_ok
            & cross_up
            & (prev_consec_below >= min_bars)
            & (vol_ratio >= vol_surge_thresh)
            & (range_pos >= range_pos_thresh)
        )

        # Buy PE: VWAP cross down, 2.5+ min above VWAP, 3x volume surge, weak close
        buy_pe = (
            in_session
            & vix_ok
            & cross_down
            & (prev_consec_above >= min_bars)
            & (vol_ratio >= vol_surge_thresh)
            & (range_pos <= (1.0 - range_pos_thresh))
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),
            target_points=np.full(n, 7.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,         # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
