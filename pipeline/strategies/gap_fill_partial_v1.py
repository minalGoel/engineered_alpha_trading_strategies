"""Gap Fill Partial v1 — NIFTY 5-second index options strategy.

On NIFTY gap-up/gap-down days (0.4–1.2%), short-term gap traders begin
profit-taking as price moves toward the half-gap level (midpoint between
today's open and yesterday's close). At 5-second resolution the first
EMA(12)/EMA(36) cross in the fill direction — before the half-gap is reached
— marks the onset of this profit-taking flow. Hold 15–90 seconds to capture
the first 8–10 NIFTY spot point leg toward the midpoint.

DIFFERENTIATED from gap_fill_full_v1: entry trigger is EMA cross (not VWAP
cross), time window is tighter (09:20–09:50, not 09:20–10:30), and target is
smaller (5 pts vs 7 pts). We enter earlier in the fill cycle, before
institutional VWAP flow has even registered.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "gap_fill_partial_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST (used by pipeline for EOD flatten)
    max_trades_per_day = 4
    max_lookback = 60             # 5-min warmup for EMA_36 to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min_pct", 0.004, 0.002, 0.008),    # minimum gap magnitude (0.4%)
            TunableParam("gap_max_pct", 0.012, 0.008, 0.020),    # maximum gap magnitude (1.2%)
            TunableParam("vix_min", 11.0, 8.0, 15.0),            # VIX floor
            TunableParam("vix_max", 20.0, 16.0, 26.0),           # VIX ceiling
            TunableParam("entry_window_end", 590.0, 570.0, 630.0),  # 09:50 = 590 mins
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Pull params ──────────────────────────────────────────────────────
        gap_min = params.get("gap_min_pct", 0.004)
        gap_max = params.get("gap_max_pct", 0.012)
        vix_min = params.get("vix_min", 11.0)
        vix_max = params.get("vix_max", 20.0)
        entry_end = int(params.get("entry_window_end", 590))

        # ── Base arrays ───────────────────────────────────────────────────────
        close_np = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        open_np = spot_df["open"].fill_null(strategy="forward").to_numpy().astype(float)
        day_id_np = spot_df["day_id"].to_numpy().astype(int)
        time_min = spot_df["time_minutes"].to_numpy().astype(int)

        # ── VIX ──────────────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy().astype(float)

        # ── Per-day gap_pct and half_gap ──────────────────────────────────────
        gap_pct = np.zeros(n)
        half_gap = np.zeros(n)

        sorted_days = sorted(set(day_id_np.tolist()))

        # Build day_id → last close map (used to get prev day's close)
        day_close_map: dict[int, float] = {}
        for d in sorted_days:
            idx = np.where(day_id_np == d)[0]
            if len(idx) > 0:
                day_close_map[d] = close_np[idx[-1]]

        # Day boundary mask (first bar of each day)
        first_bar_mask = np.zeros(n, dtype=bool)

        for enum_i, d in enumerate(sorted_days):
            idx = np.where(day_id_np == d)[0]
            if len(idx) == 0:
                continue

            first_bar_mask[idx[0]] = True

            if enum_i > 0:
                prev_d = sorted_days[enum_i - 1]
                prev_close = day_close_map.get(prev_d, np.nan)
                if prev_close > 0 and not np.isnan(prev_close):
                    day_open = open_np[idx[0]]
                    gp = (day_open - prev_close) / prev_close
                    gap_pct[idx] = gp
                    # Half-gap = midpoint between open and prev close
                    hg = (day_open + prev_close) / 2.0
                    half_gap[idx] = hg
            # Days with no previous session data keep gap_pct=0 → no signal

        # ── EMA(12) and EMA(36) ───────────────────────────────────────────────
        # EMA(12) ≈ 1-min fast trigger; EMA(36) ≈ 3-min slow confirmation
        ema12 = np.zeros(n)
        ema36 = np.zeros(n)
        k12 = 2.0 / (12 + 1)
        k36 = 2.0 / (36 + 1)

        ema12[0] = close_np[0]
        ema36[0] = close_np[0]
        for i in range(1, n):
            ema12[i] = close_np[i] * k12 + ema12[i - 1] * (1.0 - k12)
            ema36[i] = close_np[i] * k36 + ema36[i - 1] * (1.0 - k36)

        # ── EMA cross detection (ignore day-boundary bars) ─────────────────────
        ema12_prev = np.roll(ema12, 1)
        ema36_prev = np.roll(ema36, 1)

        # Bullish cross: EMA12 moves from ≤ EMA36 to > EMA36 (momentum turns up)
        bullish_cross = (
            (ema12_prev <= ema36_prev) &
            (ema12 > ema36) &
            ~first_bar_mask
        )
        # Bearish cross: EMA12 moves from ≥ EMA36 to < EMA36 (momentum turns down)
        bearish_cross = (
            (ema12_prev >= ema36_prev) &
            (ema12 < ema36) &
            ~first_bar_mask
        )

        # ── Gap direction flags ───────────────────────────────────────────────
        gap_down = (gap_pct <= -gap_min) & (gap_pct >= -gap_max)
        gap_up   = (gap_pct >= gap_min)  & (gap_pct <= gap_max)
        has_gap  = half_gap > 0.0

        # Price still has room to fill (price has NOT passed the half-gap yet)
        # gap-up fill: price must still be above half_gap (room to fall toward it)
        above_half = close_np > half_gap
        # gap-down fill: price must still be below half_gap (room to rise toward it)
        below_half = close_np < half_gap

        # ── Session and VIX filters ──────────────────────────────────────────
        # Narrow entry window: 09:20–09:50 (partial fills complete in opening burst)
        in_entry_window = (time_min >= self.session_start_minutes) & (time_min <= entry_end)
        vix_ok = (vix_close >= vix_min) & (vix_close < vix_max)

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: gap-down day, EMA bullish cross, price below half_gap (room to rise)
        buy_ce = (
            gap_down &
            bullish_cross &
            below_half &
            has_gap &
            in_entry_window &
            vix_ok
        )

        # buy_pe: gap-up day, EMA bearish cross, price above half_gap (room to fall)
        buy_pe = (
            gap_up &
            bearish_cross &
            above_half &
            has_gap &
            in_entry_window &
            vix_ok
        )

        # gap_down and gap_up are mutually exclusive; no explicit exclusion needed
        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 3.0),    # 3 opt pts = ~6 NIFTY spot pts at delta 0.5
            target_points=np.full(n, 5.0),  # 5 opt pts = ~10 NIFTY spot pts (first leg)
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
