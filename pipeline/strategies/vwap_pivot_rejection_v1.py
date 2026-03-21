"""VWAP + Pivot Rejection — vwap_pivot_rejection_v1

Mechanism: On NIFTY, when the session VWAP drifts within 25 spot points of a daily
pivot level (S1, PP, or R1 from previous day's H/L/C), two independent algorithmic
order clusters concentrate at the same price region — VWAP-benchmarked futures algos
defending their average fill cost and pivot-level conditional order queues. A 30-second
test into this double-friction zone followed by a dominant directional rejection wick
(55%+ of 30s range as the rejection wick, net close in rejection direction) signals
institutional absorption held and predicts 15-25 spot point follow-through in 45-90s.

Converted from: trading_strategies/unique_strategies_all/Strategy_14.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_session_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Session VWAP, reset each day."""
    n = len(close)
    vwap = np.zeros(n)
    typical_price = (high + low + close) / 3.0
    cum_tp_vol = 0.0
    cum_vol = 0.0
    prev_day = -1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        v = volume[i] if volume[i] > 0 else 0
        cum_tp_vol += typical_price[i] * v
        cum_vol += v
        vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0 else close[i]
    return vwap


def _compute_pivot_levels(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    day_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Previous day S1, PP, R1 for each bar. First session uses NaN → forward-filled later."""
    n = len(close)
    pivot_pp = np.full(n, np.nan)
    pivot_r1 = np.full(n, np.nan)
    pivot_s1 = np.full(n, np.nan)

    days = np.unique(day_id)
    daily_high: dict[int, float] = {}
    daily_low: dict[int, float] = {}
    daily_close: dict[int, float] = {}

    for d in days:
        mask = day_id == d
        daily_high[d] = float(high[mask].max())
        daily_low[d] = float(low[mask].min())
        daily_close[d] = float(close[mask][-1])

    for idx, d in enumerate(days):
        if idx == 0:
            continue  # no previous day
        prev_d = days[idx - 1]
        ph = daily_high[prev_d]
        pl_ = daily_low[prev_d]
        pc = daily_close[prev_d]
        pp = (ph + pl_ + pc) / 3.0
        r1 = 2.0 * pp - pl_
        s1 = 2.0 * pp - ph
        mask = day_id == d
        pivot_pp[mask] = pp
        pivot_r1[mask] = r1
        pivot_s1[mask] = s1

    # Forward-fill NaN (first day stays NaN → will be neutralised below)
    for i in range(1, n):
        if np.isnan(pivot_pp[i]):
            pivot_pp[i] = pivot_pp[i - 1]
            pivot_r1[i] = pivot_r1[i - 1]
            pivot_s1[i] = pivot_s1[i - 1]

    # Replace remaining NaN (first session) with neutral: far from any close so
    # zone_active will be False and strategy won't fire on first day.
    big = 9999.0
    pivot_pp = np.where(np.isnan(pivot_pp), close + big, pivot_pp)
    pivot_r1 = np.where(np.isnan(pivot_r1), close + big, pivot_r1)
    pivot_s1 = np.where(np.isnan(pivot_s1), close + big, pivot_s1)

    return pivot_s1, pivot_pp, pivot_r1


class Strategy(BaseStrategy):
    name = "vwap_pivot_rejection_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min ORB noise / VWAP warmup
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 240            # 20 min warmup for VWAP stability
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            # How close VWAP must be to nearest pivot for zone to activate (spot pts)
            TunableParam("confluence_pts", 25.0, 10.0, 50.0),
            # How close current price must be to VWAP (spot pts)
            TunableParam("vwap_proximity_pts", 15.0, 8.0, 30.0),
            # Minimum fraction of 30s range that must be the rejection wick
            TunableParam("wick_threshold", 0.55, 0.40, 0.75),
            # VIX upper bound
            TunableParam("vix_threshold", 20.0, 14.0, 28.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill in Polars before numpy extraction
        spot_filled = spot_df.with_columns([
            pl.col("close").forward_fill(),
            pl.col("high").forward_fill(),
            pl.col("low").forward_fill(),
            pl.col("open").forward_fill(),
            pl.col("volume").fill_null(0),
        ])

        close = spot_filled["close"].to_numpy()
        high = spot_filled["high"].to_numpy()
        low = spot_filled["low"].to_numpy()
        open_ = spot_filled["open"].to_numpy()
        volume = spot_filled["volume"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # --- Parameters ---
        confluence_pts = params.get("confluence_pts", 25.0)
        vwap_proximity_pts = params.get("vwap_proximity_pts", 15.0)
        wick_threshold = params.get("wick_threshold", 0.55)
        vix_threshold = params.get("vix_threshold", 20.0)

        # --- VIX filter ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- Session VWAP ---
        vwap = _compute_session_vwap(high, low, close, volume, day_id)

        # --- Previous day pivot levels ---
        pivot_s1, pivot_pp, pivot_r1 = _compute_pivot_levels(close, high, low, day_id)

        # --- Zone activation: VWAP within confluence_pts of any pivot level ---
        min_pivot_dist = np.minimum(
            np.minimum(np.abs(vwap - pivot_s1), np.abs(vwap - pivot_pp)),
            np.abs(vwap - pivot_r1),
        )
        zone_active = min_pivot_dist <= confluence_pts

        # --- Price near VWAP ---
        near_vwap = np.abs(close - vwap) <= vwap_proximity_pts

        # --- 6-bar (30s) rejection detection ---
        # Bullish rejection: lower wick dominates (price tested below and closed up)
        # Bearish rejection: upper wick dominates (price tested above and closed down)
        window = 6
        bullish_rejection = np.zeros(n, dtype=bool)
        bearish_rejection = np.zeros(n, dtype=bool)

        for i in range(window, n):
            w_high = np.max(high[i - window: i + 1])
            w_low = np.min(low[i - window: i + 1])
            w_range = w_high - w_low
            if w_range < 1.0:
                continue

            w_close = close[i]
            w_open = open_[i - window]

            lower_wick_frac = (w_close - w_low) / w_range    # bullish if dominant
            upper_wick_frac = (w_high - w_close) / w_range   # bearish if dominant

            # Bullish: lower wick ≥ threshold AND net positive move over window
            if lower_wick_frac >= wick_threshold and w_close > w_open:
                bullish_rejection[i] = True

            # Bearish: upper wick ≥ threshold AND net negative move over window
            if upper_wick_frac >= wick_threshold and w_close < w_open:
                bearish_rejection[i] = True

        # --- Combine signals ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < vix_threshold
        in_zone = zone_active & near_vwap

        buy_ce = in_session & vix_ok & in_zone & bullish_rejection
        buy_pe = in_session & vix_ok & in_zone & bearish_rejection

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 4 pts ≈ 8 NIFTY spot pts — zone breach invalidates thesis
            stop_points=np.full(n, 3),
            # Target: 7 pts ≈ 14 NIFTY spot pts — lower bound of 15-25 pt post-rejection move
            target_points=np.full(n, 6),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
