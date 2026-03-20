"""vwap_gap_fill_v1 — NIFTY VWAP Gap Fill (Previous Day VWAP Anchor)

Thesis: When NIFTY opens 0.25-1.5% away from the previous session's VWAP,
VWAP-benchmarked institutional algorithms (40-60% of NSE derivatives volume)
are systematically mis-positioned vs their benchmark and will rebalance
throughout the morning, creating persistent directional legs of 12-20 NIFTY
pts per 60-90 seconds toward prev_VWAP.

Entry: Momentum in gap-fill direction — close through 60s EMA with positive
1-min return, while fill is <40% complete. Buy CE for gap-down fills, PE for
gap-up fills.

Session: 09:20-10:30 IST only (gap fill is strictly a morning phenomenon).
Hold: 30-90 seconds (time_stop_bars=18).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average with alpha = 2/(period+1). No reset at day boundaries."""
    alpha = 2.0 / (period + 1)
    result = np.empty(len(arr), dtype=np.float64)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
    return result


def _compute_prev_day_vwap(
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """For each bar, return the previous trading day's full-session VWAP.

    Previous day VWAP = sum(close * vol) / sum(vol) over ALL bars in prior day.
    First day in dataset returns 0.0 (no prior day available — no trades will fire).
    """
    n = len(close)
    unique_days = sorted(set(int(d) for d in day_id))

    # Compute full-session VWAP per day
    day_vwap: dict[int, float] = {}
    for d in unique_days:
        mask = day_id == d
        v = volume[mask].astype(np.float64)
        c = close[mask]
        total_vol = float(np.sum(v))
        if total_vol > 0.0:
            day_vwap[d] = float(np.sum(c * v) / total_vol)
        else:
            day_vwap[d] = float(np.mean(c))

    # Map each day -> previous day's VWAP
    day_prev_vwap: dict[int, float] = {}
    for idx, d in enumerate(unique_days):
        if idx == 0:
            day_prev_vwap[d] = 0.0  # no prior day
        else:
            day_prev_vwap[d] = day_vwap[unique_days[idx - 1]]

    return np.array([day_prev_vwap[int(d)] for d in day_id], dtype=np.float64)


def _compute_day_first_open(
    open_: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Return the first bar's open price for each bar's trading day."""
    day_first: dict[int, float] = {}
    n = len(open_)
    for i in range(n):
        d = int(day_id[i])
        if d not in day_first:
            day_first[d] = float(open_[i])
    return np.array([day_first[int(d)] for d in day_id], dtype=np.float64)


class Strategy(BaseStrategy):
    name = "vwap_gap_fill_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip pre-open noise
    session_end_minutes = 630     # 10:30 IST — gap fills are a morning phenomenon
    max_trades_per_day = 4
    max_lookback = 120            # 10 min warmup (120 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold_pct", 0.25, 0.15, 0.60),
            TunableParam("max_gap_pct", 1.5, 0.8, 2.5),
            TunableParam("gap_fill_max_frac", 0.40, 0.20, 0.65),
            TunableParam("vix_max", 20.0, 14.0, 28.0),
            TunableParam("stop_pts", 4.0, 2.5, 7.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        gap_threshold = params.get("gap_threshold_pct", 0.25) / 100.0
        max_gap = params.get("max_gap_pct", 1.5) / 100.0
        gap_fill_max = params.get("gap_fill_max_frac", 0.40)
        vix_max = params.get("vix_max", 20.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Extract and forward-fill OHLCV ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── VIX: join asof to spot timeline ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Previous day's VWAP (institutional benchmark anchor) ──
        prev_vwap = _compute_prev_day_vwap(close, volume, day_id)

        # ── Today's first-bar open (gap origin) ──
        day_open = _compute_day_first_open(open_, day_id)

        # ── Gap pct: (today_open - prev_vwap) / prev_vwap (day-level constant) ──
        # Positive = gap-up above prev_VWAP, Negative = gap-down below prev_VWAP
        gap_pct = np.where(
            prev_vwap > 0.0,
            (day_open - prev_vwap) / prev_vwap,
            0.0,
        )

        # ── Gap fill fraction: how much of the gap (open→prev_vwap) has been filled ──
        # 0.0 at open, 1.0 when close == prev_vwap, <0 if gap extends further
        gap_size = day_open - prev_vwap  # positive for gap-up, negative for gap-down
        gap_fill_frac = np.where(
            np.abs(gap_size) > 0.5,  # require at least 0.5 pt gap to avoid div-by-zero noise
            np.where(
                gap_size > 0.0,
                (day_open - close) / gap_size,        # gap-up fill: close moving down toward prev_vwap
                (close - day_open) / (-gap_size),     # gap-down fill: close moving up toward prev_vwap
            ),
            0.0,
        )

        # ── EMA(12) = 60-second fast trend for entry confirmation ──
        ema12 = _compute_ema(close, 12)

        # ── 12-bar (1-minute) return for gap-fill momentum confirmation ──
        ret12 = np.zeros(n, dtype=np.float64)
        denom = np.maximum(close[:n - 12], 1.0)
        ret12[12:] = (close[12:] - close[:n - 12]) / denom

        # ── Session and gap-fill window filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── VIX regime: exclude panic markets where gaps extend rather than fill ──
        low_vix = vix_close < vix_max

        # ── Gap size conditions (day-level constants) ──
        has_gap_up = (gap_pct > gap_threshold) & (gap_pct < max_gap)
        has_gap_down = (gap_pct < -gap_threshold) & (gap_pct > -max_gap)

        # ── Fill in progress: started (>= 0) but not yet 40% complete ──
        fill_in_progress = (gap_fill_frac >= 0.0) & (gap_fill_frac < gap_fill_max)

        # ── Prior day data available ──
        has_prev_day = prev_vwap > 0.0

        # ── Entry signals ──
        # Gap-down day: VWAP-benchmarked algos accumulate to lift NIFTY toward prev_VWAP → buy CE
        # Confirmation: close > ema12 (price above 60s EMA) AND positive 1-min return
        buy_ce = (
            in_session
            & low_vix
            & has_gap_down
            & fill_in_progress
            & has_prev_day
            & (close > ema12)
            & (ret12 > 0.0001)
        )

        # Gap-up day: VWAP-benchmarked algos distribute to bring NIFTY back toward prev_VWAP → buy PE
        # Confirmation: close < ema12 (price below 60s EMA) AND negative 1-min return
        buy_pe = (
            in_session
            & low_vix
            & has_gap_up
            & fill_in_progress
            & has_prev_day
            & (close < ema12)
            & (ret12 < -0.0001)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
