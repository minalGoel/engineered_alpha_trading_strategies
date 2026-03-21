"""gap_fill_stall_001 — NIFTY Opening Range Fakeout / Gap Fill Stall

Thesis: On gap-down days (>1.2%), when NIFTY dips below the opening range low
but closes back above it in the same 5-second bar, stop cascades are exhausted
and VWAP-benchmarked institutional buyers drive a reversal toward session VWAP.
Mirror logic for gap-up + OR high fakeout.

Session: 09:30-10:45 IST (OR must be fully formed before trading begins).
Hold: 15-90 seconds (time_stop_bars=18).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Cumulative session VWAP — resets at day boundary."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -1
    for i in range(n):
        d = int(day_id[i])
        if d != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = d
        v = float(volume[i])
        cum_pv += close[i] * v
        cum_v += v
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


def _compute_opening_range(
    high: np.ndarray,
    low: np.ndarray,
    time_min: np.ndarray,
    day_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """OR high/low from 09:15-09:30 (time_minutes 555-569), carried forward per day."""
    n = len(high)
    or_high_by_day: dict[int, float] = {}
    or_low_by_day: dict[int, float] = {}

    for i in range(n):
        d = int(day_id[i])
        t = int(time_min[i])
        if 555 <= t < 570:  # 09:15 to 09:30 window
            if d not in or_high_by_day:
                or_high_by_day[d] = high[i]
                or_low_by_day[d] = low[i]
            else:
                or_high_by_day[d] = max(or_high_by_day[d], high[i])
                or_low_by_day[d] = min(or_low_by_day[d], low[i])

    or_high = np.empty(n)
    or_low = np.empty(n)
    for i in range(n):
        d = int(day_id[i])
        # Fallback to current close if OR not yet formed (shouldn't trade during this period)
        or_high[i] = or_high_by_day.get(d, high[i])
        or_low[i] = or_low_by_day.get(d, low[i])

    return or_high, or_low


def _compute_gap_pct(
    open_: np.ndarray,
    close: np.ndarray,
    day_id: np.ndarray,
    time_min: np.ndarray,
) -> np.ndarray:
    """Gap pct = day_open / prev_day_close - 1, carried through each day."""
    n = len(open_)
    # Find first open per day and last close per day
    day_open: dict[int, float] = {}
    day_last_close: dict[int, float] = {}

    for i in range(n):
        d = int(day_id[i])
        if d not in day_open:
            day_open[d] = open_[i]
        day_last_close[d] = close[i]

    all_days = sorted(day_open.keys())
    day_gap: dict[int, float] = {}
    for j, d in enumerate(all_days):
        if j > 0:
            prev_d = all_days[j - 1]
            prev_close = day_last_close.get(prev_d, 0.0)
            if prev_close > 0.0:
                day_gap[d] = day_open[d] / prev_close - 1.0
            else:
                day_gap[d] = 0.0
        else:
            day_gap[d] = 0.0  # first day: no gap info

    gap_pct = np.zeros(n)
    for i in range(n):
        d = int(day_id[i])
        gap_pct[i] = day_gap.get(d, 0.0)

    return gap_pct


def _rolling_rel_vol(volume: np.ndarray, window: int = 120) -> np.ndarray:
    """volume / SMA(volume, window); returns 1.0 for bars before warmup."""
    n = len(volume)
    rel_vol = np.ones(n)
    for i in range(window, n):
        mean_v = np.mean(volume[i - window:i])
        rel_vol[i] = volume[i] / mean_v if mean_v > 0.0 else 1.0
    return rel_vol


class Strategy(BaseStrategy):
    name = "gap_fill_stall_001"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — after OR is fully formed
    session_end_minutes = 645     # 10:45 IST
    max_trades_per_day = 4
    max_lookback = 180            # 15 min warmup to form the opening range

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.012, 0.008, 0.020),
            TunableParam("rel_vol_threshold", 1.2, 1.0, 2.0),
            TunableParam("trend_filter_threshold", 0.004, 0.002, 0.008),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        gap_threshold = params.get("gap_threshold", 0.012)
        rel_vol_threshold = params.get("rel_vol_threshold", 1.2)
        trend_threshold = params.get("trend_filter_threshold", 0.004)

        # ── Extract and forward-fill OHLCV ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Session VWAP ──
        vwap = _compute_vwap(close, volume, day_id)

        # ── Opening range (09:15-09:30) ──
        or_high, or_low = _compute_opening_range(high, low, time_min, day_id)

        # ── Gap pct (day-level, carried forward) ──
        gap_pct = _compute_gap_pct(open_, close, day_id, time_min)

        # ── Relative volume vs 10-min rolling average ──
        rel_vol = _rolling_rel_vol(volume, window=120)

        # ── 5-min return (60 bars) for trend filter ──
        ret_60 = np.zeros(n)
        for i in range(60, n):
            if close[i - 60] > 0.0:
                ret_60[i] = close[i] / close[i - 60] - 1.0

        # ── Session and OR-formed filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Trend filter: NIFTY not in a strong directional 5-min move ──
        not_trending = np.abs(ret_60) <= trend_threshold

        # ── Volume participation filter ──
        vol_ok = rel_vol >= rel_vol_threshold

        # ── Bullish: gap-down fakeout below OR low, closes back above ──
        # Gap-down day, price wicks below OR low but closes above it (stall),
        # still below VWAP (room to fill), bullish bar close
        buy_ce = (
            in_session
            & (gap_pct <= -gap_threshold)     # gap-down day
            & (low < or_low)                   # fakeout — wick below OR low
            & (close >= or_low)                # stall — close recovers above OR low
            & (close < vwap)                   # upside gap-fill room
            & (close > open_)                  # bullish bar
            & vol_ok
            & not_trending
        )

        # ── Bearish: gap-up fakeout above OR high, closes back below ──
        buy_pe = (
            in_session
            & (gap_pct >= gap_threshold)       # gap-up day
            & (high > or_high)                 # fakeout — wick above OR high
            & (close <= or_high)               # stall — close rejects back below OR high
            & (close > vwap)                   # downside gap-fill room
            & (close < open_)                  # bearish bar
            & vol_ok
            & not_trending
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4),
            target_points=np.full(n, 8),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,
            max_trades_per_day=self.max_trades_per_day,
        )
