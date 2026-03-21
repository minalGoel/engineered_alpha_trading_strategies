"""orb_closing_range_v1 — Closing Range Breakout on NIFTY.

Thesis: NIFTY's 14:30-15:00 IST window is when mutual funds deploy SIP cash and FIIs
flatten intraday delta before overnight risk. This compresses NIFTY into a 20-40 spot pt
closing range. A breakout above/below this range at 15:00+ signals that one side has
overwhelmed the closing equilibrium, with short-covering accelerating the move.

Entry: 15:00-15:20 IST only, aligned with session VWAP and day range position.
Stop: 5 option pts (~10 spot pts — breakout stalling this much = failed thesis).
Target: 8 option pts (~16 spot pts — ~50% of expected 20-35 pt institutional continuation).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam

# IST minute constants
_CRB_START = 870   # 14:30 IST
_CRB_END   = 900   # 15:00 IST
_ENTRY_START = 900  # 15:00 IST
_ENTRY_END   = 920  # 15:20 IST


def _compute_closing_range(
    high: np.ndarray,
    low: np.ndarray,
    time_min: np.ndarray,
    day_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CRB high/low arrays: max(high)/min(low) of the 14:30-15:00 window.

    Values within the window update incrementally.  After 15:00 they are
    forward-filled so that the 15:00-15:20 entry logic can read them.
    Before the first 14:30 bar of each day the values are 0.0 (invalid;
    guarded by crb_valid mask in compute()).
    """
    n = len(high)
    crb_high = np.zeros(n)
    crb_low  = np.zeros(n)

    cur_day   = -1
    day_crb_h = -1e18
    day_crb_l =  1e18
    had_crb   = False   # did we see any 14:30-15:00 bar today?

    for i in range(n):
        d = day_id[i]
        if d != cur_day:
            cur_day   = d
            day_crb_h = -1e18
            day_crb_l =  1e18
            had_crb   = False

        t = time_min[i]
        if _CRB_START <= t < _CRB_END:
            if high[i] > day_crb_h:
                day_crb_h = high[i]
            if low[i] < day_crb_l:
                day_crb_l = low[i]
            had_crb = True

        if had_crb:
            crb_high[i] = day_crb_h
            crb_low[i]  = day_crb_l
        # else stays 0.0 (invalid, caught by crb_valid mask)

    return crb_high, crb_low


def _compute_session_vwap(
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Cumulative session VWAP, reset at each new day_id."""
    n = len(close)
    vwap    = np.empty(n)
    cum_pv  = 0.0
    cum_vol = 0.0
    cur_day = -1

    for i in range(n):
        if day_id[i] != cur_day:
            cur_day  = day_id[i]
            cum_pv   = 0.0
            cum_vol  = 0.0
        v = volume[i]
        if v > 0:
            cum_pv  += close[i] * v
            cum_vol += v
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else close[i]

    return vwap


def _compute_day_range_pos(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Close position in the day's expanding range: 0 = at day low, 1 = at day high."""
    n       = len(close)
    pos     = np.full(n, 0.5)
    day_h   = -1e18
    day_l   =  1e18
    cur_day = -1

    for i in range(n):
        if day_id[i] != cur_day:
            cur_day = day_id[i]
            day_h   = -1e18
            day_l   =  1e18
        if high[i] > day_h:
            day_h = high[i]
        if low[i] < day_l:
            day_l = low[i]
        rng = day_h - day_l
        pos[i] = (close[i] - day_l) / rng if rng > 0.0 else 0.5

    return pos


class Strategy(BaseStrategy):
    name                  = "orb_closing_range_v1"
    underlying            = "NIFTY"
    session_start_minutes = 555   # 09:15 IST — need full day for CRB + VWAP
    session_end_minutes   = 925   # 15:25 IST — standard EOD flatten
    max_trades_per_day    = 2
    max_lookback          = 400   # warmup: covers pre-14:30 data needed to form CRB

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max",            22.0, 15.0, 30.0),
            TunableParam("mom_threshold",        0.0, -3.0,  3.0),
            TunableParam("day_range_threshold",  0.5,  0.3,  0.7),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill before numpy conversion) ──────────
        close    = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high     = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low      = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume   = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        # ── Parameters ──────────────────────────────────────────────────────────
        vix_max           = params.get("vix_max", 22.0)
        mom_threshold     = params.get("mom_threshold", 0.0)
        day_range_thr     = params.get("day_range_threshold", 0.5)

        # ── VIX (aligned to spot bars via asof join) ─────────────────────────────
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

        # ── Closing range (14:30-15:00 IST = 870-899 min) ──────────────────────
        crb_high, crb_low = _compute_closing_range(high, low, time_min, day_id)

        # ── Session VWAP (cumulative, reset each day) ───────────────────────────
        session_vwap = _compute_session_vwap(close, volume, day_id)

        # ── 2-minute momentum (24 bars × 5s = 120s) ────────────────────────────
        momentum = np.zeros(n)
        if n > 24:
            momentum[24:] = close[24:] - close[:-24]

        # ── Close position in expanding day range (0=low, 1=high) ──────────────
        day_range_pos = _compute_day_range_pos(high, low, close, day_id)

        # ── Masks ────────────────────────────────────────────────────────────────
        # Entry window: 15:00-15:20 IST only
        in_window = (time_min >= _ENTRY_START) & (time_min < _ENTRY_END)

        # CRB valid: the 14:30-15:00 window must have been observed
        crb_valid = (crb_high > 0.0) & (crb_low > 0.0) & (crb_high > crb_low)

        # VIX filter
        low_vix = vix_close < vix_max

        # ── Signals ──────────────────────────────────────────────────────────────
        buy_ce = (
            in_window
            & crb_valid
            & low_vix
            & (close > crb_high)
            & (momentum > mom_threshold)
            & (close > session_vwap)
            & (day_range_pos > day_range_thr)
        )

        buy_pe = (
            in_window
            & crb_valid
            & low_vix
            & (close < crb_low)
            & (momentum < -mom_threshold)
            & (close < session_vwap)
            & (day_range_pos < (1.0 - day_range_thr))
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 5.0),
            target_points=np.full(n, 8.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=120,           # 10 minutes max hold
            max_trades_per_day=self.max_trades_per_day,
        )
