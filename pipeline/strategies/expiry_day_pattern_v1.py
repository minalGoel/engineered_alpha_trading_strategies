"""expiry_day_pattern_v1 — Expiry day gamma pin and unwind strategy.

On NIFTY/BANKNIFTY expiry days, option market makers short gamma at near-ATM
strikes create a two-phase pattern:
  - Morning (09:20-13:30): magnetic pull toward the max-OI strike (pin).
    When spot deviates 25+ pts below max-OI, hedgers absorb selling and push
    price back. Mirror image above max-OI.
  - Afternoon (13:30-15:25): gamma decays to near-zero; market makers unwind
    delta hedges, collapsing the pin force. Directional push becomes
    self-reinforcing for 30-60 seconds.

Signals fire ONLY on expiry days (detected from option_df where expiry ==
session_date). Max-OI strike is derived from option_df open_interest sums.
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_expiry_and_max_oi(
    spot_df: pl.DataFrame,
    option_df: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_expiry, max_oi_strike) arrays aligned to spot_df.

    is_expiry[i] is True if bar i falls on an expiry day.
    max_oi_strike[i] is the strike with highest combined CE+PE OI for that day
    (np.nan if no expiry or no OI data).
    """
    n = len(spot_df)
    is_expiry = np.zeros(n, dtype=bool)
    max_oi_strike = np.full(n, np.nan, dtype=np.float64)

    if option_df is None or option_df.is_empty():
        return is_expiry, max_oi_strike

    # Check required columns exist
    required = {"session_date", "expiry", "strike", "option_type", "open_interest"}
    if not required.issubset(set(option_df.columns)):
        return is_expiry, max_oi_strike

    # Unique session dates in spot_df
    try:
        unique_dates = spot_df["session_date"].unique().to_list()
    except Exception:
        return is_expiry, max_oi_strike

    for sd in unique_dates:
        # Filter option_df for this session date
        day_opts = option_df.filter(pl.col("session_date") == sd)
        if day_opts.is_empty():
            continue

        # Check if today is an expiry day: any option with expiry == session_date
        expiry_opts = day_opts.filter(pl.col("expiry") == sd)
        if expiry_opts.is_empty():
            continue

        # Compute max-OI strike: highest combined CE+PE OI using last reading per
        # (strike, option_type) to avoid summing OI across time bars
        oi_by_type = (
            expiry_opts
            .group_by(["strike", "option_type"])
            .agg(pl.col("open_interest").last().alias("last_oi"))
        )
        oi_by_strike = (
            oi_by_type
            .group_by("strike")
            .agg(pl.col("last_oi").sum().alias("total_oi"))
            .sort("total_oi", descending=True)
        )
        if oi_by_strike.is_empty():
            continue

        best_strike = float(oi_by_strike["strike"][0])

        # Mark all bars for this day
        day_mask = (spot_df["session_date"] == sd).to_numpy()
        is_expiry[day_mask] = True
        max_oi_strike[day_mask] = best_strike

    return is_expiry, max_oi_strike


class Strategy(BaseStrategy):
    name = "expiry_day_pattern_v1"
    underlying = "BOTH"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_lookback = 24             # 2 min warmup (small — signals are day-level)
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum spot-point deviation from max-OI strike to trigger morning signal
            TunableParam("dist_threshold_pts", 30.0, 15.0, 60.0),
            TunableParam("stop_pts",   4.0,  2.0,  8.0),
            TunableParam("target_pts", 7.0,  4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        dist_threshold = params.get("dist_threshold_pts", 30.0)
        stop_pts       = params.get("stop_pts",   4.0)
        target_pts     = params.get("target_pts", 7.0)

        # ── Core spot arrays ──────────────────────────────────────────────────
        close    = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high_arr = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low_arr  = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Expiry detection & max-OI strike ─────────────────────────────────
        is_expiry, max_oi_strike = _compute_expiry_and_max_oi(spot_df, option_df)
        has_max_oi = ~np.isnan(max_oi_strike)

        # Distance from max-OI strike in spot points (positive = above pin)
        dist = np.where(has_max_oi, close - max_oi_strike, 0.0)

        # ── 1-minute momentum (12 bars at 5s) ────────────────────────────────
        mom_12 = np.zeros(n, dtype=np.float64)
        if n > 12:
            mom_12[12:] = close[12:] - close[:-12]

        # ── 1-minute rolling high / low (12 bars) — afternoon breakout level ─
        roll_high_12 = np.full(n, np.nan, dtype=np.float64)
        roll_low_12  = np.full(n, np.nan, dtype=np.float64)
        for i in range(12, n):
            roll_high_12[i] = np.max(high_arr[i - 12:i])
            roll_low_12[i]  = np.min(low_arr[i - 12:i])

        valid_roll = ~np.isnan(roll_high_12) & ~np.isnan(roll_low_12)

        # ── VIX filter: 12-25 for meaningful gamma effects ────────────────────
        vix_arr = np.full(n, 15.0, dtype=np.float64)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_arr = vix_joined["vix_close"].fill_null(15.0).to_numpy()
        vix_ok = (vix_arr >= 12.0) & (vix_arr <= 25.0)

        # ── Session and phase masks ───────────────────────────────────────────
        in_session     = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        morning_phase  = (time_min >= 560) & (time_min < 810)   # 09:20-13:30
        afternoon_phase = (time_min >= 810) & (time_min < 925)  # 13:30-15:25

        base_ok = in_session & is_expiry & has_max_oi & vix_ok

        # ── MORNING: mean-reversion toward max-OI strike ──────────────────────
        # CE: price below pin level → expect rally back to pin
        morning_ce = (
            base_ok & morning_phase
            & (dist < -dist_threshold)
            & (mom_12 >= 0)        # 1-min momentum turning up (not still falling)
        )
        # PE: price above pin level → expect pullback back to pin
        morning_pe = (
            base_ok & morning_phase
            & (dist > dist_threshold)
            & (mom_12 <= 0)        # 1-min momentum turning down
        )

        # ── AFTERNOON: gamma-unwind momentum burst ────────────────────────────
        # CE: upside breakout above prior-1-min candle range
        afternoon_ce = (
            base_ok & afternoon_phase & valid_roll
            & (close > roll_high_12)
            & (mom_12 > 0)
        )
        # PE: downside breakout below prior-1-min candle range
        afternoon_pe = (
            base_ok & afternoon_phase & valid_roll
            & (close < roll_low_12)
            & (mom_12 < 0)
        )

        # ── Combine phases; no simultaneous long and short ────────────────────
        buy_ce = morning_ce | afternoon_ce
        buy_pe = morning_pe | afternoon_pe

        conflict = buy_ce & buy_pe
        buy_ce = buy_ce & ~conflict
        buy_pe = buy_pe & ~conflict

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts, dtype=np.float64),
            target_points=np.full(n, target_pts, dtype=np.float64),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
