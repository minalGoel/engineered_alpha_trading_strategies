"""
orb_wide_range_fade_v1 — NIFTY Opening Range Wide-Range Fade

Mechanism: On NIFTY, when the 09:15-09:30 opening range exceeds 2x its 20-session
average, institutional option writers at the ORB extreme strikes become heavily
gamma-exposed and delta-hedge against further extension. Simultaneously, VWAP-
benchmarked desks begin fading the deviation to protect their intraday benchmark.
Each subsequent test of the ORB extreme generates a 10-20 spot point micro-rejection
within 30-90 seconds. We enter the instant RSI(36) confirms exhaustion at the extreme,
capturing the first leg of the reversion before the move extends.

Converted from: trading_strategies/unique_strategies_all/Strategy_228.json
Original: Fade NIFTY50 stocks when 15-min ORB > 2x avg_ORB_range_20d, hold 20-120 min.
Adaptation: Same wide-ORB context + RSI exhaustion trigger, but targeting the micro-
rejection at each ORB extreme test (15-120s hold) not the full session reversion.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI on a numpy array."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.empty(n)
    delta[0] = 0.0
    delta[1:] = close[1:] - close[:-1]

    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    # Seed with simple average over first period
    avg_gain[period] = np.mean(gains[1 : period + 1])
    avg_loss[period] = np.mean(losses[1 : period + 1])

    # Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    for i in range(period, n):
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


class Strategy(BaseStrategy):
    name = "orb_wide_range_fade_v1"
    underlying = "NIFTY"
    session_start_minutes = 575   # 09:35 IST — after ORB has fully formed
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 240            # 20 min warmup (ORB window + RSI warmup)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("wide_range_mult", 2.0, 1.5, 3.0),
            TunableParam("orb_entry_pct", 0.15, 0.08, 0.25),
            TunableParam("rsi_overbought", 70.0, 60.0, 80.0),
            TunableParam("rsi_oversold", 30.0, 20.0, 40.0),
            TunableParam("stop_pts", 3.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        wide_range_mult = params.get("wide_range_mult", 2.0)
        orb_entry_pct = params.get("orb_entry_pct", 0.15)
        rsi_overbought = params.get("rsi_overbought", 70.0)
        rsi_oversold = params.get("rsi_oversold", 30.0)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 6.0)

        # ORB window: 09:15–09:30 IST (minutes 555 to 570)
        ORB_START = 555   # inclusive
        ORB_END = 570     # exclusive (09:30 bar not included)

        # ── Step 1: Compute per-session ORB ranges ─────────────────────────────
        unique_days = np.unique(day_id)
        day_orb_data: dict[int, tuple[float, float, float]] = {}  # day -> (high, low, range)

        for d in unique_days:
            mask_orb = (day_id == d) & (time_min >= ORB_START) & (time_min < ORB_END)
            if not np.any(mask_orb):
                continue
            d_high = float(np.max(high[mask_orb]))
            d_low = float(np.min(low[mask_orb]))
            d_range = d_high - d_low
            day_orb_data[d] = (d_high, d_low, d_range)

        # ── Step 2: Rolling 20-session average ORB range ──────────────────────
        sorted_days = sorted(day_orb_data.keys())
        day_avg_orb: dict[int, float] = {}

        for i, d in enumerate(sorted_days):
            # Use up to 20 PREVIOUS sessions (exclude current)
            prev_window = sorted_days[max(0, i - 20) : i]
            if not prev_window:
                # No history — use current range as baseline (no signal until history builds)
                day_avg_orb[d] = day_orb_data[d][2]
            else:
                avg = float(np.mean([day_orb_data[w][2] for w in prev_window]))
                day_avg_orb[d] = avg if avg > 0 else day_orb_data[d][2]

        # ── Step 3: Fill per-bar arrays ────────────────────────────────────────
        orb_high_arr = np.zeros(n)
        orb_low_arr = np.zeros(n)
        orb_range_arr = np.zeros(n)
        is_wide_range = np.zeros(n, dtype=bool)

        for d in unique_days:
            if d not in day_orb_data:
                continue
            d_high, d_low, d_range = day_orb_data[d]
            avg_range = day_avg_orb.get(d, d_range)

            d_mask = day_id == d
            orb_high_arr[d_mask] = d_high
            orb_low_arr[d_mask] = d_low
            orb_range_arr[d_mask] = d_range

            if d_range > wide_range_mult * avg_range and avg_range > 0:
                is_wide_range[d_mask] = True

        # ── Step 4: Price position within ORB range (0=low, 1=high) ───────────
        safe_range = np.where(orb_range_arr > 0, orb_range_arr, 1.0)
        price_position = (close - orb_low_arr) / safe_range

        # ── Step 5: RSI(36) — 3-minute RSI for micro-exhaustion ───────────────
        # 36 bars × 5s = 3 min. Detects intrabar overbought/oversold at the ORB
        # extreme rather than session-level reversal (original 14-min RSI).
        rsi = _compute_rsi(close, 36)

        # ── Step 6: VIX filter ─────────────────────────────────────────────────
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

        # VIX < 22: extreme VIX days may extend the ORB rather than revert
        vix_ok = vix_close < 22.0

        # ── Step 7: Session filter ─────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # ORB must have formed (range > 0 means ORB bars were seen)
        orb_formed = orb_range_arr > 0

        # ── Step 8: Entry signals ──────────────────────────────────────────────
        # buy_pe: price near ORB_high + RSI overbought on wide-range day → fade down
        buy_pe = (
            in_session
            & is_wide_range
            & orb_formed
            & (price_position >= (1.0 - orb_entry_pct))
            & (rsi >= rsi_overbought)
            & vix_ok
        )

        # buy_ce: price near ORB_low + RSI oversold on wide-range day → fade up
        buy_ce = (
            in_session
            & is_wide_range
            & orb_formed
            & (price_position <= orb_entry_pct)
            & (rsi <= rsi_oversold)
            & vix_ok
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )
