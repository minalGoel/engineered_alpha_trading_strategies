"""gap_fade_v1 — Large NIFTY Gap Congestion-Zone Fade

Mechanism: On NIFTY, overnight gaps ≥0.5% cause a 30-60s burst of retail/algo momentum
in the gap direction, which exhausts into a tight congestion band as institutional VWAP
algorithms begin fading toward previous close. We detect this congestion zone via 90-second
range compression and enter on the first 30-second momentum break opposite the gap direction.

Differentiated from gap_fade_reversion_v103 (immediate reversal + volume, 0.3% threshold,
VIX-capped) and gap_fade_large_v1 (RSI exhaustion + extension-from-open, 0.3% threshold,
VIX-capped) by: (1) higher gap threshold targeting the large-gap-magnitude regime,
(2) congestion zone detection via range compression (not RSI or momentum confirmation),
(3) no VIX cap — deliberate coverage of elevated-VIX sessions the others exclude.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


def _compute_gap_pct(day_id: np.ndarray, open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Return per-bar gap_pct = (day_open - prev_day_close) / prev_day_close.

    Constant within each day. Zero for the first day in the dataset (no prev close).
    """
    n = len(day_id)
    gap_pct = np.zeros(n)

    # Find first bar index and last close for each day
    first_bar: dict[int, int] = {}
    last_close: dict[int, float] = {}
    for i in range(n):
        d = int(day_id[i])
        if d not in first_bar:
            first_bar[d] = i
        last_close[d] = close[i]

    day_ids_sorted = sorted(first_bar.keys())

    # Map each day to previous day's close
    prev_close_map: dict[int, float] = {}
    for k, d in enumerate(day_ids_sorted):
        if k > 0:
            prev_close_map[d] = last_close[day_ids_sorted[k - 1]]

    # Fill gap_pct per bar
    for i in range(n):
        d = int(day_id[i])
        if d in prev_close_map:
            pc = prev_close_map[d]
            if pc > 0:
                fb = first_bar[d]
                gap_pct[i] = (open_[fb] - pc) / pc

    return gap_pct


class Strategy(BaseStrategy):
    name = "gap_fade_v1"
    underlying = "NIFTY"
    session_start_minutes = 555   # 09:15 IST — gap fade must be active from open
    session_end_minutes = 565     # 09:25 IST — gap fade thesis expires after first 10 minutes
    max_trades_per_day = 3
    max_lookback = 36             # 3 min warmup — enough for 18-bar range + 6-bar momentum

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Gap threshold: 0.3%-1.0% of NIFTY (60-200 spot pts)
            TunableParam("gap_threshold", 0.005, 0.003, 0.010),
            # Congestion zone: max 90-second range as fraction of close
            TunableParam("range_threshold", 0.0007, 0.0004, 0.0014),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 9.0, 5.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN in Polars before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        gap_threshold = params.get("gap_threshold", 0.005)
        range_threshold = params.get("range_threshold", 0.0007)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 9.0)

        # --- Indicator 1: gap_pct (daily scalar, constant within each day) ---
        gap_pct = _compute_gap_pct(day_id, open_, close)

        # --- Indicator 2: range_18 — 90-second rolling range as fraction of close ---
        # Initialized to np.inf so that uncomputed bars fail the < range_threshold test.
        # Cross-day windows are excluded by the same-day check (bars spanning yesterday's
        # close have range > 0.01 anyway due to overnight gap, but we exclude explicitly).
        range_18 = np.full(n, np.inf)
        for i in range(18, n):
            # Oldest bar in window is i-18; newest is i-1. Check all are same day.
            if day_id[i - 18] == day_id[i]:
                h = np.max(high[i - 18:i])
                lo = np.min(low[i - 18:i])
                if close[i] > 0:
                    range_18[i] = (h - lo) / close[i]

        # --- Indicator 3: mom_6 — 30-second return ---
        # Exclude cross-day momentum (would include overnight gap).
        mom_6 = np.zeros(n)
        for i in range(6, n):
            if day_id[i - 6] == day_id[i] and close[i - 6] > 0:
                mom_6[i] = (close[i] - close[i - 6]) / close[i - 6]

        # --- Entry window: 09:15-09:25 IST only ---
        entry_window = (time_min >= 555) & (time_min < 565)

        # --- Signals ---
        # Buy CE: gap down + congestion zone formed + momentum turning up (fading gap-down)
        buy_ce = (
            entry_window
            & (gap_pct < -gap_threshold)
            & (range_18 < range_threshold)     # congestion zone active (inf excluded)
            & (mom_6 > 0)                      # 30s break upward from congestion
        )

        # Buy PE: gap up + congestion zone formed + momentum turning down (fading gap-up)
        buy_pe = (
            entry_window
            & (gap_pct > gap_threshold)
            & (range_18 < range_threshold)
            & (mom_6 < 0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,              # 60-second time stop
            max_trades_per_day=self.max_trades_per_day,
        )
