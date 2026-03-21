"""Gap-and-Go with Volume — NIFTY 5-second options strategy.

When NIFTY gaps open with elevated institutional participation (first-5-minute
cumulative volume > 2.5x the session average rate), VWAP-benchmarked execution
algorithms re-engage aggressively on the first pullback to session VWAP. At 5s
resolution, we detect the exact VWAP touch bar and enter before the continuation
leg, targeting 15-30 spot points of institutional re-accumulation.

Original: Strategy_49.json (gap_and_go_with_volume_v1, 1-min FnO equity)
"""

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "gap_and_go_with_volume_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip pre-open noise
    session_end_minutes = 925     # 15:25 IST — EOD flatten
    max_lookback = 120            # 10 min warmup (120 × 5s)
    max_trades_per_day = 3        # gap trades are rare; 0-3 per day

    # Entry window: gap-and-go only reliable in first 45 min
    _entry_end_minutes = 600      # 10:00 IST

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.004, 0.002, 0.010),      # 0.4% default
            TunableParam("vol_ratio_threshold", 2.5, 1.5, 5.0),       # 2.5x default
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before numpy) ──────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        open_px = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        gap_threshold = params.get("gap_threshold", 0.004)
        vol_ratio_threshold = params.get("vol_ratio_threshold", 2.5)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── VIX filter ────────────────────────────────────────────────────────
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

        # ── Per-bar arrays initialised to neutral ────────────────────────────
        gap_pct = np.zeros(n)          # positive = gap-up, negative = gap-down
        session_vwap = np.zeros(n)     # cumulative VWAP from 09:15 per day
        vol_ratio = np.zeros(n)        # opening 5-min vol / morning session rate

        # ── Per-day computations ──────────────────────────────────────────────
        unique_days = np.unique(day_id)

        for d in unique_days:
            day_mask = day_id == d
            day_idxs = np.where(day_mask)[0]
            if len(day_idxs) == 0:
                continue

            day_close = close[day_idxs]
            day_high = high[day_idxs]
            day_low = low[day_idxs]
            day_volume = volume[day_idxs]
            day_time = time_min[day_idxs]

            # ── Gap: first-bar open vs prev day last close ────────────────
            first_idx = day_idxs[0]
            day_open_0 = open_px[first_idx]  # opening auction price at 09:15
            if first_idx > 0:
                prev_close = close[first_idx - 1]
                day_gap = (
                    (day_open_0 - prev_close) / prev_close
                    if prev_close > 0
                    else 0.0
                )
            else:
                day_gap = 0.0
            gap_pct[day_idxs] = day_gap

            # ── Session VWAP (cumulative from first bar of day) ───────────
            cum_pv = np.cumsum(day_close * day_volume)
            cum_vol = np.cumsum(day_volume)
            cum_vol_safe = np.where(cum_vol > 0, cum_vol, 1.0)
            day_vwap = cum_pv / cum_vol_safe
            session_vwap[day_idxs] = day_vwap

            # ── Volume ratio: opening 5 min vs morning session rate ───────
            # Opening window: 09:15-09:20 (time_minutes 555-559)
            open_5min_mask = (day_time >= 555) & (day_time < 560)
            open_5min_vol = (
                np.sum(day_volume[open_5min_mask])
                if np.any(open_5min_mask)
                else 0.0
            )

            # Baseline: morning session 09:20-13:00 (time_minutes 560-779)
            # Avoids including the opening 5 min in its own denominator
            morning_mask = (day_time >= 560) & (day_time < 780)
            morning_bars = int(np.sum(morning_mask))
            morning_vol = (
                np.sum(day_volume[morning_mask]) if morning_bars > 0 else 0.0
            )

            if morning_bars > 0 and morning_vol > 0:
                avg_vol_per_bar = morning_vol / morning_bars
                expected_5min_vol = avg_vol_per_bar * 60  # normalise to 5-min
                day_vol_ratio = (
                    open_5min_vol / expected_5min_vol
                    if expected_5min_vol > 0
                    else 0.0
                )
            else:
                day_vol_ratio = 0.0

            vol_ratio[day_idxs] = day_vol_ratio

        # ── VWAP touch detection (3-bar lookback, respects day boundaries) ──
        # Touch = the bar's high-low range straddles session VWAP
        sv_safe = np.where(session_vwap > 0, session_vwap, np.inf)
        vwap_in_bar = (low <= sv_safe) & (high >= sv_safe)

        # Rolling 3-bar: did a VWAP touch occur in bars [i-3, i-2, i-1]?
        recent_touch = np.zeros(n, dtype=bool)
        # Use shift approach with day-boundary guard
        day_start = np.zeros(n, dtype=bool)
        for d in unique_days:
            idxs = np.where(day_id == d)[0]
            if len(idxs) > 0:
                day_start[idxs[0]] = True

        # Build a "days elapsed" cumsum to detect cross-day shifts
        day_boundary_cumsum = np.cumsum(day_start)

        for shift in (1, 2, 3):
            if n <= shift:
                continue
            shifted_touch = np.zeros(n, dtype=bool)
            shifted_touch[shift:] = vwap_in_bar[:-shift]
            # Zero out bars that crossed a day boundary
            shifted_day = np.zeros(n, dtype=int)
            shifted_day[shift:] = day_boundary_cumsum[:-shift]
            crossed_boundary = day_boundary_cumsum != shifted_day
            shifted_touch[crossed_boundary] = False
            recent_touch |= shifted_touch

        # ── Entry conditions ──────────────────────────────────────────────────
        # Time: entry window 09:20-10:00 IST only
        in_entry_window = (
            (time_min >= self.session_start_minutes)
            & (time_min < self._entry_end_minutes)
        )

        vix_ok = vix_close < 22.0

        sv_nonzero = np.where(session_vwap > 0, session_vwap, 1.0)

        # Gap-up + high opening vol + VWAP touch + close back above VWAP → buy CE
        gap_up = gap_pct > gap_threshold
        price_above_vwap = close > sv_nonzero
        buy_ce = (
            in_entry_window
            & vix_ok
            & gap_up
            & (vol_ratio > vol_ratio_threshold)
            & recent_touch
            & price_above_vwap
        )

        # Gap-down + high opening vol + VWAP touch + close back below VWAP → buy PE
        gap_dn = gap_pct < -gap_threshold
        price_below_vwap = close < sv_nonzero
        buy_pe = (
            in_entry_window
            & vix_ok
            & gap_dn
            & (vol_ratio > vol_ratio_threshold)
            & recent_touch
            & price_below_vwap
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
