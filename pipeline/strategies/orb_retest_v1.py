"""ORB Retest Strategy (orb_retest_v1)

After NIFTY breaks above (below) its 09:15-09:30 opening range high (low), institutional
VWAP algorithms and options market makers accumulate resting limit orders at that level.
When price retests it mid-session, the absorption of supply (demand) creates a detectable
15-second acceleration away from the level. We buy CE (PE) at the micro-bounce confirmation.

Hold: 30-90 seconds (time_stop_bars=18). Session: 09:30-14:30 IST.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "orb_retest_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — after ORB formation (09:15-09:30) closes
    session_end_minutes = 870     # 14:30 IST — avoid late-day positioning noise
    max_trades_per_day = 4
    max_lookback = 360            # 30 min warmup: need ORB to form + initial breakout to develop

    # ORB window: 09:15-09:30 IST
    _ORB_START = 555   # minutes from midnight
    _ORB_END = 570
    _RETEST_WINDOW = 24  # 2 minutes of bars to look back for proximity touch
    _MOMENTUM_BARS = 3   # 15 seconds for directional confirmation

    def tunable_params(self) -> list[TunableParam]:
        return [
            # proximity_pct: % of orb level within which low/high must touch (e.g. 0.10 = 0.10%)
            # On NIFTY ~22000: 0.10% ≈ 22 pts, 0.04% ≈ 9 pts, 0.20% ≈ 44 pts
            TunableParam("proximity_pct", 0.10, 0.04, 0.20),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high_arr = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low_arr = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        proximity_frac = params.get("proximity_pct", 0.10) / 100.0
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        unique_days = np.unique(day_id)

        for d in unique_days:
            day_mask = day_id == d
            day_idx = np.where(day_mask)[0]  # global indices for this day
            day_time = time_min[day_idx]

            # --- Compute ORB high/low from 09:15-09:30 bars ---
            orb_mask = (day_time >= self._ORB_START) & (day_time < self._ORB_END)
            if not np.any(orb_mask):
                continue

            orb_high = np.max(high_arr[day_idx[orb_mask]])
            orb_low = np.min(low_arr[day_idx[orb_mask]])
            orb_range = orb_high - orb_low

            # Degenerate ORB (gap day or data issue) — skip
            if orb_range < 5.0:
                continue

            # --- Post-ORB bars: track breakouts and retests ---
            post_orb_mask = day_time >= self._ORB_END
            if not np.any(post_orb_mask):
                continue

            post_orb_idx = day_idx[post_orb_mask]  # global indices post 09:30
            post_orb_time = day_time[post_orb_mask]
            m = len(post_orb_idx)

            breakout_up = False   # cumulative: NIFTY close > orb_high at any point today
            breakout_down = False  # cumulative: NIFTY close < orb_low at any point today

            for j in range(m):
                gi = post_orb_idx[j]  # global index in spot_df
                t = post_orb_time[j]

                # Update cumulative breakout flags before session filter
                if close[gi] > orb_high:
                    breakout_up = True
                if close[gi] < orb_low:
                    breakout_down = True

                # Session window filter
                if t < self.session_start_minutes or t >= self.session_end_minutes:
                    continue

                # Need enough history for proximity window + momentum look-back
                if j < self._RETEST_WINDOW + self._MOMENTUM_BARS:
                    continue

                # -- Buy CE: upside breakout, retest ORB high, bounce confirmed --
                if breakout_up and not breakout_down:
                    # Recent lows within last RETEST_WINDOW bars (2 min)
                    recent_lows = low_arr[post_orb_idx[j - self._RETEST_WINDOW:j]]
                    # At least one low touched within proximity_frac of orb_high
                    touched = np.any(
                        (recent_lows <= orb_high * (1.0 + proximity_frac)) &
                        (recent_lows >= orb_high * (1.0 - proximity_frac))
                    )
                    # Current close still above ORB high (level held as support)
                    still_above = close[gi] > orb_high
                    # 15-second positive momentum (bouncing)
                    bouncing = close[gi] > close[post_orb_idx[j - self._MOMENTUM_BARS]]

                    if touched and still_above and bouncing:
                        buy_ce[gi] = True

                # -- Buy PE: downside breakout, retest ORB low, rejection confirmed --
                if breakout_down and not breakout_up:
                    # Recent highs within last RETEST_WINDOW bars (2 min)
                    recent_highs = high_arr[post_orb_idx[j - self._RETEST_WINDOW:j]]
                    # At least one high touched within proximity_frac of orb_low
                    touched = np.any(
                        (recent_highs >= orb_low * (1.0 - proximity_frac)) &
                        (recent_highs <= orb_low * (1.0 + proximity_frac))
                    )
                    # Current close still below ORB low (level held as resistance)
                    still_below = close[gi] < orb_low
                    # 15-second negative momentum (rejecting)
                    rejecting = close[gi] < close[post_orb_idx[j - self._MOMENTUM_BARS]]

                    if touched and still_below and rejecting:
                        buy_pe[gi] = True

        # Apply session mask (belt-and-suspenders)
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        buy_ce = buy_ce & in_session
        buy_pe = buy_pe & in_session

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,             # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
