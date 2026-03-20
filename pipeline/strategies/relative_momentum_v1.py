"""
relative_momentum_v1 — Intraday Momentum Acceleration Strategy

Original: Stock intraday return > 2× NIFTY intraday return signals persistent
institutional buying continuation.

Adaptation: On NIFTY itself, compare the 2-minute return rate to the session
drift rate. When recent rate exceeds 2× the session drift rate, TWAP algorithms
have entered an urgency phase — buy CE (if session up) or PE (if session down)
to ride the 30-90 second continuation burst.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "relative_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST
    session_end_minutes = 870     # 14:30 IST
    max_trades_per_day = 8
    max_lookback = 288            # 24 min warmup to establish meaningful session drift rate

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rel_ratio_threshold", 2.0, 1.5, 4.0),
            TunableParam("session_min_return", 0.05, 0.02, 0.15),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        rel_ratio_threshold = params.get("rel_ratio_threshold", 2.0)
        session_min_return = params.get("session_min_return", 0.05)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # ── Compute session drift and recent acceleration ──
        session_return = np.zeros(n)   # % return from day open
        recent_return_24 = np.zeros(n)  # 2-min (24 bars × 5s) return %
        session_rate = np.zeros(n)     # per-bar session drift rate
        recent_rate = np.zeros(n)      # per-bar recent drift rate

        day_open = np.zeros(n)
        current_day_open = close[0]
        current_day_id = day_id[0]
        day_start_idx = 0

        for i in range(n):
            # Detect new trading day
            if day_id[i] != current_day_id:
                current_day_id = day_id[i]
                current_day_open = close[i]
                day_start_idx = i

            day_open[i] = current_day_open

            # Session cumulative return (% from day open)
            if current_day_open > 0:
                session_return[i] = (close[i] - current_day_open) / current_day_open * 100.0

            # Session drift rate: per-bar average drift since open
            bars_from_open = max(1, i - day_start_idx)
            session_rate[i] = session_return[i] / bars_from_open

            # Recent 2-minute return (24 bars = 120s)
            if i >= 24:
                base = close[i - 24]
                if base > 0:
                    recent_return_24[i] = (close[i] - base) / base * 100.0
            recent_rate[i] = recent_return_24[i] / 24.0

        # ── Build signal masks ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        has_warmup = np.arange(n) >= self.max_lookback

        session_pos = session_return > session_min_return
        session_neg = session_return < -session_min_return

        # buy_ce: session trending up AND recent rate > ratio × session rate AND recent burst up
        # recent_rate > rel_ratio_threshold * session_rate with both positive:
        # e.g. session_rate=0.001, threshold=2.0 → recent_rate must be > 0.002
        buy_ce = (
            in_session
            & has_warmup
            & session_pos
            & (recent_rate > rel_ratio_threshold * session_rate)
            & (recent_return_24 > 0.0)
        )

        # buy_pe: session trending down AND recent rate < ratio × session rate (both negative)
        # e.g. session_rate=-0.001, threshold=2.0 → recent_rate must be < -0.002
        buy_pe = (
            in_session
            & has_warmup
            & session_neg
            & (recent_rate < rel_ratio_threshold * session_rate)
            & (recent_return_24 < 0.0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,   # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
