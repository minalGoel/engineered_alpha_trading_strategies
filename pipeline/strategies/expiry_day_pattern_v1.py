"""Expiry Day Pattern: exploits gamma hedging and pin risk on expiry days."""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "expiry_day_pattern_v1"
    underlying = "NIFTY"
    session_start_minutes = 570  # 09:30
    session_end_minutes = 900    # 15:00 (earlier exit on expiry)
    max_trades_per_day = 4
    max_lookback = 120  # 10 min warmup
    assumptions = [
        "Expiry detected from option_df expiry dates matching session_date",
        "Gamma squeeze: max OI strike acts as magnet",
        "Pin risk accelerates in last 2 hours",
        "12 trading days — may contain 2-3 expiry days only",
    ]

    def tunable_params(self):
        return [
            TunableParam("distance_threshold", 0.003, 0.001, 0.008),
            TunableParam("momentum_confirm", 0.0002, 0.00005, 0.0005),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        # Detect expiry days
        is_expiry_day = np.zeros(n, dtype=bool)
        max_pain_strike = np.zeros(n)

        if option_df is not None and not option_df.is_empty():
            expiry_dates = set(option_df["expiry"].unique().to_list())
            session_dates = spot_df["session_date"].to_list()

            for i in range(n):
                if session_dates[i] in expiry_dates:
                    is_expiry_day[i] = True

            # Max pain per expiry day
            for day_val in spot_df["day_id"].unique().to_list():
                day_mask = day_id == day_val
                if not np.any(day_mask & is_expiry_day):
                    continue

                day_indices = np.where(day_mask)[0]
                session_date = spot_df["session_date"][day_indices[0]]

                day_opts = option_df.filter(
                    (pl.col("session_date") == session_date) &
                    (pl.col("expiry") == session_date)
                )
                if day_opts.is_empty():
                    continue

                oi_by_strike = day_opts.group_by("strike").agg(
                    pl.col("open_interest").max().alias("max_oi")
                ).sort("max_oi", descending=True)

                if not oi_by_strike.is_empty():
                    mp = oi_by_strike["strike"].head(1).to_list()[0]
                    max_pain_strike[day_indices] = mp

        # Distance to max pain
        distance_to_mp = np.zeros(n)
        valid_mp = max_pain_strike > 0
        distance_to_mp[valid_mp] = (close[valid_mp] - max_pain_strike[valid_mp]) / close[valid_mp]

        # Short-term momentum
        momentum = np.zeros(n)
        for i in range(60, n):
            momentum[i] = (close[i] - close[i - 60]) / close[i - 60]

        dist_thresh = params.get("distance_threshold", 0.003)
        mom_confirm = params.get("momentum_confirm", 0.0002)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        buy_ce = (in_session & warmed & is_expiry_day &
                  (distance_to_mp < -dist_thresh) & (momentum > mom_confirm))

        buy_pe = (in_session & warmed & is_expiry_day &
                  (distance_to_mp > dist_thresh) & (momentum < -mom_confirm))

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,
            max_trades_per_day=self.max_trades_per_day,
        )
