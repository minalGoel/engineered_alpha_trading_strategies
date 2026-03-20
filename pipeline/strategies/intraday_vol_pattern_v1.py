"""Intraday Vol Pattern: exploits U-shaped volatility curve with vol surprise signals."""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "intraday_vol_pattern_v1"
    underlying = "NIFTY"
    session_start_minutes = 600  # 10:00 IST (skip first 45 min)
    session_end_minutes = 920
    max_trades_per_day = 6
    max_lookback = 180  # 15 min warmup
    assumptions = [
        "U-shaped vol pattern exists in NIFTY at 5-second frequency",
        "Vol surprise = current vol / expected vol for that time of day",
        "Expected vol estimated from prior days in dataset",
        "12 trading days — vol patterns need more data for robust estimation",
    ]

    def tunable_params(self):
        return [
            TunableParam("vol_surprise_high", 1.5, 1.2, 2.0),
            TunableParam("vol_surprise_low", 0.5, 0.3, 0.8),
            TunableParam("momentum_threshold", 0.0003, 0.0001, 0.0008),
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            TunableParam("target_pts", 8.0, 4.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        # 5-second returns
        ret = np.zeros(n)
        ret[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)

        # Rolling 180-bar (15-min) realized vol
        current_vol = np.zeros(n)
        for i in range(180, n):
            current_vol[i] = np.std(ret[i-180:i])

        # Expected vol by time bucket (estimate from PRIOR days only — no look-ahead)
        # Group by time_minutes (5-min buckets), compute RUNNING mean from past days
        time_bucket = (time_min // 5) * 5  # round to 5-min
        expected_vol = np.zeros(n)
        # Track per-bucket running sum/count using only PAST days
        bucket_sum = {}    # bucket -> cumulative sum of vol from completed days
        bucket_count = {}  # bucket -> count
        # First pass: accumulate vol per day per bucket, then use prior-day averages
        prev_day = -1
        day_bucket_vals = {}  # current day's bucket vals
        for i in range(180, n):
            bucket = time_bucket[i]
            if day_id[i] != prev_day:
                # New day: flush previous day's data into running sums
                if prev_day >= 0:
                    for b, vals in day_bucket_vals.items():
                        bucket_sum[b] = bucket_sum.get(b, 0.0) + sum(vals)
                        bucket_count[b] = bucket_count.get(b, 0) + len(vals)
                prev_day = day_id[i]
                day_bucket_vals = {}
            # Record current vol for end-of-day flush
            if bucket not in day_bucket_vals:
                day_bucket_vals[bucket] = []
            day_bucket_vals[bucket].append(current_vol[i])
            # Use only prior-day averages for expected vol
            if bucket in bucket_count and bucket_count[bucket] > 0:
                expected_vol[i] = bucket_sum[bucket] / bucket_count[bucket]
            else:
                expected_vol[i] = 0.001  # no prior data for this bucket

        # Vol surprise
        vol_surprise = np.ones(n)
        valid = expected_vol > 1e-6
        vol_surprise[valid] = current_vol[valid] / expected_vol[valid]

        # Short-term momentum (60-bar = 5-min)
        momentum = np.zeros(n)
        for i in range(60, n):
            momentum[i] = (close[i] - close[i-60]) / close[i-60]

        vol_high = params.get("vol_surprise_high", 1.5)
        vol_low = params.get("vol_surprise_low", 0.5)
        mom_thresh = params.get("momentum_threshold", 0.0003)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= 180

        # Low vol surprise (quieter than expected) + positive momentum → trend trade CE
        buy_ce = in_session & warmed & (vol_surprise < vol_low) & (momentum > mom_thresh)

        # High vol surprise (more volatile than expected) + negative momentum → fear trade PE
        buy_pe = in_session & warmed & (vol_surprise > vol_high) & (momentum < -mom_thresh)

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
