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

        # Expected vol by time bucket (estimate from data)
        # Group by time_minutes (5-min buckets) and compute mean vol
        time_bucket = (time_min // 5) * 5  # round to 5-min
        expected_vol = np.zeros(n)
        bucket_vols = {}
        for i in range(180, n):
            bucket = time_bucket[i]
            if bucket not in bucket_vols:
                bucket_vols[bucket] = []
            bucket_vols[bucket].append(current_vol[i])
        # Compute running mean per bucket
        bucket_means = {b: np.mean(v) for b, v in bucket_vols.items() if len(v) > 0}
        for i in range(n):
            bucket = time_bucket[i]
            expected_vol[i] = bucket_means.get(bucket, 0.001)

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
