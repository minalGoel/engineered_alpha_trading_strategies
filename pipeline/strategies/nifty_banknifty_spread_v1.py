"""NIFTY-BANKNIFTY Spread: trades mean reversion of index spread."""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "nifty_banknifty_spread_v1"
    underlying = "NIFTY"  # trade NIFTY options based on spread signal
    session_start_minutes = 570
    session_end_minutes = 920
    max_trades_per_day = 5
    max_lookback = 360  # 30 min warmup
    assumptions = [
        "BANKNIFTY data loaded separately and joined to NIFTY spot",
        "Spread computed from cumulative returns, not raw prices",
        "Mean reversion assumes normal correlation regime",
        "12 trading days — spread dynamics may not be representative",
    ]

    def tunable_params(self):
        return [
            TunableParam("z_threshold", 2.0, 1.5, 3.0),
            TunableParam("velocity_filter", 0.0001, 0.00005, 0.0005),
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)

        # We need BANKNIFTY data — load it from the same spot file
        from pipeline.config import SPOT_FILE, BANKNIFTY_SYMBOL
        from pipeline.data_loader import load_spot_data

        bn_df = load_spot_data(BANKNIFTY_SYMBOL)
        bn_close = np.full(n, 0.0)

        if bn_df is not None and not bn_df.is_empty():
            # Join BANKNIFTY close to NIFTY bars
            joined = spot_df.select("datetime").join_asof(
                bn_df.select(["datetime", pl.col("close").alias("bn_close")]).sort("datetime"),
                on="datetime", strategy="backward",
            )
            bn_close = joined["bn_close"].fill_null(strategy="forward").fill_null(50000.0).to_numpy().astype(np.float64)

        # Cumulative returns from day start
        nifty_ret = np.zeros(n)
        bn_ret = np.zeros(n)

        prev_day = -1
        nifty_base = close[0]
        bn_base = bn_close[0]

        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                nifty_base = close[i]
                bn_base = bn_close[i] if bn_close[i] > 0 else 50000.0
            if nifty_base > 0:
                nifty_ret[i] = (close[i] - nifty_base) / nifty_base
            if bn_base > 0:
                bn_ret[i] = (bn_close[i] - bn_base) / bn_base

        # Spread: NIFTY return - BANKNIFTY return
        spread = nifty_ret - bn_ret

        # Z-score of spread
        spread_z = np.zeros(n)
        lookback = 360
        for i in range(lookback, n):
            window = spread[i-lookback:i]
            mean = np.mean(window)
            std = np.std(window)
            if std > 1e-6:
                spread_z[i] = (spread[i] - mean) / std

        # Spread velocity (rate of change)
        spread_vel = np.zeros(n)
        for i in range(12, n):
            spread_vel[i] = spread[i] - spread[i-12]

        z_thresh = params.get("z_threshold", 2.0)
        vel_filter = params.get("velocity_filter", 0.0001)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= lookback

        # NIFTY lagging (spread very negative) + spread reversing up → BUY_CE (NIFTY catching up)
        buy_ce = in_session & warmed & (spread_z < -z_thresh) & (spread_vel > vel_filter)

        # NIFTY leading (spread very positive) + spread reversing down → BUY_PE (NIFTY giving back)
        buy_pe = in_session & warmed & (spread_z > z_thresh) & (spread_vel < -vel_filter)

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
