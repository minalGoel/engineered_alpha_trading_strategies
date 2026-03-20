"""IV-RV Dislocation: trades mean reversion of VIX relative to realized vol."""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "implied_vol_dislocation_v1"
    underlying = "NIFTY"
    session_start_minutes = 570  # 09:30 IST
    session_end_minutes = 920    # 15:20 IST
    max_trades_per_day = 5
    max_lookback = 360  # 30 min warmup
    assumptions = [
        "VIX serves as proxy for ATM implied volatility",
        "Realized vol computed from 5-second index returns",
        "12 trading days — insufficient for robust vol estimates",
    ]

    def tunable_params(self):
        return [
            TunableParam("iv_rv_high", 1.3, 1.1, 1.6),
            TunableParam("iv_rv_low", 0.7, 0.5, 0.9),
            TunableParam("vix_slope_threshold", 0.002, 0.0005, 0.005),
            TunableParam("stop_pts", 5.0, 3.0, 10.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)

        # Get VIX aligned to spot bars
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_c")]).sort("datetime"),
                on="datetime", strategy="backward",
            )
            vc = joined["vix_c"].fill_null(15.0).to_numpy().astype(np.float64)
            vix_close = vc

        # Compute 5-second log returns
        log_ret = np.zeros(n)
        log_ret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-10))

        # Realized vol: rolling 360-bar (30-min) std of returns, annualized
        # Trading session = 6.25 hours = 22500 seconds = 4500 bars
        annualize_factor = np.sqrt(252 * 4500)
        rv = np.zeros(n)
        lookback = 360
        for i in range(lookback, n):
            rv[i] = np.std(log_ret[i-lookback:i]) * annualize_factor

        # IV from VIX (already annualized %)
        iv = vix_close / 100.0

        # IV / RV ratio
        iv_rv = np.ones(n)
        valid = rv > 0.01
        iv_rv[valid] = iv[valid] / rv[valid]

        # VIX slope: rate of change over 12 bars (1 minute)
        vix_slope = np.zeros(n)
        for i in range(12, n):
            if vix_close[i-12] > 0:
                vix_slope[i] = (vix_close[i] - vix_close[i-12]) / vix_close[i-12]

        # Parameters
        iv_rv_high = params.get("iv_rv_high", 1.3)
        iv_rv_low = params.get("iv_rv_low", 0.7)
        vix_slope_thresh = params.get("vix_slope_threshold", 0.002)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= lookback

        # VIX overpricing fear + VIX starting to fall → BUY_CE (expect rally)
        buy_ce = in_session & warmed & (iv_rv > iv_rv_high) & (vix_slope < -vix_slope_thresh)

        # VIX underpricing risk + VIX starting to rise → BUY_PE (expect drop)
        buy_pe = in_session & warmed & (iv_rv < iv_rv_low) & (vix_slope > vix_slope_thresh)

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
