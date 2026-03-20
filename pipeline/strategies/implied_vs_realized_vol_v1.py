"""Variance Risk Premium: exploits systematic VIX overpricing of realized vol."""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "implied_vs_realized_vol_v1"
    underlying = "NIFTY"
    session_start_minutes = 570
    session_end_minutes = 920
    max_trades_per_day = 3
    max_lookback = 720  # 60 min warmup for VRP estimation
    assumptions = [
        "VRP estimated from VIX minus 60-min realized vol",
        "VRP z-score computed over 720-bar (1-hour) rolling window",
        "12 trading days - VRP estimation is noisy with this little data",
    ]

    def tunable_params(self):
        return [
            TunableParam("vrp_z_high", 2.0, 1.5, 3.0),
            TunableParam("vrp_z_low", -2.0, -3.0, -1.5),
            TunableParam("stop_pts", 6.0, 3.0, 12.0),
            TunableParam("target_pts", 10.0, 5.0, 20.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)

        # VIX
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_c")]).sort("datetime"),
                on="datetime", strategy="backward",
            )
            vix_close = joined["vix_c"].fill_null(15.0).to_numpy().astype(np.float64)

        # Realized vol
        log_ret = np.zeros(n)
        log_ret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-10))
        annualize = np.sqrt(252 * 4500)

        rv = np.zeros(n)
        vrp = np.zeros(n)
        vrp_mean = np.zeros(n)
        vrp_std = np.ones(n)
        vrp_z = np.zeros(n)

        for i in range(720, n):
            rv[i] = np.std(log_ret[i-720:i]) * annualize * 100  # in % like VIX
            vrp[i] = vix_close[i] - rv[i]

        for i in range(720, n):
            window = vrp[max(0, i-720):i]
            vrp_mean[i] = np.mean(window)
            s = np.std(window)
            vrp_std[i] = s if s > 0.1 else 0.1
            vrp_z[i] = (vrp[i] - vrp_mean[i]) / vrp_std[i]

        z_high = params.get("vrp_z_high", 2.0)
        z_low = params.get("vrp_z_low", -2.0)
        stop_pts = params.get("stop_pts", 6.0)
        target_pts = params.get("target_pts", 10.0)

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= 720

        # VRP very high (VIX way above RV) → fear overpriced → BUY_CE
        buy_ce = in_session & warmed & (vrp_z > z_high)

        # VRP very low (VIX below RV) → complacency → BUY_PE
        buy_pe = in_session & warmed & (vrp_z < z_low)

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
