"""Vol Surface Signal: uses put-call IV skew to predict direction."""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "vol_surface_signal_v1"
    underlying = "NIFTY"
    session_start_minutes = 570  # 09:30
    session_end_minutes = 920
    max_trades_per_day = 4
    max_lookback = 240  # 20 min warmup
    assumptions = [
        "Put-call skew computed from ATM±2 strikes in nearest expiry",
        "Skew change rate signals institutional positioning shifts",
        "OTM put IV > OTM call IV = normal skew",
        "12 trading days — limited skew history",
    ]

    def tunable_params(self):
        return [
            TunableParam("skew_z_high", 2.0, 1.5, 3.0),
            TunableParam("skew_z_low", -2.0, -3.0, -1.5),
            TunableParam("stop_pts", 6.0, 3.0, 12.0),
            TunableParam("target_pts", 10.0, 5.0, 20.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        atm_strike = spot_df["atm_strike"].fill_null(strategy="forward").to_numpy().astype(np.float64)

        # Compute put-call skew from option data
        # For each bar: get nearest-expiry ATM±100 put IV proxy vs call IV proxy
        # Use option premium as IV proxy (higher premium ≈ higher IV for ATM options)
        skew = np.zeros(n)

        if option_df is not None and not option_df.is_empty():
            # Get unique session dates and compute skew per date
            spot_dates = spot_df.select("datetime", "session_date", "atm_strike").to_pandas()

            # Group option data by session_date for efficiency
            for day_id_val in spot_df["day_id"].unique().to_list():
                day_mask = spot_df["day_id"].to_numpy() == day_id_val
                if not np.any(day_mask):
                    continue

                day_indices = np.where(day_mask)[0]
                day_spot = spot_df.filter(pl.col("day_id") == day_id_val)

                if day_spot.is_empty():
                    continue

                session_date = day_spot["session_date"].head(1).to_list()[0]

                # Get options for this day
                day_opts = option_df.filter(pl.col("session_date") == session_date)
                if day_opts.is_empty():
                    continue

                # Get nearest expiry
                nearest_expiry = day_opts["expiry"].min()
                day_opts = day_opts.filter(pl.col("expiry") == nearest_expiry)

                # For each bar, compute skew
                day_atm = atm_strike[day_indices]

                for local_i, global_i in enumerate(day_indices):
                    atm = day_atm[local_i]
                    bar_time = spot_df["datetime"][global_i]

                    # Get OTM put (ATM - 100) and OTM call (ATM + 100) premiums
                    otm_put = day_opts.filter(
                        (pl.col("strike") == atm - 100) &
                        (pl.col("option_type") == "PE") &
                        (pl.col("datetime") <= bar_time)
                    )
                    otm_call = day_opts.filter(
                        (pl.col("strike") == atm + 100) &
                        (pl.col("option_type") == "CE") &
                        (pl.col("datetime") <= bar_time)
                    )

                    if not otm_put.is_empty() and not otm_call.is_empty():
                        put_prem = otm_put["close"].tail(1).to_list()[0]
                        call_prem = otm_call["close"].tail(1).to_list()[0]
                        if call_prem > 0:
                            skew[global_i] = put_prem / call_prem - 1.0

        # Skew z-score
        skew_mean = np.zeros(n)
        skew_std = np.ones(n)
        skew_z = np.zeros(n)
        lookback = 240
        for i in range(lookback, n):
            window = skew[i-lookback:i]
            skew_mean[i] = np.mean(window)
            s = np.std(window)
            skew_std[i] = s if s > 0.001 else 0.001
            skew_z[i] = (skew[i] - skew_mean[i]) / skew_std[i]

        z_high = params.get("skew_z_high", 2.0)
        z_low = params.get("skew_z_low", -2.0)
        stop_pts = params.get("stop_pts", 6.0)
        target_pts = params.get("target_pts", 10.0)

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= lookback

        # Skew very high (puts expensive vs calls) → extreme fear → contrarian BUY_CE
        buy_ce = in_session & warmed & (skew_z > z_high)

        # Skew very low (calls expensive vs puts) → extreme greed → contrarian BUY_PE
        buy_pe = in_session & warmed & (skew_z < z_low)

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
