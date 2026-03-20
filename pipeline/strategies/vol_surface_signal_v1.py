"""Vol Surface Signal: uses put-call IV skew to predict direction.

FIXED: Skew is computed from FIXED reference strikes per day (day-open ATM ±100),
not from shifting ATM. This eliminates discontinuities when ATM shifts bar-to-bar.
"""
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
        "Put-call skew computed from FIXED day-open ATM±100 strikes in nearest expiry",
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
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)
        atm_strike = spot_df["atm_strike"].fill_null(strategy="forward").to_numpy().astype(np.float64)

        skew = np.zeros(n)

        if option_df is not None and not option_df.is_empty():
            datetimes = spot_df["datetime"].to_list()

            for day_id_val in spot_df["day_id"].unique().to_list():
                day_mask = day_id == day_id_val
                day_indices = np.where(day_mask)[0]
                if len(day_indices) == 0:
                    continue

                day_spot = spot_df.filter(pl.col("day_id") == day_id_val)
                if day_spot.is_empty():
                    continue

                session_date = day_spot["session_date"].head(1).to_list()[0]

                # FIXED: Use the ATM strike at the FIRST bar of the day as
                # the reference for the entire day. This prevents strike-switching
                # noise in the skew indicator.
                day_open_atm = atm_strike[day_indices[0]]
                fixed_put_strike = day_open_atm - 100  # OTM put
                fixed_call_strike = day_open_atm + 100  # OTM call

                day_opts = option_df.filter(pl.col("session_date") == session_date)
                if day_opts.is_empty():
                    continue

                nearest_expiry = day_opts["expiry"].min()
                day_opts = day_opts.filter(pl.col("expiry") == nearest_expiry)

                # Pre-fetch the two fixed strike series for the entire day
                put_series = day_opts.filter(
                    (pl.col("strike") == fixed_put_strike) &
                    (pl.col("option_type") == "PE")
                ).sort("datetime")

                call_series = day_opts.filter(
                    (pl.col("strike") == fixed_call_strike) &
                    (pl.col("option_type") == "CE")
                ).sort("datetime")

                if put_series.is_empty() or call_series.is_empty():
                    continue

                put_times = put_series["datetime"].to_list()
                put_closes = put_series["close"].to_numpy()
                call_times = call_series["datetime"].to_list()
                call_closes = call_series["close"].to_numpy()

                # Two-pointer fill: for each spot bar, get latest put and call premium
                put_idx = 0
                call_idx = 0
                sample_step = 12  # every minute

                for local_i in range(0, len(day_indices), sample_step):
                    global_i = int(day_indices[local_i])
                    bar_time = datetimes[global_i]

                    # Advance put pointer
                    while put_idx < len(put_times) - 1 and put_times[put_idx + 1] <= bar_time:
                        put_idx += 1
                    # Advance call pointer
                    while call_idx < len(call_times) - 1 and call_times[call_idx + 1] <= bar_time:
                        call_idx += 1

                    if (put_idx < len(put_times) and put_times[put_idx] <= bar_time and
                            call_idx < len(call_times) and call_times[call_idx] <= bar_time):
                        put_prem = put_closes[put_idx]
                        call_prem = call_closes[call_idx]
                        if call_prem > 0:
                            s = put_prem / call_prem - 1.0
                            end = min(local_i + sample_step, len(day_indices))
                            for j in range(local_i, end):
                                skew[int(day_indices[j])] = s

        # Skew z-score
        skew_z = np.zeros(n)
        lookback = 240
        for i in range(lookback, n):
            window = skew[i-lookback:i]
            mean = np.mean(window)
            std = np.std(window)
            if std > 0.001:
                skew_z[i] = (skew[i] - mean) / std

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
