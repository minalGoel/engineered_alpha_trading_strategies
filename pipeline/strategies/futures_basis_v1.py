"""Futures Basis Change → Spot Direction: futures premium expansion predicts spot.

When futures premium over spot expands rapidly, it signals aggressive buying
in the derivative market that hasn't yet reflected in spot. The spot tends
to catch up. Use BANKNIFTY futures-spot basis as a leading indicator for
NIFTY direction (since BANKNIFTY futures are more actively traded).

Note: We approximate futures basis from the option put-call parity relationship
since we don't have direct futures data. ATM CE - ATM PE ≈ futures - strike.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "futures_basis_v1"
    underlying = "NIFTY"
    session_start_minutes = 570  # 09:30 IST (need option data to stabilize)
    session_end_minutes = 925
    max_trades_per_day = 8
    max_lookback = 72

    def tunable_params(self):
        return [
            TunableParam("basis_change_threshold", 3.0, 1.5, 6.0),
            TunableParam("stop_pts", 4.0, 2.0, 7.0),
            TunableParam("target_pts", 8.0, 5.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params):
        n = len(spot_df)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)
        atm_strike = spot_df["atm_strike"].fill_null(strategy="forward").to_numpy().astype(np.float64)

        basis_thresh = params.get("basis_change_threshold", 3.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # Compute synthetic futures from put-call parity
        # F ≈ strike + CE_premium - PE_premium
        synth_future = np.zeros(n)

        if option_df is not None and not option_df.is_empty():
            datetimes = spot_df["datetime"].to_list()
            session_dates = spot_df["session_date"].to_list()

            for day_val in sorted(spot_df["day_id"].unique().to_list()):
                day_mask = day_id == day_val
                day_indices = np.where(day_mask)[0]
                if len(day_indices) == 0:
                    continue

                sd = session_dates[day_indices[0]]
                day_opts = option_df.filter(pl.col("session_date") == sd)
                if day_opts.is_empty():
                    continue

                nearest_expiry = day_opts["expiry"].min()
                day_opts = day_opts.filter(pl.col("expiry") == nearest_expiry)

                # Get ATM strike for this day
                day_atm = atm_strike[day_indices[0]]

                ce_series = day_opts.filter(
                    (pl.col("strike") == day_atm) & (pl.col("option_type") == "CE")
                ).sort("datetime")
                pe_series = day_opts.filter(
                    (pl.col("strike") == day_atm) & (pl.col("option_type") == "PE")
                ).sort("datetime")

                if ce_series.is_empty() or pe_series.is_empty():
                    continue

                ce_times = ce_series["datetime"].to_list()
                ce_closes = ce_series["close"].to_numpy()
                pe_times = pe_series["datetime"].to_list()
                pe_closes = pe_series["close"].to_numpy()

                ce_idx = 0
                pe_idx = 0
                for si in day_indices:
                    bar_time = datetimes[si]
                    while ce_idx < len(ce_times) - 1 and ce_times[ce_idx + 1] <= bar_time:
                        ce_idx += 1
                    while pe_idx < len(pe_times) - 1 and pe_times[pe_idx + 1] <= bar_time:
                        pe_idx += 1

                    if (ce_idx < len(ce_times) and ce_times[ce_idx] <= bar_time and
                            pe_idx < len(pe_times) and pe_times[pe_idx] <= bar_time):
                        synth_future[si] = day_atm + ce_closes[ce_idx] - pe_closes[pe_idx]

        # Basis = synthetic future - spot
        basis = synth_future - close

        # 6-bar basis change
        basis_change = np.zeros(n)
        for i in range(6, n):
            if day_id[i] == day_id[i - 6] and synth_future[i] > 0 and synth_future[i - 6] > 0:
                basis_change[i] = basis[i] - basis[i - 6]

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed = np.arange(n) >= self.max_lookback

        # Basis expanding (futures premium increasing) → bullish → BUY CE
        buy_ce = in_session & warmed & (basis_change > basis_thresh) & (synth_future > 0)
        # Basis contracting (futures premium decreasing) → bearish → BUY PE
        buy_pe = in_session & warmed & (basis_change < -basis_thresh) & (synth_future > 0)

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
