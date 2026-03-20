"""
previous_day_hl_breakout_v1 — PDH/PDL Breakout on NIFTY Index

When NIFTY breaks above its previous session high with a volume surge (>2.5x 5-min average)
and 3-bar (15s) confirmation, institutional buy-side flow and delta-hedger short-covering
near PDH strike clusters drive an 8-15 spot point impulse in the breakout direction.
The 3-bar confirmation filter distinguishes sustained institutional breakouts from stop-hunt
spikes that reverse within 1-2 bars.
"""

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "previous_day_hl_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip opening noise
    session_end_minutes = 900     # 15:00 IST — avoid late session
    max_lookback = 240            # 20 min warmup
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_ratio_threshold", 2.5, 1.5, 4.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        vol_threshold = params.get("vol_ratio_threshold", 2.5)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        time_min = spot_df["time_minutes"].to_numpy()

        # ── Compute PDH/PDL from previous session ──────────────────────────
        # Aggregate daily high/low, shift by 1 day to get previous day's values
        daily_hl = (
            spot_df
            .group_by("day_id")
            .agg([
                pl.col("high").max().alias("day_high"),
                pl.col("low").min().alias("day_low"),
            ])
            .sort("day_id")
            .with_columns([
                pl.col("day_high").shift(1).alias("pdh"),
                pl.col("day_low").shift(1).alias("pdl"),
            ])
            .select(["day_id", "pdh", "pdl"])
        )

        # Join PDH/PDL back; first day has no previous day → fill with extremes (no signal fires)
        spot_ext = spot_df.join(daily_hl, on="day_id", how="left")
        pdh = spot_ext["pdh"].fill_null(1e9).to_numpy()
        pdl = spot_ext["pdl"].fill_null(-1e9).to_numpy()

        # ── VWAP from session open (cumulative, per day) ────────────────────
        spot_ext = spot_ext.with_columns([
            (pl.col("close") * pl.col("volume").cast(pl.Float64)).alias("_cv")
        ]).with_columns([
            (
                pl.col("_cv").cum_sum().over("day_id")
                / (pl.col("volume").cast(pl.Float64).cum_sum().over("day_id") + 1e-9)
            ).alias("vwap")
        ])
        vwap = spot_ext["vwap"].fill_null(strategy="forward").to_numpy()
        close = spot_ext["close"].to_numpy()
        vwap = np.where(np.isnan(vwap), close, vwap)

        # ── Rolling 60-bar (5-min) volume average for ratio ─────────────────
        volume = spot_df["volume"].to_numpy().astype(np.float64)
        vol_mean = np.empty(n)
        vol_mean[0] = volume[0] if volume[0] > 0 else 1.0
        for i in range(1, n):
            lo = max(0, i - 60)
            vol_mean[i] = np.mean(volume[lo:i]) if i > lo else volume[0]
        vol_mean = np.where(vol_mean <= 0, 1.0, vol_mean)
        volume_ratio = volume / vol_mean

        # ── Session + time filter: 09:30-13:00 IST ─────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        time_ok = in_session & (time_min < 780)  # before 13:00 — afternoon breaks have low follow-through

        # ── Boolean level masks ─────────────────────────────────────────────
        above_pdh = close > pdh
        below_pdl = close < pdl

        # ── 3-bar confirmation entry logic ──────────────────────────────────
        # Entry at bar i when: bars i, i-1, i-2 all confirm break AND bar i-3 did NOT
        # Volume surge required on the initial break bar (i-2)
        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        for i in range(3, n):
            if not time_ok[i]:
                continue

            # Bullish: 3 consecutive closes above PDH, fresh break, volume + VWAP confirmed
            if (above_pdh[i] and above_pdh[i - 1] and above_pdh[i - 2]
                    and not above_pdh[i - 3]
                    and volume_ratio[i - 2] > vol_threshold
                    and close[i] > vwap[i]):
                buy_ce[i] = True

            # Bearish: 3 consecutive closes below PDL, fresh break, volume + VWAP confirmed
            if (below_pdl[i] and below_pdl[i - 1] and below_pdl[i - 2]
                    and not below_pdl[i - 3]
                    and volume_ratio[i - 2] > vol_threshold
                    and close[i] < vwap[i]):
                buy_pe[i] = True

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds — capture the initial impulse
            max_trades_per_day=self.max_trades_per_day,
        )
