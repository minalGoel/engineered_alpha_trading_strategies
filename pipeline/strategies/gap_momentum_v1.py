"""gap_momentum_v1 — NIFTY opening gap continuation strategy.

When NIFTY gaps up/down at open (driven by overnight SGX Nifty futures or global
equity moves), institutional TWAP/VWAP algos with pre-positioned directional orders
continue pushing the index in the gap direction for the first 30-90 seconds. A
3-bar (15s) momentum confirmation filters gap-and-fade openings. We ride the next
10-15 spot points of continuation then exit.

Session: 09:15-09:35 IST only (gap continuation window).
Hold: 15-90 seconds (time_stop_bars=18 at 5s/bar = 90s).
"""

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "gap_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 555   # 09:15 IST — capture gap open from first bar
    session_end_minutes = 575     # 09:35 IST — gap window only (first 20 min)
    max_trades_per_day = 3
    max_lookback = 120            # 10 min warmup; gap itself is a daily event

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_threshold", 0.004, 0.002, 0.010),
            TunableParam("vix_max", 20.0, 15.0, 30.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 14.0),
        ]

    def _compute_gap_pct(self, spot_df: pl.DataFrame) -> np.ndarray:
        """Compute overnight gap_pct per bar.

        gap_pct = (first_bar_open_today - last_close_yesterday) / last_close_yesterday

        Bars on day 0 (first day in df) get gap_pct = 0.0 (no prior close available).
        """
        # Get first open and last close per day (sort ensures correct first/last)
        day_stats = (
            spot_df
            .sort(["day_id", "time_minutes"])
            .group_by("day_id")
            .agg([
                pl.col("open").first().alias("day_open"),
                pl.col("close").last().alias("day_last_close"),
            ])
            .sort("day_id")
        )

        # Shift last_close by 1 to get previous day's close aligned to each day
        day_with_prev = day_stats.with_columns(
            pl.col("day_last_close").shift(1).alias("prev_close")
        ).with_columns(
            pl.when(
                pl.col("prev_close").is_not_null() & (pl.col("prev_close") > 0)
            )
            .then(
                (pl.col("day_open") - pl.col("prev_close")) / pl.col("prev_close")
            )
            .otherwise(0.0)
            .alias("gap_pct")
        ).select(["day_id", "gap_pct"])

        # Join gap_pct back to every bar in spot_df
        result = spot_df.select("day_id").join(
            day_with_prev, on="day_id", how="left"
        )
        return result["gap_pct"].fill_null(0.0).to_numpy()

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill spot close before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        gap_threshold = params.get("gap_threshold", 0.004)
        vix_max = params.get("vix_max", 20.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 8.0)

        # ── VIX close (aligned to spot bars) ──────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")])
                .sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Overnight gap percentage (per bar, based on day_id) ───────────────
        gap_pct = self._compute_gap_pct(spot_df)

        # ── 3-bar (15-second) momentum confirmation ───────────────────────────
        # ret_3[i] = (close[i] - close[i-3]) / close[i-3]
        # Confirms price is continuing in gap direction (not immediately fading).
        safe_close = np.where(close > 0, close, 1.0)
        ret_3 = np.zeros(n)
        ret_3[3:] = (close[3:] - close[:-3]) / safe_close[:-3]

        # ── Session filter: 09:15-09:35 only ─────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes) &
            (time_min < self.session_end_minutes)
        )

        # ── Regime filter ─────────────────────────────────────────────────────
        low_vix = vix_close < vix_max

        # ── Gap direction ─────────────────────────────────────────────────────
        gap_up = gap_pct > gap_threshold
        gap_down = gap_pct < -gap_threshold

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: gap-up + 15s continuation + low VIX + in gap window
        buy_ce = in_session & gap_up & (ret_3 > 0) & low_vix

        # buy_pe: gap-down + 15s continuation + low VIX + in gap window
        buy_pe = in_session & gap_down & (ret_3 < 0) & low_vix

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,             # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
