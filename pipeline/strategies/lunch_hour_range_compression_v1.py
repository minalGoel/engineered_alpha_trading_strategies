"""Lunch Hour Range Compression — NIFTY 5-second options strategy.

During 12:00-13:30, NIFTY consolidates in a compressed range as institutional flows pause.
At 13:30, MF NAV-linked and FII afternoon orders re-enter simultaneously, breaking the range
with directional momentum that persists for 30-90 seconds. We trade the initial burst.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "lunch_hour_range_compression_v1"
    underlying = "NIFTY"
    session_start_minutes = 810   # 13:30 IST — only trade post-lunch
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 3
    max_lookback = 1080           # 90 min warmup — needs full lunch window before firing

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("compression_threshold", 0.003, 0.001, 0.006),
            TunableParam("volume_multiplier", 1.2, 0.8, 2.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()

        compression_threshold = params.get("compression_threshold", 0.003)
        volume_multiplier = params.get("volume_multiplier", 1.2)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── Lunch range: MAX(high) and MIN(low) from 12:00-13:30 per day ──
        # time_minutes 720=12:00, 810=13:30 (exclusive)
        lunch_agg = (
            spot_df
            .filter(
                (pl.col("time_minutes") >= 720) & (pl.col("time_minutes") < 810)
            )
            .group_by("day_id")
            .agg([
                pl.col("high").max().alias("lunch_h"),
                pl.col("low").min().alias("lunch_l"),
            ])
        )

        # ── Morning direction: sign(close@12:00 - open@09:15) ──
        # Use last bar at time_minutes==720 as the 12:00 close
        noon_df = (
            spot_df
            .filter(pl.col("time_minutes") == 720)
            .group_by("day_id")
            .agg(pl.col("close").last().alias("noon_close"))
        )
        # Use first bar open as day open proxy (09:15 first 5s bar)
        day_open_df = (
            spot_df
            .group_by("day_id")
            .agg(pl.col("open").first().alias("day_open_price"))
        )
        morning_df = noon_df.join(day_open_df, on="day_id", how="left").with_columns(
            pl.when(pl.col("noon_close") > pl.col("day_open_price"))
            .then(pl.lit(1))
            .when(pl.col("noon_close") < pl.col("day_open_price"))
            .then(pl.lit(-1))
            .otherwise(pl.lit(0))
            .cast(pl.Int32)
            .alias("mdir")
        )

        # ── Join per-day aggregates back to every bar ──
        spot_aug = (
            spot_df
            .select(["datetime", "day_id"])
            .join(lunch_agg, on="day_id", how="left")
            .join(morning_df.select(["day_id", "mdir"]), on="day_id", how="left")
        )

        lunch_h = spot_aug["lunch_h"].fill_null(0.0).to_numpy()
        lunch_l = spot_aug["lunch_l"].fill_null(0.0).to_numpy()
        mdir = spot_aug["mdir"].fill_null(0).to_numpy()

        # Valid lunch range: both bounds positive and high > low
        lunch_valid = (lunch_h > 0.0) & (lunch_l > 0.0) & (lunch_h > lunch_l)
        lunch_range = np.where(lunch_valid, lunch_h - lunch_l, 0.0)

        # ── 60-bar (5-min) rolling volume SMA for breakout volume filter ──
        vol_sma = np.zeros(n)
        for i in range(60, n):
            vol_sma[i] = np.mean(volume[i - 60:i])
        # Bootstrap first 60 bars with their own value so filter is non-zero
        vol_sma[:60] = np.where(volume[:60] > 0, volume[:60], 1.0)

        # ── Session filter: 13:30-15:25 only ──
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Compression filter: lunch range < threshold * spot price ──
        compressed = (
            lunch_valid
            & (lunch_range > 0.0)
            & (lunch_range < compression_threshold * close)
        )

        # ── Volume surge at breakout ──
        vol_ok = volume > (vol_sma * volume_multiplier)

        # ── Directional breakouts ──
        breaks_up = lunch_valid & (close > lunch_h)
        breaks_down = lunch_valid & (close < lunch_l)

        # Buy CE: bullish breakout, morning was bullish, compressed, volume surge
        buy_ce = in_session & compressed & breaks_up & (mdir > 0) & vol_ok

        # Buy PE: bearish breakout, morning was bearish, compressed, volume surge
        buy_pe = in_session & compressed & breaks_down & (mdir < 0) & vol_ok

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds = 18 × 5s bars
            max_trades_per_day=self.max_trades_per_day,
        )
