"""
pivot_camarilla_v1 — Camarilla Pivot Point Bounce & Breakout on NIFTY

Mechanism:
    Camarilla S3/R3 levels (prev_close ± prev_range × 0.275) are displayed by default
    on every major Indian retail terminal (Zerodha Kite, Dhan, Moneycontrol), creating
    genuine limit-order clustering on NIFTY at these levels. When NIFTY spot touches S3,
    resting buy-limit orders absorb selling pressure and produce a snap-back toward R3.
    At 5-second resolution we detect the precise touch-and-bounce without waiting a full
    minute. For R4/S4 breakouts, the breach signals institutional directional buying has
    overwhelmed all retail supply — momentum algorithms extend the move 20-50 spot points.
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "pivot_camarilla_v1"
    underlying = "NIFTY"
    session_start_minutes = 560    # 09:20 IST
    session_end_minutes = 925      # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 120             # 10 min warmup

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("s3_touch_pct", 0.0005, 0.0002, 0.002),
            TunableParam("r3_touch_pct", 0.0005, 0.0002, 0.002),
            TunableParam("vix_range_max", 22.0, 16.0, 28.0),
            TunableParam("stop_pts_range", 4.0, 2.0, 8.0),
            TunableParam("target_pts_range", 6.0, 4.0, 12.0),
            TunableParam("stop_pts_breakout", 5.0, 3.0, 10.0),
            TunableParam("target_pts_breakout", 8.0, 5.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        s3_touch_pct = params.get("s3_touch_pct", 0.0005)
        r3_touch_pct = params.get("r3_touch_pct", 0.0005)
        vix_range_max = params.get("vix_range_max", 22.0)
        stop_pts_range = params.get("stop_pts_range", 4.0)
        target_pts_range = params.get("target_pts_range", 6.0)
        stop_pts_breakout = params.get("stop_pts_breakout", 5.0)
        target_pts_breakout = params.get("target_pts_breakout", 8.0)

        # ── 1. Compute previous day OHLC per session_date ──
        daily = (
            spot_df
            .group_by("session_date")
            .agg([
                pl.col("high").max().alias("d_high"),
                pl.col("low").min().alias("d_low"),
                pl.col("close").last().alias("d_close"),
            ])
            .sort("session_date")
        )

        # Shift to get PREVIOUS day values
        daily = daily.with_columns([
            pl.col("d_high").shift(1).alias("prev_high"),
            pl.col("d_low").shift(1).alias("prev_low"),
            pl.col("d_close").shift(1).alias("prev_close"),
        ])

        # Camarilla level formulas
        daily = daily.with_columns([
            (pl.col("prev_close") + (pl.col("prev_high") - pl.col("prev_low")) * (1.1 / 4.0)).alias("cam_R3"),
            (pl.col("prev_close") + (pl.col("prev_high") - pl.col("prev_low")) * (1.1 / 2.0)).alias("cam_R4"),
            (pl.col("prev_close") - (pl.col("prev_high") - pl.col("prev_low")) * (1.1 / 4.0)).alias("cam_S3"),
            (pl.col("prev_close") - (pl.col("prev_high") - pl.col("prev_low")) * (1.1 / 2.0)).alias("cam_S4"),
        ])

        # Join daily Camarilla levels back to spot_df on session_date
        spot_with_levels = spot_df.join(
            daily.select(["session_date", "cam_R3", "cam_R4", "cam_S3", "cam_S4"]),
            on="session_date",
            how="left",
        )

        # Forward-fill NaN (first trading day in data has no previous day)
        spot_with_levels = spot_with_levels.with_columns([
            pl.col("cam_R3").forward_fill(),
            pl.col("cam_R4").forward_fill(),
            pl.col("cam_S3").forward_fill(),
            pl.col("cam_S4").forward_fill(),
        ])

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        cam_R3 = spot_with_levels["cam_R3"].fill_null(0.0).to_numpy()
        cam_R4 = spot_with_levels["cam_R4"].fill_null(0.0).to_numpy()
        cam_S3 = spot_with_levels["cam_S3"].fill_null(0.0).to_numpy()
        cam_S4 = spot_with_levels["cam_S4"].fill_null(0.0).to_numpy()

        # ── 2. VIX ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── 3. Session and validity filters ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        levels_valid = (cam_S3 > 0.0) & (cam_R3 > 0.0)

        # ── 4. Touch detection: did price reach within touch_pct of S3/R3 in last 3 bars? ──
        touched_s3 = np.zeros(n, dtype=bool)
        touched_r3 = np.zeros(n, dtype=bool)
        for i in range(3, n):
            if cam_S3[i] > 0.0:
                min_low_3 = np.min(low[i - 3:i])
                touched_s3[i] = min_low_3 <= cam_S3[i] * (1.0 + s3_touch_pct)
            if cam_R3[i] > 0.0:
                max_high_3 = np.max(high[i - 3:i])
                touched_r3[i] = max_high_3 >= cam_R3[i] * (1.0 - r3_touch_pct)

        # ── 5. Entry signals ──
        # S3 bounce → buy CE (range trade): low touched S3, close reclaimed above S3
        buy_ce_range = (
            in_session
            & levels_valid
            & touched_s3
            & (close > cam_S3)       # bounced above S3 on this bar
            & (close > cam_S4)       # not in S4 breakdown territory
            & (vix_close < vix_range_max)
        )

        # R3 rejection → buy PE (range trade): high touched R3, close rejected below R3
        buy_pe_range = (
            in_session
            & levels_valid
            & touched_r3
            & (close < cam_R3)       # rejected below R3 on this bar
            & (close < cam_R4)       # not in R4 breakout territory
            & (vix_close < vix_range_max)
        )

        # R4 breakout → buy CE (trend day): close above R4
        buy_ce_breakout = (
            in_session
            & levels_valid
            & (close > cam_R4)
        )

        # S4 breakdown → buy PE (trend day): close below S4
        buy_pe_breakout = (
            in_session
            & levels_valid
            & (close < cam_S4)
        )

        buy_ce = buy_ce_range | buy_ce_breakout
        buy_pe = buy_pe_range | buy_pe_breakout

        # Mutual exclusion: if both fire simultaneously, suppress both
        both = buy_ce & buy_pe
        buy_ce = buy_ce & ~both
        buy_pe = buy_pe & ~both

        # ── 6. Per-bar stop / target (range vs breakout calibration) ──
        is_range_signal = (buy_ce_range | buy_pe_range) & ~both
        stop_points = np.where(is_range_signal, stop_pts_range, stop_pts_breakout).astype(np.float64)
        target_points = np.where(is_range_signal, target_pts_range, target_pts_breakout).astype(np.float64)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_points,
            target_points=target_points,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,         # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
