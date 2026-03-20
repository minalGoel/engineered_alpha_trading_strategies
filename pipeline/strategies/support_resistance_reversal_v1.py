"""
support_resistance_reversal_v1 — Previous Day High/Low Reversal on NIFTY

Thesis: NIFTY's previous day high and low are the densest institutional limit-order
clusters in the market. When price probes these levels, option writers, delta-hedgers,
and stop-reversal algos create a supply/demand wall. At 5-second resolution, range
contraction + a visible 15-second rejection candle signals the inflection point before
the full bounce is confirmed.

Original: support_resistance_reversal_v1 (1-min equity, nifty200 stocks, HL reversal pattern)
Conversion: NIFTY index, 5s bars, prev day H/L as S/R, range contraction + ret_3 as trigger
"""
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam
import numpy as np
import polars as pl


class Strategy(BaseStrategy):
    name = "support_resistance_reversal_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 72             # 6 minutes: 12 bars for range comparison + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("sr_threshold_pct", 0.0015, 0.0005, 0.004),  # % proximity to S/R
            TunableParam("ret3_threshold", 0.0002, 0.00005, 0.0006),   # 15s return for reversal candle
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
            TunableParam("vix_max", 20.0, 15.0, 28.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        sr_threshold = params.get("sr_threshold_pct", 0.0015)
        ret3_threshold = params.get("ret3_threshold", 0.0002)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)
        vix_max = params.get("vix_max", 20.0)

        # --- VIX filter ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- Previous day high/low ---
        # Compute daily H/L per day_id, then apply to the NEXT day
        daily_hl = (
            spot_df.group_by("day_id")
            .agg([
                pl.col("high").max().alias("day_high"),
                pl.col("low").min().alias("day_low"),
            ])
            .sort("day_id")
            .with_columns(
                (pl.col("day_id") + 1).alias("next_day_id")
            )
            .select([
                pl.col("next_day_id").alias("day_id"),
                pl.col("day_high").alias("prev_day_high"),
                pl.col("day_low").alias("prev_day_low"),
            ])
        )

        spot_with_sr = spot_df.select(["day_id"]).join(daily_hl, on="day_id", how="left")
        prev_day_high = (
            spot_with_sr["prev_day_high"]
            .fill_null(strategy="forward")
            .fill_null(0.0)
            .to_numpy()
        )
        prev_day_low = (
            spot_with_sr["prev_day_low"]
            .fill_null(strategy="forward")
            .fill_null(0.0)
            .to_numpy()
        )

        # --- Bar range ---
        bar_range = high - low

        # --- Range contraction: mean range last 6 bars vs prior 6 bars ---
        range_6 = np.zeros(n)
        range_prior_6 = np.zeros(n)
        for i in range(12, n):
            range_6[i] = np.mean(bar_range[i - 6:i])
            range_prior_6[i] = np.mean(bar_range[i - 12:i - 6])

        # --- 15-second return (3 bars) for initial reversal confirmation ---
        ret_3 = np.zeros(n)
        for i in range(3, n):
            prev_close = close[i - 3]
            if prev_close > 0:
                ret_3[i] = (close[i] - prev_close) / prev_close

        # --- Session filter ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # --- VIX regime filter ---
        low_vix = vix_close < vix_max

        # --- Range contraction (limit order accumulation at S/R) ---
        range_contracting = (range_prior_6 > 0) & (range_6 < range_prior_6)

        # --- Proximity to S/R level ---
        # Near support (prev day low): buy CE on bounce
        has_prev_low = prev_day_low > 0
        near_support = has_prev_low & (
            np.abs(close - prev_day_low) / np.where(has_prev_low, prev_day_low, 1.0) < sr_threshold
        )

        # Near resistance (prev day high): buy PE on rejection
        has_prev_high = prev_day_high > 0
        near_resistance = has_prev_high & (
            np.abs(close - prev_day_high) / np.where(has_prev_high, prev_day_high, 1.0) < sr_threshold
        )

        # --- Entry signals ---
        # Buy CE: price at prev day low + range contracting + first bounce bar
        buy_ce = (
            in_session
            & low_vix
            & near_support
            & range_contracting
            & (ret_3 > ret3_threshold)
        )

        # Buy PE: price at prev day high + range contracting + first rejection bar
        buy_pe = (
            in_session
            & low_vix
            & near_resistance
            & range_contracting
            & (ret_3 < -ret3_threshold)
        )

        # Mutual exclusion: if both fire (very rare — price near both levels), skip
        both = buy_ce & buy_pe
        buy_ce = buy_ce & ~both
        buy_pe = buy_pe & ~both

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
