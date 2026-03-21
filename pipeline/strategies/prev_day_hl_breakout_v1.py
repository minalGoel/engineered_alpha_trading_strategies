"""prev_day_hl_breakout_v1 — Previous Day High/Low Breakout on NIFTY

Mechanism: When NIFTY closes above its previous day high (PDH) for three consecutive
5-second bars, overnight short-sellers' stop orders have been swept, creating a
micro-liquidity vacuum. Breakout-scanning algorithms detect the confirmed level breach
and inject fresh buy orders, accelerating the index 15-30 spot points beyond PDH within
30-90 seconds. Symmetric logic applies at PDL for bearish breaks.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "prev_day_hl_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip pre-open noise
    session_end_minutes = 920     # 15:20 IST — avoid EOD illiquidity
    max_trades_per_day = 6
    max_lookback = 480            # 40 min warmup (240 bars for rel_vol + 240 buffer)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rel_vol_threshold", 1.5, 1.0, 3.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 15.0),
            TunableParam("vix_max", 24.0, 16.0, 32.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()

        rel_vol_threshold = params.get("rel_vol_threshold", 1.5)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)
        vix_max = params.get("vix_max", 24.0)

        # ── Previous day high / low ──────────────────────────────────────────
        # For each day_id D, find high/low of day D-1 and assign to all bars of day D.
        daily_hl = (
            spot_df
            .group_by("day_id")
            .agg([
                pl.col("high").max().alias("day_high"),
                pl.col("low").min().alias("day_low"),
            ])
            # Shift: the PDH/PDL of day D appears on bars of day D+1
            .with_columns((pl.col("day_id") + 1).alias("next_day_id"))
        )

        spot_with_pdh = spot_df.select(["day_id"]).join(
            daily_hl.select(["next_day_id", "day_high", "day_low"]),
            left_on="day_id",
            right_on="next_day_id",
            how="left",
        )
        # First day has no previous day data — fill with sentinel values so no signal fires
        pdh = spot_with_pdh["day_high"].fill_null(999_999.0).to_numpy()
        pdl = spot_with_pdh["day_low"].fill_null(0.0).to_numpy()

        # ── Session VWAP (cumulative from 09:15 each day) ───────────────────
        vwap_df = (
            spot_df
            .with_columns(
                ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical")
            )
            .with_columns(
                (pl.col("typical") * pl.col("volume").fill_null(0)).alias("tp_vol")
            )
            .with_columns([
                pl.col("tp_vol").cum_sum().over("day_id").alias("cum_tp_vol"),
                pl.col("volume").fill_null(0).cum_sum().over("day_id").alias("cum_vol"),
            ])
        )
        cum_tp_vol = vwap_df["cum_tp_vol"].fill_null(0.0).to_numpy().astype(float)
        cum_vol = vwap_df["cum_vol"].fill_null(0.0).to_numpy().astype(float)
        cum_vol_safe = np.where(cum_vol > 0, cum_vol, 1.0)
        vwap = cum_tp_vol / cum_vol_safe

        # ── Relative volume: rolling 240-bar (20 min) mean ──────────────────
        window = 240
        cs = np.zeros(n + 1)
        cs[1:] = np.cumsum(volume)
        vol_sma = np.zeros(n)
        for i in range(n):
            w = min(i, window)
            if w > 0:
                vol_sma[i] = (cs[i] - cs[i - w]) / w
            else:
                vol_sma[i] = 1.0
        rel_vol = np.where(vol_sma > 0, volume / vol_sma, 1.0)

        # ── VIX (aligned to spot bars) ───────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── 3-bar consecutive breakout confirmation ──────────────────────────
        above_pdh = close > pdh
        below_pdl = close < pdl

        consec_above = np.zeros(n, dtype=bool)
        consec_below = np.zeros(n, dtype=bool)
        for i in range(2, n):
            consec_above[i] = above_pdh[i] and above_pdh[i - 1] and above_pdh[i - 2]
            consec_below[i] = below_pdl[i] and below_pdl[i - 1] and below_pdl[i - 2]

        # ── Session filter ───────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry signals ────────────────────────────────────────────────────
        # Buy CE: confirmed PDH break + volume surge + price above VWAP + VIX calm
        buy_ce = (
            in_session
            & consec_above
            & (rel_vol > rel_vol_threshold)
            & (close > vwap)
            & (vix_close < vix_max)
        )

        # Buy PE: confirmed PDL break + volume surge + price below VWAP + VIX calm
        buy_pe = (
            in_session
            & consec_below
            & (rel_vol > rel_vol_threshold)
            & (close < vwap)
            & (vix_close < vix_max)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
