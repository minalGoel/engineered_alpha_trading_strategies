"""orb_plus_sector_filter_v1 — NIFTY ORB confirmed by VIX ORB (dual-index sector confirmation).

Original: Stock ORB + sector-index ORB confirmation (Strategy_63.json).
Adaptation: India VIX acts as the 'sector index' — VIX breaking its own ORB in the confirming
direction signals broad multi-sector institutional flow backing the NIFTY breakout.
- Buy CE: NIFTY > ORB high AND VIX < VIX ORB low (fear dropping = institutional buying)
- Buy PE: NIFTY < ORB low AND VIX > VIX ORB high (fear rising = institutional selling)
Session: 09:35-14:00 IST (entry after 20-min ORB fully formed).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_orb_levels(
    high: np.ndarray,
    low: np.ndarray,
    time_min: np.ndarray,
    day_id: np.ndarray,
    orb_start: int = 555,
    orb_end: int = 575,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute opening range high/low per day; propagate values to post-ORB bars only."""
    n = len(high)
    orb_high = np.full(n, np.nan)
    orb_low = np.full(n, np.nan)
    for d in np.unique(day_id):
        day_mask = day_id == d
        orb_mask = day_mask & (time_min >= orb_start) & (time_min < orb_end)
        orb_idx = np.where(orb_mask)[0]
        if len(orb_idx) == 0:
            continue
        h_vals = high[orb_idx]
        l_vals = low[orb_idx]
        h_vals = h_vals[~np.isnan(h_vals)]
        l_vals = l_vals[~np.isnan(l_vals)]
        if len(h_vals) == 0 or len(l_vals) == 0:
            continue
        day_orb_high = np.max(h_vals)
        day_orb_low = np.min(l_vals)
        post_idx = np.where(day_mask & (time_min >= orb_end))[0]
        orb_high[post_idx] = day_orb_high
        orb_low[post_idx] = day_orb_low
    return orb_high, orb_low


def _compute_vwap_series(spot_df: pl.DataFrame) -> np.ndarray:
    """Cumulative session VWAP per day using Polars window functions."""
    df = (
        spot_df
        .with_columns(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_tp")
        )
        .with_columns(
            (pl.col("_tp") * pl.col("volume").cast(pl.Float64)).alias("_tpv")
        )
        .with_columns([
            pl.col("_tpv").cum_sum().over("day_id").alias("_cum_tpv"),
            pl.col("volume").cast(pl.Float64).cum_sum().over("day_id").alias("_cum_vol"),
        ])
        .with_columns(
            (
                pl.col("_cum_tpv")
                / pl.when(pl.col("_cum_vol") > 0)
                .then(pl.col("_cum_vol"))
                .otherwise(1.0)
            ).alias("_vwap")
        )
    )
    return df["_vwap"].fill_null(strategy="forward").to_numpy()


class Strategy(BaseStrategy):
    name = "orb_plus_sector_filter_v1"
    underlying = "NIFTY"
    session_start_minutes = 575   # 09:35 IST — entry after 20-min ORB fully formed
    session_end_minutes = 840     # 14:00 IST — original strategy's cutoff
    max_trades_per_day = 6
    max_lookback = 300            # 25 min warmup (covers ORB window + buffer)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_level_cap", 25.0, 16.0, 32.0),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)
        vix_level_cap = params.get("vix_level_cap", 25.0)

        # ── NIFTY ORB levels (09:15-09:34 = time_min 555–574) ──
        nifty_orb_high, nifty_orb_low = _compute_orb_levels(
            high, low, time_min, day_id, orb_start=555, orb_end=575
        )

        # ── Session VWAP ──
        vwap = _compute_vwap_series(spot_df)

        # ── VIX data: aligned close + ORB levels ──
        vix_close = np.full(n, 15.0)
        vix_orb_high = np.full(n, np.nan)
        vix_orb_low = np.full(n, np.nan)

        if vix_df is not None and not vix_df.is_empty():
            vix_aligned = spot_df.select("datetime").join_asof(
                vix_df.select([
                    "datetime",
                    pl.col("close").alias("vix_close"),
                    pl.col("high").alias("vix_h"),
                    pl.col("low").alias("vix_l"),
                ]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_aligned["vix_close"].fill_null(15.0).to_numpy()
            vix_h = vix_aligned["vix_h"].fill_null(strategy="forward").to_numpy()
            vix_l = vix_aligned["vix_l"].fill_null(strategy="forward").to_numpy()

            # VIX ORB per day (same 09:15-09:34 window)
            for d in np.unique(day_id):
                day_mask = day_id == d
                orb_mask = day_mask & (time_min >= 555) & (time_min < 575)
                orb_idx = np.where(orb_mask)[0]
                if len(orb_idx) == 0:
                    continue
                vh_vals = vix_h[orb_idx]
                vl_vals = vix_l[orb_idx]
                vh_vals = vh_vals[~np.isnan(vh_vals)]
                vl_vals = vl_vals[~np.isnan(vl_vals)]
                if len(vh_vals) == 0 or len(vl_vals) == 0:
                    continue
                vh = np.max(vh_vals)
                vl = np.min(vl_vals)
                post_idx = np.where(day_mask & (time_min >= 575))[0]
                vix_orb_high[post_idx] = vh
                vix_orb_low[post_idx] = vl

        # ── Filters ──
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        valid_nifty_orb = ~np.isnan(nifty_orb_high) & ~np.isnan(nifty_orb_low)
        valid_vix_orb = ~np.isnan(vix_orb_high) & ~np.isnan(vix_orb_low)
        vix_cap = vix_close < vix_level_cap

        # ── Entry signals: dual ORB confirmation ──
        # Bullish: NIFTY breaks ORB high AND VIX breaks ORB low (fear drops = broad participation)
        buy_ce = (
            in_session
            & valid_nifty_orb
            & valid_vix_orb
            & vix_cap
            & (close > nifty_orb_high)
            & (vix_close < vix_orb_low)
            & (close > vwap)
        )

        # Bearish: NIFTY breaks ORB low AND VIX breaks ORB high (fear rises = broad selling)
        buy_pe = (
            in_session
            & valid_nifty_orb
            & valid_vix_orb
            & vix_cap
            & (close < nifty_orb_low)
            & (vix_close > vix_orb_high)
            & (close < vwap)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,            # 90 seconds — ORB continuation should move fast
            max_trades_per_day=self.max_trades_per_day,
        )
