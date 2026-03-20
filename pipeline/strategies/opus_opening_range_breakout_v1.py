"""Opening Range Breakout (15-min) — Opus_9

Thesis: The 15-minute opening range (09:15-09:30) defines a key
support/resistance zone. Breakouts above/below with VWAP confirmation
and sufficient range width produce trending moves.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "opus_opening_range_breakout_v1"
    is_long_only = False
    session_start = 570   # 09:30 (after ORB forms)
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("min_orb_range_bps", default=30.0, low=15.0, high=60.0),
            TunableParam("rel_vol_thresh", default=1.0, low=0.5, high=2.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("trailing_stop_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_activate_pct", default=0.004, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        min_orb_bps = params.get("min_orb_range_bps", 30.0)
        rel_vol_thresh = params.get("rel_vol_thresh", 1.0)
        vix_max = params.get("vix_max", 22.0)
        trailing_pct = params.get("trailing_stop_pct", 0.003)
        trailing_act = params.get("trailing_activate_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Compute ORB (09:15-09:30, time_minutes 555-569) per day ──
        orb_high = np.zeros(n, dtype=np.float64)
        orb_low = np.zeros(n, dtype=np.float64)
        orb_mid = np.zeros(n, dtype=np.float64)
        orb_range_bps = np.zeros(n, dtype=np.float64)

        unique_days = np.unique(day_ids)
        for d in unique_days:
            day_mask = day_ids == d
            day_indices = np.where(day_mask)[0]
            if len(day_indices) == 0:
                continue
            # ORB bars: time_minutes 555 to 569
            orb_mask = (time_mins[day_indices] >= 555) & (time_mins[day_indices] <= 569)
            orb_indices = day_indices[orb_mask]
            if len(orb_indices) == 0:
                continue
            oh = np.max(high[orb_indices])
            ol = np.min(low[orb_indices])
            om = (oh + ol) / 2.0
            safe_om = max(om, 1e-10)
            orb_bps = (oh - ol) / safe_om * 10000.0

            orb_high[day_indices] = oh
            orb_low[day_indices] = ol
            orb_mid[day_indices] = om
            orb_range_bps[day_indices] = orb_bps

        # ── Relative volume ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        rel_vol = volume / avg_vol_20

        # ── Filters ──
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 570) & (time_mins <= 870)
        range_ok = orb_range_bps >= min_orb_bps
        vol_ok = rel_vol > rel_vol_thresh

        # ── Entries ──
        long_entry = ((close > orb_high) & (close > vwap) & range_ok
                      & vol_ok & vix_ok & time_ok)
        short_entry = ((close < orb_low) & (close < vwap) & range_ok
                       & vol_ok & vix_ok & time_ok)

        # ── Target: 1x ORB range as target_indicator ──
        # For longs: orb_high + orb_range, for shorts: orb_low - orb_range
        # Use target as pct approximation since target_indicator is single array
        # Use orb_range as fraction of price for target_pct
        orb_range_frac = np.zeros(n, dtype=np.float64)
        safe_close = np.clip(np.abs(close), 1e-10, None)
        orb_range_abs = orb_high - orb_low
        orb_range_frac = orb_range_abs / safe_close
        # Use median as scalar target_pct
        valid_frac = orb_range_frac[orb_range_frac > 0]
        target_pct = float(np.median(valid_frac)) if len(valid_frac) > 0 else 0.004

        # ── Stop at ORB midpoint: use stop_loss_pct as half the range ──
        stop_pct = target_pct / 2.0 if target_pct > 0 else 0.002

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=trailing_pct,
            trailing_activate_pct=trailing_act,
            time_stop_bars=300,
        )
