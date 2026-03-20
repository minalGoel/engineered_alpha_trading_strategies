"""ORB Plus Sector (Index) Filter — Opus_46

Thesis: Opening Range Breakout confirmed by index ORB breakout in
the same direction, with VWAP and VIX filters. Target is 1x ORB range;
stop at ORB midpoint.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "opus_orb_plus_sector_filter_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 840     # 14:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("trailing_stop_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("trailing_activate_pct", default=0.004, low=0.003, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_max = params.get("vix_max", 22.0)
        trail_pct = params.get("trailing_stop_pct", 0.003)
        trail_act = params.get("trailing_activate_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        atr = _compute_atr(high, low, close, 20)

        # ── Compute ORB per day: stock and index (555-570 = first 15 minutes) ──
        orb_high = np.zeros(n, dtype=np.float64)
        orb_low = np.zeros(n, dtype=np.float64)
        orb_mid = np.zeros(n, dtype=np.float64)
        orb_range = np.zeros(n, dtype=np.float64)
        idx_orb_high = np.zeros(n, dtype=np.float64)
        idx_orb_low = np.zeros(n, dtype=np.float64)

        prev_day = -1
        day_hi = 0.0
        day_lo = 1e18
        idx_hi = 0.0
        idx_lo = 1e18
        orb_set = False

        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                day_hi = high[i]
                day_lo = low[i]
                idx_hi = index_close[i]
                idx_lo = index_close[i]
                orb_set = False

            if time_mins[i] <= 570:  # ORB period
                day_hi = max(day_hi, high[i])
                day_lo = min(day_lo, low[i])
                idx_hi = max(idx_hi, index_close[i])
                idx_lo = min(idx_lo, index_close[i])
            elif not orb_set:
                orb_set = True

            orb_high[i] = day_hi
            orb_low[i] = day_lo
            orb_mid[i] = (day_hi + day_lo) / 2.0
            orb_range[i] = day_hi - day_lo
            idx_orb_high[i] = idx_hi
            idx_orb_low[i] = idx_lo

        # Target: 1x ORB range as pct of close
        safe_close = np.where(close > 0, close, 1.0)
        target_pct_arr = orb_range / safe_close
        # Use median for scalar
        valid = target_pct_arr[target_pct_arr > 0]
        tgt_pct = float(np.median(valid)) if len(valid) > 0 else 0.004
        tgt_pct = np.clip(tgt_pct, 0.001, 0.01)

        # Stop: distance from entry to ORB midpoint as pct
        stop_dist = np.abs(close - orb_mid) / safe_close
        valid_stop = stop_dist[stop_dist > 0]
        stp_pct = float(np.median(valid_stop)) if len(valid_stop) > 0 else 0.003
        stp_pct = np.clip(stp_pct, 0.001, 0.008)

        vix_ok = vix < vix_max
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)
        after_orb = time_mins > 570

        # Long: close > stock ORB high AND index > index ORB high AND close > VWAP
        long_entry = ((close > orb_high) & (index_close > idx_orb_high) &
                      (close > vwap) & vix_ok & time_ok & after_orb)

        # Short: close < stock ORB low AND index < index ORB low
        short_entry = ((close < orb_low) & (index_close < idx_orb_low) &
                       (close < vwap) & vix_ok & time_ok & after_orb)

        # Signal exit: index crosses back inside its ORB
        exit_long = (index_close < idx_orb_high) & (index_close > idx_orb_low) & after_orb
        exit_short = (index_close > idx_orb_low) & (index_close < idx_orb_high) & after_orb

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=exit_long,
            signal_exit_short=exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stp_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=180,
        )
