"""MOC Drift — claude_project_15_of_25

Thesis: In the final hour of trading, stocks that are trading above VWAP
with positive afternoon returns tend to drift further into the close due
to market-on-close order flow imbalance.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "moc_drift_v1"
    is_long_only = False
    session_start = 870   # 14:30
    session_end = 925     # 15:25
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("afternoon_return_thresh", default=0.001, low=0.0005, high=0.003),
            TunableParam("vol_ratio_thresh", default=1.2, low=1.0, high=2.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        afternoon_return_thresh = params.get("afternoon_return_thresh", 0.001)
        vol_ratio_thresh = params.get("vol_ratio_thresh", 1.2)
        stop_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        volume = np.nan_to_num(volume, nan=1.0)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Compute close_at_14:00 (840 minutes) per day ──
        close_at_1400 = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        for d in unique_days:
            day_mask = day_ids == d
            day_indices = np.where(day_mask)[0]
            day_times = time_mins[day_indices]
            # Find bar closest to 14:00 (840 min)
            at_1400_mask = day_times <= 840
            if np.any(at_1400_mask):
                idx_1400 = day_indices[at_1400_mask][-1]
                close_at_1400[day_indices] = close[idx_1400]
            else:
                # Use first bar of the day if no bar before 14:00
                close_at_1400[day_indices] = close[day_indices[0]]

        # ── Afternoon return ──
        afternoon_return = np.where(
            close_at_1400 > 0,
            (close - close_at_1400) / close_at_1400,
            0.0,
        )

        # ── Volume ratio vs 20-bar avg ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        vol_ratio = volume / avg_vol_20

        # ── Entry window: 14:30-14:45 (870-885) ──
        entry_time_ok = (time_mins >= 870) & (time_mins <= 885)

        # ── Entry ──
        long_entry = (
            entry_time_ok
            & (close > vwap)
            & (afternoon_return > afternoon_return_thresh)
            & (vol_ratio > vol_ratio_thresh)
        )
        short_entry = (
            entry_time_ok
            & (close < vwap)
            & (afternoon_return < -afternoon_return_thresh)
            & (vol_ratio > vol_ratio_thresh)
        )

        # ── Exit: flatten at 15:25 is handled by session_end; use time_stop ──
        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=55,
        )
