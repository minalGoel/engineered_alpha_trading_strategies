"""Opening Range Breakout with volume/VIX/index filters — Grok_1_of_10

Thesis: Directional breakouts from the 15-min opening range caused by
institutional order flow. Target is 1.2x the opening range width.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "orb_breakout_v1"
    is_long_only = False
    session_start = 555   # 09:15 (need to capture OR)
    session_end = 915     # 15:15
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", default=22.0, low=14.0, high=30.0),
            TunableParam("rel_vol_thresh", default=1.8, low=1.0, high=3.0),
            TunableParam("target_mult", default=1.2, low=0.5, high=2.5),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.012),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_max = params.get("vix_max", 22.0)
        rv_thresh = params.get("rel_vol_thresh", 1.8)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        idx = df["index_close"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Relative volume
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        rel_vol = volume / avg_vol_20

        # Index 5-bar return
        idx = np.nan_to_num(idx, nan=0.0)
        idx_ret5 = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            if idx[i - 5] > 0:
                idx_ret5[i] = (idx[i] - idx[i - 5]) / idx[i - 5]

        # Opening range (first 15 bars: 555-569)
        or_high = np.full(n, np.nan, dtype=np.float64)
        or_low = np.full(n, np.nan, dtype=np.float64)
        for d in np.unique(day_ids):
            day_idx = np.where(day_ids == d)[0]
            or_mask = (time_mins[day_idx] >= 555) & (time_mins[day_idx] <= 569)
            or_bars = day_idx[or_mask]
            if len(or_bars) == 0:
                continue
            or_h = np.max(high[or_bars])
            or_l = np.min(low[or_bars])
            or_high[day_idx] = or_h
            or_low[day_idx] = or_l

        or_high = np.nan_to_num(or_high, nan=1e10)
        or_low = np.nan_to_num(or_low, nan=-1e10)

        # Filters
        after_or = time_mins >= 570
        vix_ok = vix < vix_max
        vol_ok = rel_vol > rv_thresh

        long_entry = (close > or_high) & after_or & vix_ok & vol_ok & (idx_ret5 > -0.0015)
        short_entry = (close < or_low) & after_or & vix_ok & vol_ok & (idx_ret5 < 0.0015)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=0.01,
            time_stop_bars=45,
        )
