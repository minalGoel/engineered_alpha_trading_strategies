"""Nifty Intraday Momentum Factor — claude_project_18_of_25

Thesis: Stocks with strong morning returns (measured from open to
late-morning) tend to continue in the same direction for the rest
of the day, especially when confirmed by above-average volume.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "nifty_intraday_momentum_factor_v1"
    is_long_only = False
    session_start = 645   # 10:45
    session_end = 870     # 14:30
    max_trades_per_day = 12

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("morning_return_thresh", default=0.003, low=0.001, high=0.006),
            TunableParam("target_pct", default=0.005, low=0.002, high=0.01),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.008),
            TunableParam("breakeven_pct", default=0.0025, low=0.001, high=0.005),
            TunableParam("vix_max", default=24.0, low=18.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        morning_return_thresh = params.get("morning_return_thresh", 0.003)
        target_pct = params.get("target_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.004)
        breakeven_pct = params.get("breakeven_pct", 0.0025)
        vix_max = params.get("vix_max", 24.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        vix = df["vix"].to_numpy().astype(np.float64)

        volume = np.nan_to_num(volume, nan=1.0)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── Compute morning return per day: close at 10:45 vs open at 09:15 ──
        morning_return = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        for d in unique_days:
            day_mask = day_ids == d
            day_indices = np.where(day_mask)[0]
            if len(day_indices) == 0:
                continue
            day_times = time_mins[day_indices]
            # Day open (first bar, ~09:15 = 555)
            day_open = open_[day_indices[0]]
            # Find close at 10:45 (645) or closest before
            before_1045 = day_times <= 645
            if np.any(before_1045):
                idx_1045 = day_indices[before_1045][-1]
                close_1045 = close[idx_1045]
            else:
                close_1045 = day_open
            if day_open > 0:
                ret = (close_1045 - day_open) / day_open
                morning_return[day_indices] = ret

        # ── Volume filter: volume > 20-bar avg ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        vol_ok = volume > avg_vol_20

        # ── VIX filter ──
        vix_ok = vix < vix_max

        # ── Entry ──
        long_entry = (morning_return > morning_return_thresh) & vol_ok & vix_ok
        short_entry = (morning_return < -morning_return_thresh) & vol_ok & vix_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            breakeven_pct=breakeven_pct,
            time_stop_bars=240,
        )
