"""Gap Fade — GPT_9_of_10

Thesis: Large opening gaps (>1%) tend to fill intraday as initial momentum
retraces. Fade the gap direction: go long on gap-down, short on gap-up.
Entry only in first 10 minutes of the session.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "gap_fade_v1"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 915     # 15:15
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh", default=0.01, low=0.005, high=0.03),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.008),
            TunableParam("target_pct", default=0.005, low=0.002, high=0.015),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_thresh = params.get("gap_thresh", 0.01)
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_pct = params.get("target_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Compute opening gap per day ──
        # Gap = (today's first bar open - yesterday's last bar close) / yesterday's last bar close
        gap_pct = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)

        prev_close = np.nan
        for d in unique_days:
            day_mask = day_ids == d
            day_indices = np.where(day_mask)[0]
            if len(day_indices) == 0:
                continue
            day_open = open_[day_indices[0]]
            if not np.isnan(prev_close) and prev_close > 0:
                gap = (day_open - prev_close) / prev_close
                gap_pct[day_indices] = gap
            prev_close = close[day_indices[-1]]

        # ── Volume filter: volume > avg_5 ──
        avg_vol_5 = df["volume"].rolling_mean(5).to_numpy().astype(np.float64)
        avg_vol_5 = np.nan_to_num(avg_vol_5, nan=1.0)
        avg_vol_5 = np.clip(avg_vol_5, 1.0, None)
        vol_ok = volume > avg_vol_5

        # ── Time filter: entry only in first 10 minutes (09:15-09:24 → 555-564) ──
        time_ok = time_mins <= 564

        # ── Entry: fade the gap ──
        long_entry = (gap_pct < -gap_thresh) & time_ok & vol_ok
        short_entry = (gap_pct > gap_thresh) & time_ok & vol_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=30,
        )
