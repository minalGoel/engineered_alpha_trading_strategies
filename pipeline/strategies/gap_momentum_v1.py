"""Gap Momentum — Grok_2_of_10

Thesis: Large gaps (>2%) driven by overnight news tend to continue in direction
confirmed by the first 1-min candle. Unlike gap-fade, this RIDES the gap.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "gap_momentum_v1"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 915     # 15:15
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh", default=0.02, low=0.01, high=0.05),
            TunableParam("vix_max", default=20.0, low=12.0, high=30.0),
            TunableParam("rel_vol_thresh", default=1.3, low=1.0, high=2.5),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.015),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_thresh = params.get("gap_thresh", 0.02)
        vix_max = params.get("vix_max", 20.0)
        rv_thresh = params.get("rel_vol_thresh", 1.3)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Relative volume
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        rel_vol = volume / avg_vol_20

        # Gap % and signal candle direction per day
        gap_pct = np.zeros(n, dtype=np.float64)
        signal_bullish = np.zeros(n, dtype=np.bool_)

        prev_close = np.nan
        for d in np.unique(day_ids):
            day_idx = np.where(day_ids == d)[0]
            if len(day_idx) == 0:
                continue
            day_open = open_[day_idx[0]]
            # Signal candle: first bar close > open = bullish
            first_bar_bullish = close[day_idx[0]] > open_[day_idx[0]]

            if not np.isnan(prev_close) and prev_close > 0:
                gap = (day_open - prev_close) / prev_close
                gap_pct[day_idx] = gap
                signal_bullish[day_idx] = first_bar_bullish

            prev_close = close[day_idx[-1]]

        # Filters
        time_ok = time_mins <= 584  # first 30 min (09:15-09:44)
        vix_ok = vix < vix_max
        vol_ok = rel_vol > rv_thresh

        # Entry: ride gap direction if confirmed by first candle
        long_entry = (gap_pct > gap_thresh) & signal_bullish & time_ok & vix_ok & vol_ok
        short_entry = (gap_pct < -gap_thresh) & (~signal_bullish) & time_ok & vix_ok & vol_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=360,
        )
