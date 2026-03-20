"""Last 30-Minute Momentum — Opus_39

Thesis: Stocks up or down >1% by 14:45 tend to accelerate into
the close. Enter in direction of day return when confirmed by VWAP.
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
    name = "opus_last_30_min_momentum_v1"
    is_long_only = False
    session_start = 885   # 14:45
    session_end = 929     # 15:29
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("day_ret_thresh", default=0.01, low=0.005, high=0.02),
            TunableParam("target_pct", default=0.003, low=0.002, high=0.006),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        day_ret_thresh = params.get("day_ret_thresh", 0.01)
        tgt_pct = params.get("target_pct", 0.003)
        stp_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
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

        # ── Open of day ──
        open_of_day = np.zeros(n, dtype=np.float64)
        prev_day = -1
        cur_open = 0.0
        for i in range(n):
            if day_id[i] != prev_day:
                cur_open = open_[i]
                prev_day = day_id[i]
            open_of_day[i] = cur_open

        safe_open = np.where(open_of_day > 0, open_of_day, 1.0)
        day_return = (close - open_of_day) / safe_open

        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        long_entry = (day_return > day_ret_thresh) & (close > vwap) & time_ok
        short_entry = (day_return < -day_ret_thresh) & (close < vwap) & time_ok

        # Time stop handles exit at 15:29

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stp_pct,
            time_stop_bars=44,  # max bars in 885-929 range
        )
