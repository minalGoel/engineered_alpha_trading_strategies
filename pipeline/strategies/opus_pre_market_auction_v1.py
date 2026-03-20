"""Pre-Market Auction Edge — Opus_38

Thesis: Gap direction combined with first-5-bar momentum predicts
continuation in early session. Enter when gap > 0.2% aligns with
first 5 bars direction; exit on VWAP cross.
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
    name = "opus_pre_market_auction_v1"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 630     # 10:30
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh", default=0.002, low=0.001, high=0.005),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_thresh = params.get("gap_thresh", 0.002)
        tgt_pct = params.get("target_pct", 0.004)
        stp_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        atr = _compute_atr(high, low, close, 20)

        # ── Per-day: prev_close, gap_pct, first_5_direction, open_of_day ──
        gap_pct = np.zeros(n, dtype=np.float64)
        first_5_dir = np.zeros(n, dtype=np.float64)  # +1 bullish, -1 bearish, 0 unknown
        open_of_day = np.zeros(n, dtype=np.float64)

        prev_day = -1
        prev_close = 0.0
        day_start_idx = 0
        day_open = 0.0
        bar_in_day = 0

        for i in range(n):
            if day_id[i] != prev_day:
                # new day
                if prev_day != -1:
                    prev_close = close[i - 1]
                day_start_idx = i
                day_open = open_[i]
                bar_in_day = 0
                prev_day = day_id[i]
            else:
                bar_in_day += 1

            open_of_day[i] = day_open

            # gap_pct: open vs prev_close
            if prev_close > 0:
                gap_pct[i] = (day_open - prev_close) / prev_close
            else:
                gap_pct[i] = 0.0

            # first_5_direction: after 5 bars, compare close[bar5] vs open[bar0]
            if bar_in_day >= 5:
                bar5_idx = day_start_idx + 5
                if bar5_idx < n and day_id[bar5_idx] == day_id[i]:
                    if close[bar5_idx] > day_open:
                        first_5_dir[i] = 1.0
                    elif close[bar5_idx] < day_open:
                        first_5_dir[i] = -1.0

        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        long_entry = (gap_pct > gap_thresh) & (first_5_dir > 0) & time_ok
        short_entry = (gap_pct < -gap_thresh) & (first_5_dir < 0) & time_ok

        # Signal exit: close crosses VWAP against trade
        exit_long = close < vwap
        exit_short = close > vwap

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=exit_long,
            signal_exit_short=exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stp_pct,
            time_stop_bars=75,
        )
