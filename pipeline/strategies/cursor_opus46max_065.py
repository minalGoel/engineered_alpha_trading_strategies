"""Gap Reversal Volume v1 — cursor_opus46max_065

Thesis: Gap reversal confirmed by contrary volume imbalance (OBV divergence).
Enter on reversal bar + VWAP cross.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "cursor_opus46max_065"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min", default=0.005, low=0.003, high=0.01),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.006, low=0.003, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_min = params.get("gap_min", 0.005)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.006)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high_ = df["high"].to_numpy().astype(np.float64)
        low_ = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df2 = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns(
            (pl.col("_ctv") / pl.col("_cv")).alias("_vwap")
        )
        vwap = df2["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        atr14 = _compute_atr(high_, low_, close, 14)

        # OBV
        obv = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]

        # OBV slope (15-bar)
        obv_slope = np.zeros(n, dtype=np.float64)
        for i in range(15, n):
            seg = obv[i-14:i+1]
            x = np.arange(15, dtype=np.float64)
            x_mean = np.mean(x)
            s_mean = np.mean(seg)
            num = np.sum((x - x_mean) * (seg - s_mean))
            den = np.sum((x - x_mean) ** 2)
            if den > 1e-12:
                obv_slope[i] = num / den

        # Gap per day
        gap_pct = np.zeros(n, dtype=np.float64)
        prev_close_arr = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        prev_close = np.nan
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) == 0:
                continue
            day_open = open_[idx[0]]
            if not np.isnan(prev_close) and prev_close > 0:
                gap_pct[idx] = (day_open - prev_close) / prev_close
                prev_close_arr[idx] = prev_close
            prev_close = close[idx[-1]]

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 570) & (time_mins <= 630)  # 09:30-10:30

        # Reversal bar + VWAP cross
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # Gap down reversal: OBV slope positive, close > prev high, VWAP cross
            if (gap_pct[i] < -gap_min and obv_slope[i] > 0
                    and close[i] > open_[i] and i > 0 and close[i] > high_[i-1]
                    and close[i-1] < vwap[i-1] and close[i] >= vwap[i]
                    and vix_ok[i] and time_ok[i]):
                long_entry[i] = True
            # Gap up reversal: OBV slope negative, close < prev low, VWAP cross
            if (gap_pct[i] > gap_min and obv_slope[i] < 0
                    and close[i] < open_[i] and close[i] < low_[i-1]
                    and close[i-1] > vwap[i-1] and close[i] <= vwap[i]
                    and vix_ok[i] and time_ok[i]):
                short_entry[i] = True

        # Signal exit: price reaches prev close
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if prev_close_arr[i] > 0 and gap_pct[i] < 0 and close[i] >= prev_close_arr[i]:
                signal_exit_long[i] = True
            if prev_close_arr[i] > 0 and gap_pct[i] > 0 and close[i] <= prev_close_arr[i]:
                signal_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=0.6,
            time_stop_bars=120,
        )
