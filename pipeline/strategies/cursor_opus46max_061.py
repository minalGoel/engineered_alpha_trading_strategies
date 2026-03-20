"""Gap Fill Full v1 — cursor_opus46max_061

Thesis: Overnight gaps of 0.3-1.0% fill completely ~65-70% of the time.
Fade the gap after VWAP cross confirmation.
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


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff = close[i] - close[i-1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    if avg_loss > 1e-10:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    else:
        rsi[period] = 100.0
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 1e-10:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
        else:
            rsi[i] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_061"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min", default=0.003, low=0.002, high=0.006),
            TunableParam("gap_max", default=0.01, low=0.006, high=0.02),
            TunableParam("rsi_low", default=40.0, low=25.0, high=50.0),
            TunableParam("rsi_high", default=60.0, low=50.0, high=75.0),
            TunableParam("vix_max", default=20.0, low=14.0, high=26.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_min = params.get("gap_min", 0.003)
        gap_max = params.get("gap_max", 0.01)
        rsi_low = params.get("rsi_low", 40.0)
        rsi_high = params.get("rsi_high", 60.0)
        vix_max = params.get("vix_max", 20.0)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high_ = df["high"].to_numpy().astype(np.float64)
        low_ = df["low"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
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

        atr = _compute_atr(high_, low_, close, 14)
        rsi = _compute_rsi(close, 14)

        # Gap per day
        gap_pct = np.zeros(n, dtype=np.float64)
        prev_close_arr = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        prev_close = np.nan
        for d in unique_days:
            mask = day_ids == d
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            day_open = open_[idx[0]]
            if not np.isnan(prev_close) and prev_close > 0:
                gap_pct[idx] = (day_open - prev_close) / prev_close
                prev_close_arr[idx] = prev_close
            prev_close = close[idx[-1]]

        # First 5-bar low/high per day
        first5_low = np.full(n, np.inf, dtype=np.float64)
        first5_high = np.full(n, -np.inf, dtype=np.float64)
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) < 5:
                continue
            f5l = np.min(low_[idx[:5]])
            f5h = np.max(high_[idx[:5]])
            first5_low[idx] = f5l
            first5_high[idx] = f5h

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 560) & (time_mins <= 630)  # 09:20-10:30

        # Volume ratio
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # VWAP cross detection
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            gp = gap_pct[i]
            # Gap down: long when crossing above VWAP
            if (gp < -gap_min and gp > -gap_max and rsi[i] < rsi_low
                    and close[i] > first5_low[i] and vol_ratio[i] > 0.8
                    and vix_ok[i] and time_ok[i]
                    and close[i-1] < vwap[i-1] and close[i] >= vwap[i]):
                long_entry[i] = True
            # Gap up: short when crossing below VWAP
            if (gp > gap_min and gp < gap_max and rsi[i] > rsi_high
                    and close[i] < first5_high[i] and vol_ratio[i] > 0.8
                    and vix_ok[i] and time_ok[i]
                    and close[i-1] > vwap[i-1] and close[i] <= vwap[i]):
                short_entry[i] = True

        # Signal exit: reaches prev close
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if prev_close_arr[i] > 0:
                if close[i] >= prev_close_arr[i] and gap_pct[i] < 0:
                    signal_exit_long[i] = True
                if close[i] <= prev_close_arr[i] and gap_pct[i] > 0:
                    signal_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=0.5,
            time_stop_bars=180,
        )
