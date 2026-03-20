# AUDIT FIX: Added session_end upper bound in pullback detection loop and time_ok filter in long/short_entry
"""First Pullback After Opening Move — Opus_14

Thesis: After a strong directional move in the first 45 minutes (>0.5%),
the first pullback to EMA(9) offers a high-probability continuation entry
when confirmed by VWAP alignment and RSI in the 40–70 zone.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple EMA with Wilder-style warmup."""
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    k = 2.0 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = arr[i] * k + ema[i - 1] * (1.0 - k)
    return ema


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI."""
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[1:period + 1])
    avg_loss = np.mean(loss[1:period + 1])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


class Strategy(BaseStrategy):
    name = "opus_first_pullback_v1"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 780     # 13:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("open_move_pct", default=0.005, low=0.003, high=0.01),
            TunableParam("rsi_lo", default=40.0, low=30.0, high=50.0),
            TunableParam("rsi_hi", default=70.0, low=60.0, high=80.0),
            TunableParam("target_atr_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("stop_atr_mult", default=1.0, low=0.5, high=1.5),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        open_move = params.get("open_move_pct", 0.005)
        rsi_lo = params.get("rsi_lo", 40.0)
        rsi_hi = params.get("rsi_hi", 70.0)
        tgt_mult = params.get("target_atr_mult", 1.5)
        stp_mult = params.get("stop_atr_mult", 1.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_vwap = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns(
            (pl.col("_ctv") / pl.col("_cv")).alias("_vwap")
        )
        vwap = df_vwap["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        ema9 = _compute_ema(close, 9)
        rsi14 = _compute_rsi(close, 14)
        atr20 = _compute_atr(high, low, close, 20)

        # ── Open price per day ──
        open_price = np.zeros(n, dtype=np.float64)
        cur_day = day_id[0]
        cur_open = close[0]
        for i in range(n):
            if day_id[i] != cur_day:
                cur_day = day_id[i]
                cur_open = close[i]
            open_price[i] = cur_open

        safe_open = np.where(open_price > 0, open_price, 1.0)
        ret_from_open = (close - open_price) / safe_open

        # ── First pullback detection per day ──
        # Pullback = close touches EMA(9) from above (long) or below (short)
        pullback_long = np.zeros(n, dtype=np.bool_)
        pullback_short = np.zeros(n, dtype=np.bool_)
        pullback_count_long = 0
        pullback_count_short = 0
        prev_day = day_id[0]

        for i in range(1, n):
            if day_id[i] != prev_day:
                pullback_count_long = 0
                pullback_count_short = 0
                prev_day = day_id[i]

            # First pullback to EMA9 for long: was above, now touches
            if (ret_from_open[i] > open_move and
                    low[i] <= ema9[i] <= high[i] and
                    pullback_count_long == 0 and
                    self.session_start <= time_mins[i] <= self.session_end):
                pullback_long[i] = True
                pullback_count_long += 1

            # First pullback for short
            if (ret_from_open[i] < -open_move and
                    low[i] <= ema9[i] <= high[i] and
                    pullback_count_short == 0 and
                    self.session_start <= time_mins[i] <= self.session_end):
                pullback_short[i] = True
                pullback_count_short += 1

        rsi_ok = (rsi14 >= rsi_lo) & (rsi14 <= rsi_hi)
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        long_entry = pullback_long & (close > vwap) & rsi_ok & time_ok
        short_entry = pullback_short & (close < vwap) & rsi_ok & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr20,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_atr_mult=tgt_mult,
            stop_loss_atr_mult=stp_mult,
            time_stop_bars=90,
        )
