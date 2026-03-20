"""Opening Range Breakout — cursor_gpt54_opening_range_breakout_002

Thesis: Exploits delayed institutional continuation after the opening range
resolves with participation and trend alignment.  Price discovery is fragmented
between the auction, cash open, and slower institutional execution.

Entry long: close > OR-high (5 min), close > VWAP, EMA(8) > EMA(20),
    rel_vol >= 1.2, index VWAP dev >= 0, close > prev high.
Entry short: mirror conditions.
Target: 0.70% (approx 1.8R).  Stop: 0.9 * ATR.  Time stop: 22 bars.
VIX filter: < 24.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_ema(arr, period):
    n = len(arr)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    alpha = 2.0 / (period + 1)
    ema[0] = arr[0]
    for i in range(1, n):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i - 1]
    return ema


class Strategy(BaseStrategy):
    name = "opening_range_breakout_002"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 660     # 11:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rel_vol_min", default=1.2, low=1.0, high=2.0),
            TunableParam("stop_atr_mult", default=0.9, low=0.5, high=1.5),
            TunableParam("target_pct", default=0.007, low=0.004, high=0.012),
            TunableParam("vix_max", default=24.0, low=18.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rel_vol_min = params.get("rel_vol_min", 1.2)
        stop_atr = params.get("stop_atr_mult", 0.9)
        target_pct = params.get("target_pct", 0.007)
        vix_max = params.get("vix_max", 24.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        atr = _compute_atr(high, low, close, 14)

        # ── VWAP ──
        vwap_df = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = vwap_df["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Index VWAP (approximate using index_close as proxy) ──
        # Compute cumulative index "VWAP" using day_id grouping
        idx_vwap = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        for d in unique_days:
            mask = day_ids == d
            idx_d = np.where(mask)[0]
            if len(idx_d) == 0:
                continue
            cum_sum = np.cumsum(index_close[idx_d])
            counts = np.arange(1, len(idx_d) + 1, dtype=np.float64)
            idx_vwap[idx_d] = cum_sum / counts
        idx_vwap = np.where(idx_vwap > 0, idx_vwap, 1.0)
        index_vwap_dev = (index_close - idx_vwap) / idx_vwap

        # ── EMA(8) and EMA(20) ──
        ema8 = _compute_ema(close, 8)
        ema20 = _compute_ema(close, 20)

        # ── Relative volume ──
        vol_sma20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        vol_sma20 = np.nan_to_num(vol_sma20, nan=1.0)
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        rel_vol = volume / vol_sma20

        # ── Opening range (first 5 market minutes: 555-559) ──
        or_high = np.full(n, np.nan, dtype=np.float64)
        or_low = np.full(n, np.nan, dtype=np.float64)
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) == 0:
                continue
            or_bars = idx[time_mins[idx] < 560]  # first 5 min
            if len(or_bars) == 0:
                continue
            or_high[idx] = np.max(high[or_bars])
            or_low[idx] = np.min(low[or_bars])
        or_high = np.nan_to_num(or_high, nan=0.0)
        or_low = np.nan_to_num(or_low, nan=0.0)

        # ── Time filter: 09:20-11:00 (560-660) ──
        time_ok = (time_mins >= 560) & (time_mins <= 660)

        # ── Prev bar high/low ──
        prev_high = np.zeros(n, dtype=np.float64)
        prev_low = np.zeros(n, dtype=np.float64)
        prev_high[1:] = high[:-1]
        prev_low[1:] = low[:-1]

        long_entry = (
            (close > or_high) &
            (close > vwap) &
            (ema8 > ema20) &
            (rel_vol >= rel_vol_min) &
            (index_vwap_dev >= 0) &
            (close > prev_high) &
            (vix < vix_max) &
            time_ok
        )

        short_entry = (
            (close < or_low) &
            (close < vwap) &
            (ema8 < ema20) &
            (rel_vol >= rel_vol_min) &
            (index_vwap_dev <= 0) &
            (close < prev_low) &
            (vix < vix_max) &
            time_ok
        )

        # ── Signal exit: close crosses back through ema_long (20) ──
        sig_exit_long = close < ema20
        sig_exit_short = close > ema20

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_pct=target_pct,
            time_stop_bars=22,
        )
