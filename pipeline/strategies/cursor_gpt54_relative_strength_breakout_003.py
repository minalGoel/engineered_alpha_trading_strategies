"""Relative Strength Breakout — cursor_gpt54_relative_strength_breakout_003

Thesis: Exploits delayed continuation in stocks that outperform their benchmark
intraday and then break consolidation with participation.  Institutional and
discretionary traders visually chase leaders after the benchmark confirms them.

Entry long: rel_ret >= 0.004, close > 25-bar high, close > VWAP, close > EMA(20),
    rel_vol >= 1.4, index VWAP dev >= 0, close > prev high.
Entry short: mirror conditions.
Target: 0.80% (approx 2R).  Stop: 0.8*ATR.  Time stop: 40 bars.
VIX: 12-24.
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


def _rolling_max(arr, period):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        out[i] = np.max(arr[i - period + 1:i + 1])
    # fill early values
    for i in range(period - 1):
        out[i] = np.max(arr[:i + 1])
    return out


def _rolling_min(arr, period):
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        out[i] = np.min(arr[i - period + 1:i + 1])
    for i in range(period - 1):
        out[i] = np.min(arr[:i + 1])
    return out


class Strategy(BaseStrategy):
    name = "relative_strength_breakout_003"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rel_ret_thresh", default=0.004, low=0.002, high=0.008),
            TunableParam("rel_vol_min", default=1.4, low=1.0, high=2.0),
            TunableParam("stop_atr_mult", default=0.8, low=0.5, high=1.2),
            TunableParam("target_pct", default=0.008, low=0.005, high=0.012),
            TunableParam("vix_max", default=24.0, low=18.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rel_ret_thresh = params.get("rel_ret_thresh", 0.004)
        rel_vol_min = params.get("rel_vol_min", 1.4)
        stop_atr = params.get("stop_atr_mult", 0.8)
        target_pct = params.get("target_pct", 0.008)
        vix_max = params.get("vix_max", 24.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        atr = _compute_atr(high, low, close, 14)
        ema20 = _compute_ema(close, 20)

        # ── VWAP ──
        vwap_df = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = vwap_df["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Index VWAP dev (simple SMA proxy) ──
        unique_days = np.unique(day_ids)
        idx_vwap = np.zeros(n, dtype=np.float64)
        for d in unique_days:
            mask = day_ids == d
            idx_d = np.where(mask)[0]
            if len(idx_d) == 0:
                continue
            cum = np.cumsum(index_close[idx_d])
            cnt = np.arange(1, len(idx_d) + 1, dtype=np.float64)
            idx_vwap[idx_d] = cum / cnt
        idx_vwap = np.where(idx_vwap > 0, idx_vwap, 1.0)
        index_vwap_dev = (index_close - idx_vwap) / idx_vwap

        # ── Relative return: stock vs index over 10 bars ──
        rel_ret = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            stock_ret = close[i] / close[i - 10] - 1.0 if close[i - 10] > 0 else 0.0
            idx_ret = index_close[i] / index_close[i - 10] - 1.0 if index_close[i - 10] > 0 else 0.0
            rel_ret[i] = stock_ret - idx_ret

        # ── Base high/low (25 bar rolling) ──
        base_high = _rolling_max(high, 25)
        base_low = _rolling_min(low, 25)

        # ── Relative volume ──
        vol_sma20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        vol_sma20 = np.nan_to_num(vol_sma20, nan=1.0)
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        rel_vol = volume / vol_sma20

        # ── Prev bar high/low ──
        prev_high = np.zeros(n, dtype=np.float64)
        prev_low = np.zeros(n, dtype=np.float64)
        prev_high[1:] = high[:-1]
        prev_low[1:] = low[:-1]

        # ── Time filter: 09:30-14:30 (570-870) ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # ── VIX filter: 12-24 ──
        vix_ok = (vix >= 12.0) & (vix <= vix_max)

        long_entry = (
            (rel_ret >= rel_ret_thresh) &
            (close > base_high) &
            (close > vwap) &
            (close > ema20) &
            (rel_vol >= rel_vol_min) &
            (index_vwap_dev >= 0) &
            (close > prev_high) &
            time_ok &
            vix_ok
        )

        short_entry = (
            (rel_ret <= -rel_ret_thresh) &
            (close < base_low) &
            (close < vwap) &
            (close < ema20) &
            (rel_vol >= rel_vol_min) &
            (index_vwap_dev <= 0) &
            (close < prev_low) &
            time_ok &
            vix_ok
        )

        # ── Signal exit: rel_ret flips sign ──
        sig_exit_long = rel_ret < 0
        sig_exit_short = rel_ret > 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_pct=target_pct,
            time_stop_bars=40,
        )
