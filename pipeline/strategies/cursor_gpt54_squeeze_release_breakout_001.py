"""Squeeze Release Breakout — cursor_gpt54_squeeze_release_breakout_001

Thesis: Exploits volatility expansion after intraday compression resolves with
direction and volume.  Traders reduce risk during compression and then chase
once the move is obvious.

Squeeze: BB(20,2) inside KC(20,1.5*ATR(20)).  Entry on close breaking 10-bar
high/low with EMA(9) > EMA(21), close > VWAP, rel_vol >= 1.2.
Target: 0.80% (approx 2R).  Stop: 0.9*ATR.  Time stop: 30 bars.
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
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        out[i] = np.max(arr[start:i + 1])
    return out


def _rolling_min(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        out[i] = np.min(arr[start:i + 1])
    return out


class Strategy(BaseStrategy):
    name = "squeeze_release_breakout_001"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rel_vol_min", default=1.2, low=1.0, high=2.0),
            TunableParam("kc_atr_mult", default=1.5, low=1.0, high=2.0),
            TunableParam("stop_atr_mult", default=0.9, low=0.5, high=1.5),
            TunableParam("target_pct", default=0.008, low=0.005, high=0.012),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rel_vol_min = params.get("rel_vol_min", 1.2)
        kc_mult = params.get("kc_atr_mult", 1.5)
        stop_atr = params.get("stop_atr_mult", 0.9)
        target_pct = params.get("target_pct", 0.008)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr14 = _compute_atr(high, low, close, 14)
        atr20 = _compute_atr(high, low, close, 20)

        # ── VWAP ──
        vwap_df = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = vwap_df["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Bollinger Bands (SMA20, 2*std) ──
        close_sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        close_sma20 = np.nan_to_num(close_sma20, nan=0.0)
        close_std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        close_std20 = np.nan_to_num(close_std20, nan=0.0)
        bb_upper = close_sma20 + 2.0 * close_std20
        bb_lower = close_sma20 - 2.0 * close_std20

        # ── Keltner Channel (EMA20 +/- 1.5*ATR(20)) ──
        ema20 = _compute_ema(close, 20)
        kc_upper = ema20 + kc_mult * atr20
        kc_lower = ema20 - kc_mult * atr20

        # ── Squeeze: BB inside KC ──
        squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        # ── EMA(9) and EMA(21) ──
        ema9 = _compute_ema(close, 9)
        ema21 = _compute_ema(close, 21)

        # ── 10-bar high/low ──
        # AUDIT FIX: _rolling_max/min is inclusive of the current bar, so close > squeeze_high
        # is always False (close <= high <= rolling_max(high)).  Shift by 1 so we compare
        # against the PREVIOUS 10-bar high/low (breakout logic requires exceeding past range).
        _raw_squeeze_high = _rolling_max(high, 10)
        _raw_squeeze_low = _rolling_min(low, 10)
        squeeze_high = np.zeros(n, dtype=np.float64)
        squeeze_low = np.full(n, np.inf, dtype=np.float64)
        squeeze_high[1:] = _raw_squeeze_high[:-1]
        squeeze_low[1:] = _raw_squeeze_low[:-1]
        squeeze_low = np.where(np.isinf(squeeze_low), low, squeeze_low)

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

        # ── Time filter: 10:00-14:30 (600-870) ──
        time_ok = (time_mins >= 600) & (time_mins <= 870)

        long_entry = (
            squeeze_on &
            (close > squeeze_high) &
            (ema9 > ema21) &
            (close > vwap) &
            (rel_vol >= rel_vol_min) &
            (close > prev_high) &
            time_ok
        )

        short_entry = (
            squeeze_on &
            (close < squeeze_low) &
            (ema9 < ema21) &
            (close < vwap) &
            (rel_vol >= rel_vol_min) &
            (close < prev_low) &
            time_ok
        )

        # ── Signal exit: close back inside squeeze range ──
        sig_exit_long = (close < squeeze_high) & (close > squeeze_low)
        sig_exit_short = (close > squeeze_low) & (close < squeeze_high)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_pct=target_pct,
            time_stop_bars=30,
        )
