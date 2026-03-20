"""Bollinger Mean Reversion v1 — cursor_opus46max_011

Thesis: Bollinger Band (20,2.5) touches on 1-min bars identify extreme
deviations. Enter on bounce off band with bullish/bearish bar confirmation.
Target is BB midline (SMA 20).
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


class Strategy(BaseStrategy):
    name = "cursor_opus46max_011"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bb_mult", default=2.5, low=2.0, high=3.0),
            TunableParam("bb_width_min", default=0.3, low=0.15, high=0.6),
            TunableParam("vix_max", default=25.0, low=16.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        bb_mult = params.get("bb_mult", 2.5)
        bw_min = params.get("bb_width_min", 0.3)
        vix_max = params.get("vix_max", 25.0)
        stop_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)

        # ── VWAP (for proximity filter) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        close = df["close"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Bollinger Bands ──
        bb_mid = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        bb_mid = np.nan_to_num(bb_mid, nan=close[0] if n > 0 else 0.0)
        bb_std = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        bb_std = np.nan_to_num(bb_std, nan=1e10)
        bb_std = np.clip(bb_std, 1e-10, None)

        bb_upper = bb_mid + bb_mult * bb_std
        bb_lower = bb_mid - bb_mult * bb_std

        safe_mid = np.where(bb_mid > 0, bb_mid, 1.0)
        bb_width = (bb_upper - bb_lower) / safe_mid * 100.0

        bb_range = bb_upper - bb_lower
        bb_range = np.where(bb_range > 0, bb_range, 1e-10)
        bb_pct_b = (close - bb_lower) / bb_range

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Bar direction ──
        bullish = close > opn
        bearish = close < opn

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 900)
        vix_ok = vix < vix_max

        # ── Entries ──
        long_entry = (
            (low <= bb_lower)
            & (close > bb_lower)
            & (bb_pct_b > 0.0) & (bb_pct_b < 0.10)
            & (bb_width > bw_min)
            & (close > vwap * 0.995)
            & bullish
            & time_ok & vix_ok
        )
        short_entry = (
            (high >= bb_upper)
            & (close < bb_upper)
            & (bb_pct_b > 0.90) & (bb_pct_b < 1.0)
            & (bb_width > bw_min)
            & (close < vwap * 1.005)
            & bearish
            & time_ok & vix_ok
        )

        # ── Signal exit: %B further outside bands ──
        sig_exit_long = bb_pct_b < -0.05
        sig_exit_short = bb_pct_b > 1.05

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=bb_mid.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            time_stop_bars=25,
        )
