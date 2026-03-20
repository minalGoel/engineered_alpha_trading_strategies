"""Keltner Channel Reversion v1 — cursor_opus46max_012

Thesis: Keltner Channels (EMA(20) +/- 2.5*ATR(10)) are more stable than
Bollinger Bands. Enter on bounce off channel with strong rejection bar.
Target is KC midline (EMA 20).
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


def _compute_ema(close, period):
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = close[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_012"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("kc_mult", default=2.5, low=2.0, high=3.5),
            TunableParam("bounce_thresh", default=0.6, low=0.5, high=0.8),
            TunableParam("vix_max", default=25.0, low=16.0, high=30.0),
            TunableParam("stop_atr_mult", default=1.0, low=0.5, high=2.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        kc_mult = params.get("kc_mult", 2.5)
        bounce_thr = params.get("bounce_thresh", 0.6)
        vix_max = params.get("vix_max", 25.0)
        stop_atr = params.get("stop_atr_mult", 1.0)

        n = len(df)

        close = df["close"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Keltner Channel ──
        kc_mid = _compute_ema(close, 20)
        kc_atr = _compute_atr(high, low, close, 10)
        kc_upper = kc_mid + kc_mult * kc_atr
        kc_lower = kc_mid - kc_mult * kc_atr

        kc_range = kc_upper - kc_lower
        kc_range = np.where(kc_range > 0, kc_range, 1e-10)
        kc_pos = (close - kc_lower) / kc_range

        # ── ATR(14) for main stops ──
        atr14 = _compute_atr(high, low, close, 14)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── Bar bounce pattern ──
        bar_range = high - low
        bar_range = np.where(bar_range > 0, bar_range, 1e-10)
        long_bounce = ((close - low) / bar_range > bounce_thr) & (close > opn)
        short_reject = ((high - close) / bar_range > bounce_thr) & (close < opn)

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)
        vix_ok = vix < vix_max

        # ── Entries ──
        long_entry = (
            (low <= kc_lower)
            & (close > kc_lower)
            & (kc_pos > 0) & (kc_pos < 0.15)
            & vol_ok & long_bounce
            & time_ok & vix_ok
        )
        short_entry = (
            (high >= kc_upper)
            & (close < kc_upper)
            & (kc_pos > 0.85) & (kc_pos < 1.0)
            & vol_ok & short_reject
            & time_ok & vix_ok
        )

        # ── Signal exit: channel walk ──
        sig_exit_long = kc_pos < -0.1
        sig_exit_short = kc_pos > 1.1

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr14,
            target_indicator=kc_mid.copy(),
            stop_loss_atr_mult=stop_atr,
            use_target_indicator=True,
            time_stop_bars=30,
        )
