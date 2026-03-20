"""Bollinger Band Squeeze Breakout — claude_project_14_of_25

Thesis: When Bollinger Band width contracts to the lowest percentile (squeeze),
a breakout through the upper or lower band with volume confirmation signals
the start of a new trend move.
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
    name = "bb_squeeze_breakout_v1"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("width_pctile_thresh", default=10.0, low=5.0, high=20.0),
            TunableParam("vol_mult", default=1.5, low=1.0, high=3.0),
            TunableParam("trailing_atr_mult", default=1.5, low=0.5, high=3.0),
            TunableParam("target_atr_mult", default=2.0, low=1.0, high=4.0),
            TunableParam("vix_low", default=11.0, low=8.0, high=16.0),
            TunableParam("vix_high", default=24.0, low=20.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        width_pctile_thresh = params.get("width_pctile_thresh", 10.0)
        vol_mult = params.get("vol_mult", 1.5)
        trailing_atr_mult = params.get("trailing_atr_mult", 1.5)
        target_atr_mult = params.get("target_atr_mult", 2.0)
        vix_low = params.get("vix_low", 11.0)
        vix_high = params.get("vix_high", 24.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)

        volume = np.nan_to_num(volume, nan=1.0)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── Bollinger Bands (20, 2) ──
        sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        sma20 = np.nan_to_num(sma20, nan=0.0)
        std20 = np.nan_to_num(std20, nan=1e10)

        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20

        # ── BB width and rolling percentile rank over 100 bars ──
        bb_width = (bb_upper - bb_lower) / np.where(sma20 > 0, sma20, 1.0)
        bb_width = np.nan_to_num(bb_width, nan=1.0)

        width_pctile = np.full(n, 50.0, dtype=np.float64)
        rank_window = 100
        for i in range(rank_window, n):
            window = bb_width[i - rank_window:i + 1]
            width_pctile[i] = (np.sum(window < bb_width[i]) / len(window)) * 100.0

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Volume filter ──
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol_20

        # ── VIX filter ──
        vix_ok = (vix >= vix_low) & (vix <= vix_high)

        # ── Squeeze condition ──
        squeeze = width_pctile < width_pctile_thresh

        # ── Entry ──
        long_entry = squeeze & (close > bb_upper) & vol_ok & vix_ok
        short_entry = squeeze & (close < bb_lower) & vix_ok

        # ── Signal exit: close crosses SMA(20) ──
        signal_exit_long = close < sma20
        signal_exit_short = close > sma20

        # ── Trailing stop in pct from ATR ──
        # Use median ATR as a representative scalar for trailing
        median_atr = np.nanmedian(atr[atr > 0]) if np.any(atr > 0) else 0.0
        median_close = np.nanmedian(close[close > 0]) if np.any(close > 0) else 1.0
        trailing_pct = (trailing_atr_mult * median_atr / median_close) if median_close > 0 else 0.005

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=sma20.copy(),
            stop_loss_atr_mult=0.0,
            target_atr_mult=target_atr_mult,
            trailing_stop_pct=trailing_pct,
            trailing_activate_pct=0.0,
            time_stop_bars=180,
        )
