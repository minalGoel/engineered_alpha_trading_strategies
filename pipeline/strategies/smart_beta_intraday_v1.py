"""smart_beta_intraday_v1 — Keltner Channel Volatility Compression Breakout on NIFTY

Adapted from: trading_strategies/unique_strategies_all/Strategy_295.json
Original: Smart beta low-vol anomaly breakout on NIFTY100 stocks (1-min bars, 30-90 min hold)

Mechanism:
  On NIFTY, intraday consolidation phases compress Keltner Channel bands as institutional
  algos pause directional orders. When ATR falls to the bottom 30th percentile of its
  10-minute rolling window, the index is coiling. A break above/below the Keltner bands
  with simultaneous range expansion (current bar range > 1.5× 5-min average) signals
  institutional commitment that sustains directional flow for 30-90 seconds.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Wilder-style EMA (exponential, not Wilder — standard alpha=2/(period+1))."""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    # Seed with simple average of first `period` values
    first_valid = np.nanmean(arr[:period])
    if np.isnan(first_valid):
        return result
    result[period - 1] = first_valid
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        if np.isnan(arr[i]):
            result[i] = result[i - 1]
        else:
            result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
    return result


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Smoothed ATR using EMA of true range."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    return _ema(tr, period)


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        result[i] = np.mean(arr[i - period + 1: i + 1])
    return result


def _vol_percentile_rank(atr: np.ndarray, window: int) -> np.ndarray:
    """Rolling percentile rank of current ATR within a rolling window.

    Returns value in [0, 1]: 0.0 = lowest ATR in window, 1.0 = highest.
    """
    n = len(atr)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        window_vals = atr[i - window + 1: i + 1]
        valid = window_vals[~np.isnan(window_vals)]
        if len(valid) < 2:
            continue
        current = atr[i]
        if np.isnan(current):
            continue
        result[i] = float(np.sum(valid <= current)) / len(valid)
    return result


class Strategy(BaseStrategy):
    """Keltner Channel Volatility Compression Breakout on NIFTY.

    Detects when NIFTY has been coiling (ATR in bottom 30th percentile of 10-min
    rolling window) and then breaks out of the Keltner channel with range expansion.
    Trades the directional momentum continuation for 30-90 seconds.
    """

    name = "smart_beta_intraday_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip pre-open noise
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 240            # 20-min warmup for EMA/ATR convergence

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("range_ratio_threshold", 1.5, 1.1, 2.5),
            TunableParam("vol_pct_threshold", 0.30, 0.15, 0.45),
            TunableParam("momentum_threshold", 0.0003, 0.0001, 0.0010),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract price arrays (forward-fill NaN before numpy) ──────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        range_thr = params.get("range_ratio_threshold", 1.5)
        vol_pct_thr = params.get("vol_pct_threshold", 0.30)
        mom_thr = params.get("momentum_threshold", 0.0003)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── Keltner Channel: EMA(60) ± 1.5 × ATR(60) ─────────────────────────
        # 60 bars = 5 minutes — captures recent micro-trend as context
        ema60 = _ema(close, 60)
        atr60 = _atr(high, low, close, 60)

        # Fill NaN for channel bounds (use close as neutral midline)
        ema60_f = np.where(np.isnan(ema60), close, ema60)
        atr60_median = float(np.nanmedian(atr60)) if not np.all(np.isnan(atr60)) else 20.0
        atr60_f = np.where(np.isnan(atr60), atr60_median, atr60)

        keltner_upper = ema60_f + 1.5 * atr60_f
        keltner_lower = ema60_f - 1.5 * atr60_f

        # ── Range Ratio: current bar range vs 5-min SMA of bar ranges ─────────
        bar_range = high - low
        bar_range_sma = _sma(bar_range, 60)
        bar_range_sma_f = np.where(
            np.isnan(bar_range_sma) | (bar_range_sma == 0.0),
            np.where(bar_range > 0, bar_range, 1.0),
            bar_range_sma,
        )
        range_ratio = bar_range / bar_range_sma_f

        # ── Volatility Percentile: ATR(12) rank in 120-bar rolling window ─────
        # ATR(12) = 1-min ATR; 120-bar window = 10-min context
        atr12 = _atr(high, low, close, 12)
        vol_pct = _vol_percentile_rank(atr12, 120)
        vol_pct_f = np.where(np.isnan(vol_pct), 0.5, vol_pct)  # neutral = median rank

        # Lag vol_percentile by 12 bars (60 seconds): confirm compression was
        # BEFORE the current breakout bar, not caused by it
        vol_pct_lag12 = np.concatenate([np.full(12, 0.5), vol_pct_f[:-12]])

        # ── 30-second Momentum: 6-bar return ──────────────────────────────────
        # Replaces original OBV_slope (unavailable/noisy at index level)
        mom6 = np.zeros(n)
        denom = np.where(close[:-6] > 0, close[:-6], 1.0)
        mom6[6:] = (close[6:] - close[:-6]) / denom

        # ── Session Filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes) &
            (time_min < self.session_end_minutes)
        )

        # ── Entry Signals ─────────────────────────────────────────────────────
        # buy_ce: Keltner upper breakout + range expansion + prior compression + up momentum
        buy_ce = (
            in_session &
            (close > keltner_upper) &
            (range_ratio > range_thr) &
            (vol_pct_lag12 < vol_pct_thr) &
            (mom6 > mom_thr)
        )

        # buy_pe: Keltner lower breakdown + range expansion + prior compression + down momentum
        buy_pe = (
            in_session &
            (close < keltner_lower) &
            (range_ratio > range_thr) &
            (vol_pct_lag12 < vol_pct_thr) &
            (mom6 < -mom_thr)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
