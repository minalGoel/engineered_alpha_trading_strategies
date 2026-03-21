"""ATR Breakout — 5-second NIFTY options strategy.

When a 5-second NIFTY candle range exceeds 1.5× the 2-minute rolling ATR,
it signals large directional institutional order flow. We enter in the
direction of the expansion candle (bullish→CE, bearish→PE) and ride the
30-60 second continuation leg before the program order completes.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Compute True Range: max(H-L, |H-prev_C|, |L-prev_C|)."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = hl if hl >= hc and hl >= lc else (hc if hc >= lc else lc)
    return tr


def _rolling_mean(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple rolling mean; values before `period` bars are 0.0."""
    n = len(arr)
    out = np.zeros(n)
    cumsum = 0.0
    for i in range(n):
        cumsum += arr[i]
        if i >= period:
            cumsum -= arr[i - period]
            out[i] = cumsum / period
        elif i == period - 1:
            out[i] = cumsum / period
    return out


class Strategy(BaseStrategy):
    name = "atr_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 minutes warmup (covers 60-bar volume mean)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("range_mult", 1.5, 1.0, 2.5),
            TunableParam("vol_mult", 1.2, 0.8, 2.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before converting) ──────────
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        range_mult = params.get("range_mult", 1.5)
        vol_mult = params.get("vol_mult", 1.2)
        stop_pts = params.get("stop_pts", 3.0)
        target_pts = params.get("target_pts", 5.0)

        # ── ATR over 24 bars = 2 minutes ──────────────────────────────────────
        # Compressed from original 14×1min: our hold is 15-60s so we measure
        # local micro-volatility at the same timescale as the trade, not a
        # session-scale regime. 24 bars is the smallest window that filters noise.
        tr = _true_range(high, low, close)
        atr_24 = _rolling_mean(tr, 24)

        # ── Volume rolling mean over 60 bars = 5 minutes ──────────────────────
        vol_mean_60 = _rolling_mean(volume, 60)

        # ── Indicators ────────────────────────────────────────────────────────
        bar_range = high - low

        # Range expansion: current 5s candle wider than range_mult × ATR(24)
        # Guard against atr=0 at session open
        atr_nonzero = np.where(atr_24 > 0.0, atr_24, np.inf)
        range_expansion = bar_range > (range_mult * atr_nonzero)

        # Volume confirmation: meaningful activity, not just a quote tick
        vol_ok = (vol_mean_60 > 0) & (volume > vol_mult * vol_mean_60)

        # Candle direction selects CE or PE
        bullish_bar = close > open_
        bearish_bar = close < open_

        # Session and warmup guards
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed_up = np.zeros(n, dtype=bool)
        warmed_up[60:] = True   # need 60 bars for vol_mean_60

        # ── Signals ───────────────────────────────────────────────────────────
        base_signal = in_session & warmed_up & range_expansion & vol_ok

        buy_ce = base_signal & bullish_bar
        buy_pe = base_signal & bearish_bar

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,            # 60 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
