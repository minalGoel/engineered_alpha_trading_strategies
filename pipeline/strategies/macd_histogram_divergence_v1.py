"""MACD Histogram Divergence — 5-second NIFTY index options.

Mechanism: Detects momentum exhaustion via MACD histogram divergence on NIFTY.
When price makes a new intraday low but the MACD histogram trough is shallower than
the prior trough, institutional selling waves are losing force against resting buy
orders. The histogram zero-cross above 0 confirms accumulation has begun, triggering
a buy-CE entry. Symmetric logic applies for bearish divergence (buy-PE).

Indicators: MACD(12,26,9) on 5s bars = 60s/130s/45s windows, 3-bar pivot detection.
Hold: 30-120 seconds (time_stop_bars=24).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average, causal."""
    alpha = 2.0 / (period + 1.0)
    result = np.empty(len(arr))
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
    return result


def _find_pivot_lows(low: np.ndarray, k: int):
    """Detect confirmed pivot lows with k-bar lookback each side.

    CAUSALITY NOTE (audit 2025-03): Although the inner loop reads k future bars
    (low[i+1:i+k+1]), the result is stamped at ``confirm = i + k``, NOT at bar i.
    Downstream code only reads pivot_vals[j] at bar j — the confirmation bar — by
    which time all k "future" bars are already in the past.  This is the standard
    causal pivot detection pattern: detect at i, materialise at i+k, consume at i+k.
    No look-ahead bias exists.

    Returns (pivot_vals, pivot_idxs) — both length n.
      pivot_vals[j] = low value at the confirmed pivot (NaN if not a confirm bar)
      pivot_idxs[j] = index of the actual pivot bar (−1 if not a confirm bar)
    """
    n = len(low)
    pivot_vals = np.full(n, np.nan)
    pivot_idxs = np.full(n, -1, dtype=np.int32)
    for i in range(k, n - k):
        if low[i] < np.min(low[i - k : i]) and low[i] < np.min(low[i + 1 : i + k + 1]):
            confirm = i + k
            pivot_vals[confirm] = low[i]
            pivot_idxs[confirm] = i
    return pivot_vals, pivot_idxs


def _find_pivot_highs(high: np.ndarray, k: int):
    """Detect confirmed pivot highs with k-bar lookback each side.

    See _find_pivot_lows CAUSALITY NOTE — same pattern applies here.

    Returns (pivot_vals, pivot_idxs).
    """
    n = len(high)
    pivot_vals = np.full(n, np.nan)
    pivot_idxs = np.full(n, -1, dtype=np.int32)
    for i in range(k, n - k):
        if high[i] > np.max(high[i - k : i]) and high[i] > np.max(high[i + 1 : i + k + 1]):
            confirm = i + k
            pivot_vals[confirm] = high[i]
            pivot_idxs[confirm] = i
    return pivot_vals, pivot_idxs


class Strategy(BaseStrategy):
    name = "macd_histogram_divergence_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip opening 15-min noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 240            # 20 min warmup for MACD + two pivot cycles

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum histogram difference (in MACD units) to qualify as divergence
            TunableParam("hist_div_min", 0.3, 0.05, 1.5),
            # Minimum price difference (NIFTY spot pts) between pivot extremes
            TunableParam("price_div_min", 2.0, 0.5, 8.0),
            TunableParam("stop_pts", 3.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 14.0),
        ]

    def compute(self, spot_df: pl.DataFrame, option_df: pl.DataFrame,
                vix_df: pl.DataFrame, params: dict) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (forward-fill NaN before numpy) ──────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        high  = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low   = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()

        hist_div_min = params.get("hist_div_min", 0.3)
        price_div_min = params.get("price_div_min", 2.0)
        stop_pts  = params.get("stop_pts",  3.0)
        target_pts = params.get("target_pts", 6.0)

        # ── VIX filter (< 22 to avoid panic-regime divergence failures) ─────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime", strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── MACD(12, 26, 9) on spot close ───────────────────────────────────
        # At 5s bars: fast=12 bars=60s, slow=26 bars=130s, signal=9 bars=45s
        # This matches 2-4 minute divergence formation windows for 30-120s holds.
        ema12       = _compute_ema(close, 12)
        ema26       = _compute_ema(close, 26)
        macd_line   = ema12 - ema26
        macd_signal = _compute_ema(macd_line, 9)
        macd_hist   = macd_line - macd_signal

        # ── Pivot detection — 3-bar lookback (15s each side) ────────────────
        pivot_k = 3
        pl_vals, pl_idxs = _find_pivot_lows(low, pivot_k)
        ph_vals, ph_idxs = _find_pivot_highs(high, pivot_k)

        # ── Divergence detection ─────────────────────────────────────────────
        # For each new confirmed pivot, look back up to div_window bars for the
        # prior pivot. Divergence is flagged at the confirmation bar.
        div_window = 120  # 10 minutes: reasonable span for two-wave divergence on NIFTY

        bull_div = np.zeros(n, dtype=bool)  # bullish divergence confirmed at this bar
        bear_div = np.zeros(n, dtype=bool)  # bearish divergence confirmed at this bar

        for i in range(1, n):
            # Bullish divergence check (pivot lows)
            if not np.isnan(pl_vals[i]):
                cur_price_low = pl_vals[i]
                cur_pivot_bar = pl_idxs[i]
                cur_hist = macd_hist[cur_pivot_bar]
                # Search backward for the most recent prior pivot low
                for j in range(i - 1, max(0, i - div_window) - 1, -1):
                    if not np.isnan(pl_vals[j]):
                        prev_price_low = pl_vals[j]
                        prev_pivot_bar = pl_idxs[j]
                        prev_hist = macd_hist[prev_pivot_bar]
                        # Lower price low + higher histogram trough = bullish divergence
                        if (cur_price_low < prev_price_low - price_div_min and
                                cur_hist > prev_hist + hist_div_min):
                            bull_div[i] = True
                        break  # only compare to the immediately prior pivot

            # Bearish divergence check (pivot highs)
            if not np.isnan(ph_vals[i]):
                cur_price_high = ph_vals[i]
                cur_pivot_bar = ph_idxs[i]
                cur_hist = macd_hist[cur_pivot_bar]
                for j in range(i - 1, max(0, i - div_window) - 1, -1):
                    if not np.isnan(ph_vals[j]):
                        prev_price_high = ph_vals[j]
                        prev_pivot_bar = ph_idxs[j]
                        prev_hist = macd_hist[prev_pivot_bar]
                        # Higher price high + lower histogram peak = bearish divergence
                        if (cur_price_high > prev_price_high + price_div_min and
                                cur_hist < prev_hist - hist_div_min):
                            bear_div[i] = True
                        break

        # ── Histogram zero-cross signals ─────────────────────────────────────
        hist_prev    = np.empty(n)
        hist_prev[0] = 0.0
        hist_prev[1:] = macd_hist[:-1]
        zero_cross_up = (hist_prev < 0.0) & (macd_hist >= 0.0)   # cross above zero
        zero_cross_dn = (hist_prev > 0.0) & (macd_hist <= 0.0)   # cross below zero

        # ── Entry: zero-cross within prop_window bars after a divergence ─────
        # Divergence signal is valid for up to 60 bars (5 min) before expiring.
        prop_window = 60

        # Build cumulative sums for O(1) "any divergence in last N bars" check
        bull_cumsum = np.cumsum(bull_div.astype(np.int32))
        bear_cumsum = np.cumsum(bear_div.astype(np.int32))

        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok     = vix_close < 22.0

        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        for i in range(n):
            if not in_session[i] or not vix_ok[i]:
                continue
            start = max(0, i - prop_window)
            # Count divergences in [start, i]
            bull_count = bull_cumsum[i] - (bull_cumsum[start - 1] if start > 0 else 0)
            bear_count = bear_cumsum[i] - (bear_cumsum[start - 1] if start > 0 else 0)

            if zero_cross_up[i] and bull_count > 0:
                buy_ce[i] = True
            if zero_cross_dn[i] and bear_count > 0:
                buy_pe[i] = True

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,             # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
