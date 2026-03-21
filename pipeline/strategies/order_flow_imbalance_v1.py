"""Order Flow Imbalance strategy for NIFTY 5-second index options.

Thesis: When NIFTY's 5-second bars show abnormally high volume-weighted
buying pressure (volume * sign(close - open) z-score > 1.5) while price
is above session VWAP, it signals institutional TWAP accumulation whose
unexecuted tail will push NIFTY 10-20 spot points in the next 30-90 seconds.
Volume-weighting is the key differentiator — large-volume uptick bars carry
proportionally more signal than small doji bars.

Converted from: trading_strategies/unique_strategies_all/Strategy_53.json
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP, reset at each new day_id."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    current_day = -1
    for i in range(n):
        if day_id[i] != current_day:
            cum_pv = 0.0
            cum_v = 0.0
            current_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_v += volume[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0 else close[i]
    return vwap


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling sum over last `window` bars (partial sums for warm-up bars)."""
    n = len(arr)
    padded = np.zeros(n + 1)
    padded[1:] = np.cumsum(arr)
    result = np.zeros(n)
    result[:window - 1] = padded[1:window]
    result[window - 1:] = padded[window:] - padded[:n - window + 1]
    return result


class Strategy(BaseStrategy):
    name = "order_flow_imbalance_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min pre-open noise
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 72             # 6 min warmup (need 60 bars for zscore window)
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("delta_zscore_threshold", 1.5, 1.0, 2.5),
            TunableParam("stop_pts",               4.0, 2.0, 8.0),
            TunableParam("target_pts",             7.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract OHLCV arrays ───────────────────────────────────────────
        close = (
            spot_df["close"]
            .fill_nan(None)
            .fill_null(strategy="forward")
            .to_numpy()
        )
        open_ = (
            spot_df["open"]
            .fill_nan(None)
            .fill_null(strategy="forward")
            .to_numpy()
        )
        volume = (
            spot_df["volume"]
            .fill_null(0)
            .to_numpy()
            .astype(np.float64)
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────
        threshold  = params.get("delta_zscore_threshold", 1.5)
        stop_pts   = params.get("stop_pts",               4.0)
        target_pts = params.get("target_pts",             7.0)

        # ── Session VWAP (reset each day) ─────────────────────────────────
        vwap = _compute_vwap(close, volume, day_id)

        # ── Volume-weighted bar delta ─────────────────────────────────────
        # Positive on uptick bars (close>open), negative on downtick bars.
        # High-volume bars contribute proportionally more than doji bars.
        bar_delta = volume * np.sign(close - open_)

        # ── 1-minute cumulative volume delta (12 bars × 5s) ───────────────
        # Accumulation window: long enough to filter individual-bar noise,
        # short enough to remain predictive for a 15-90s hold.
        cum_delta_12 = _rolling_sum(bar_delta, 12)

        # ── Z-score vs 5-minute (60-bar) rolling baseline ─────────────────
        # Captures how abnormal the current 1-min buying/selling pressure is
        # relative to the recent session regime.
        NORM_WINDOW = 60
        delta_zscore = np.zeros(n)
        for i in range(NORM_WINDOW, n):
            window_vals = cum_delta_12[i - NORM_WINDOW:i]
            mean = np.mean(window_vals)
            std  = np.std(window_vals)
            if std > 0:
                delta_zscore[i] = (cum_delta_12[i] - mean) / std

        # ── Session filter ────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ─────────────────────────────────────────────────
        # Bullish: volume-weighted buying abnormally strong + price above VWAP
        buy_ce = in_session & (delta_zscore > threshold) & (close > vwap)

        # Bearish: volume-weighted selling abnormally strong + price below VWAP
        buy_pe = in_session & (delta_zscore < -threshold) & (close < vwap)

        # Prevent simultaneous CE + PE signals
        both   = buy_ce & buy_pe
        buy_ce = buy_ce & ~both
        buy_pe = buy_pe & ~both

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
