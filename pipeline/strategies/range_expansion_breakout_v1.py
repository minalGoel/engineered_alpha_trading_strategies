"""range_expansion_breakout_v1 — 5-second NIFTY micro-consolidation breakout.

Mechanism: When NIFTY compresses into a 2-minute micro-consolidation (range < 0.08%),
limit orders equilibrate at both extremes. A single 5-second bar expanding to > 2.5x
the base-period average bar range signals an institutional order breaking through
the resting orderbook. The expansion bar consuming the base high/low triggers
automated stops, creating a self-reinforcing push for 60-90 seconds.
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_vwap(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                  volume: np.ndarray, day_id: np.ndarray, n: int) -> np.ndarray:
    """Compute session VWAP, restarting each day."""
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_vol = 0.0
    prev_day = -1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        typical = (high[i] + low[i] + close[i]) / 3.0
        vol = max(volume[i], 0.0)
        cum_pv += typical * vol
        cum_vol += vol
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else close[i]
    return vwap


class Strategy(BaseStrategy):
    name = "range_expansion_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — allow 24-bar base to form
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 48             # 4 minutes warmup (24 bars base + buffer)
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Base tightness: % of close that defines a tight consolidation
            TunableParam("tight_range_pct", 0.08, 0.04, 0.15),
            # Expansion multiplier: how many times larger than base avg bar range
            TunableParam("expansion_ratio", 2.5, 1.8, 4.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Extract spot arrays — forward-fill NaN before numpy conversion
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        tight_range_pct = params.get("tight_range_pct", 0.08) / 100.0
        expansion_ratio_thresh = params.get("expansion_ratio", 2.5)

        BASE = 24  # 2-minute base window at 5-second bars

        # Per-bar range
        bar_range = high - low

        # Rolling base metrics over preceding BASE bars (not including current bar)
        base_range_pct = np.zeros(n)
        avg_bar_range = np.ones(n)      # avoid div-by-zero
        prev_base_high = np.zeros(n)
        prev_base_low = np.zeros(n)
        avg_volume_24 = np.zeros(n)

        for i in range(BASE, n):
            h_window = high[i - BASE:i]
            l_window = low[i - BASE:i]
            br_window = bar_range[i - BASE:i]
            vol_window = volume[i - BASE:i]

            h_max = float(np.max(h_window))
            l_min = float(np.min(l_window))
            ref_close = close[i - 1] if close[i - 1] > 0 else 1.0

            base_range_pct[i] = (h_max - l_min) / ref_close
            avg_bar_range[i] = float(np.mean(br_window)) if np.mean(br_window) > 0 else 1.0
            prev_base_high[i] = h_max
            prev_base_low[i] = l_min
            avg_volume_24[i] = float(np.mean(vol_window))

        # Expansion ratio: current bar range vs base average
        expansion_ratio_arr = bar_range / avg_bar_range

        # Directional close quality within the expansion bar
        bar_range_safe = np.where(bar_range > 0, bar_range, 1.0)
        close_vs_bar = (close - low) / bar_range_safe  # 1.0 = closed at high, 0.0 = at low

        # Session VWAP (cumulative per day)
        vwap = _compute_vwap(close, high, low, volume, day_id, n)

        # VIX filter — join asof to align vix bars to spot bars
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime", strategy="backward"
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Filters ──────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        tight_base = base_range_pct < tight_range_pct
        is_expansion = expansion_ratio_arr >= expansion_ratio_thresh
        vol_spike = volume > (avg_volume_24 * 1.5)
        vix_ok = vix_close < 22.0

        # ── Entry signals ─────────────────────────────────────────────────────
        # Bullish: expansion above base high, strong close, above VWAP
        buy_ce = (
            in_session
            & tight_base
            & is_expansion
            & (close > prev_base_high)
            & (close_vs_bar > 0.75)
            & (close > vwap)
            & vol_spike
            & vix_ok
        )

        # Bearish: expansion below base low, weak close, below VWAP
        buy_pe = (
            in_session
            & tight_base
            & is_expansion
            & (close < prev_base_low)
            & (close_vs_bar < 0.25)
            & (close < vwap)
            & vol_spike
            & vix_ok
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 4.0),    # 4 pts: retrace into 2-min base invalidates thesis
            target_points=np.full(n, 7.0),  # 7 pts: ~60-70% of typical 15-25 spot pt continuation
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
