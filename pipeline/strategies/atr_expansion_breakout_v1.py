"""ATR Expansion Breakout — NIFTY 5-second index options strategy.

Thesis: On NIFTY, institutional block orders create 5-second bars with range
2.5x+ the recent 10-minute ATR baseline. When such a bar simultaneously
creates a new session high/low with volume surge, it signals that directional
order flow has overwhelmed ALL resting liquidity accumulated since open.
Enter on the expansion bar's close; the underlying order has not fully
executed and will continue pushing the index for the next 30-90 seconds.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Simple rolling mean — returns NaN for bars before warmup completes."""
    out = np.full(len(arr), np.nan)
    cumsum = np.cumsum(arr)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    out[window - 1:] = cumsum[window - 1:] / window
    return out


class Strategy(BaseStrategy):
    name = "atr_expansion_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min of open noise
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 240            # 20-min warmup (ATR_120 needs 120 bars + buffer)
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("tr_ratio_threshold", 2.5, 1.8, 3.5),
            TunableParam("close_position_min", 0.6, 0.5, 0.8),
            TunableParam("volume_ratio_min", 2.0, 1.5, 3.0),
            TunableParam("stop_pts", 5.0, 3.0, 8.0),
            TunableParam("target_pts", 8.0, 5.0, 15.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN before numpy conversion) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ──
        tr_ratio_threshold = params.get("tr_ratio_threshold", 2.5)
        close_position_min = params.get("close_position_min", 0.6)
        volume_ratio_min = params.get("volume_ratio_min", 2.0)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)

        # ── True Range ──
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        # ── ATR (120 bars = 10 min rolling mean) ──
        atr_120 = _rolling_mean(tr, 120)

        # ── TR ratio ──
        tr_ratio = np.zeros(n)
        valid_atr = atr_120 > 0
        tr_ratio[valid_atr] = tr[valid_atr] / atr_120[valid_atr]

        # ── Volume ratio (60 bars = 5 min rolling mean) ──
        vol_mean_60 = _rolling_mean(volume, 60)
        volume_ratio = np.zeros(n)
        valid_vol = vol_mean_60 > 0
        volume_ratio[valid_vol] = volume[valid_vol] / vol_mean_60[valid_vol]

        # ── Session VWAP, session high BEFORE current bar, session low BEFORE bar ──
        # session_high_prev[i] = max(high[j]) for j in [session_start .. i-1]
        # session_low_prev[i]  = min(low[j])  for j in [session_start .. i-1]
        vwap = np.zeros(n)
        session_high_prev = np.full(n, -np.inf)
        session_low_prev = np.full(n, np.inf)

        cum_pv = 0.0
        cum_vol_sum = 0.0
        running_high = -np.inf
        running_low = np.inf
        prev_day = -1

        for i in range(n):
            if day_id[i] != prev_day:
                # New trading day — reset session accumulators
                cum_pv = 0.0
                cum_vol_sum = 0.0
                running_high = -np.inf
                running_low = np.inf
                prev_day = day_id[i]
                # First bar of day: no previous session high/low in this session
                session_high_prev[i] = -np.inf
                session_low_prev[i] = np.inf
            else:
                # Store PREVIOUS bar's session running extremes (before this bar)
                session_high_prev[i] = running_high
                session_low_prev[i] = running_low

            # Update VWAP (use mid = (H+L+C)/3 as typical price)
            vol_i = volume[i] if volume[i] > 0 else 1.0
            mid = (high[i] + low[i] + close[i]) / 3.0
            cum_pv += mid * vol_i
            cum_vol_sum += vol_i
            vwap[i] = cum_pv / cum_vol_sum

            # Update running session extremes (include current bar)
            running_high = max(running_high, high[i])
            running_low = min(running_low, low[i])

        # ── Close position within bar ──
        bar_range = high - low
        close_position = np.full(n, 0.5)  # neutral default (midpoint)
        valid_range = bar_range > 0
        close_position[valid_range] = (
            (close[valid_range] - low[valid_range]) / bar_range[valid_range]
        )

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # ATR warmup guard (must have 120 bars of data)
        atr_ready = ~np.isnan(atr_120)

        # ── Bullish expansion breakout: new session high + bullish expansion bar ──
        buy_ce = (
            in_session
            & atr_ready
            & (tr_ratio > tr_ratio_threshold)
            & (high > session_high_prev)        # new session high on this bar
            & (close > open_)                   # bullish close
            & (close_position > close_position_min)   # closed in upper 40%
            & (volume_ratio > volume_ratio_min)
            & (close > vwap)                    # above VWAP
        )

        # ── Bearish expansion breakout: new session low + bearish expansion bar ──
        buy_pe = (
            in_session
            & atr_ready
            & (tr_ratio > tr_ratio_threshold)
            & (low < session_low_prev)          # new session low on this bar
            & (close < open_)                   # bearish close
            & ((1.0 - close_position) > close_position_min)  # closed in lower 40%
            & (volume_ratio > volume_ratio_min)
            & (close < vwap)                    # below VWAP
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,            # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
