"""Multi-EMA Ribbon Squeeze — 5-second NIFTY index options strategy.

When NIFTY's 60s, 3-min, and 5-min EMAs compress into a tight band (~0.06% range),
all near-term participants are in equilibrium. A price break through the ribbon on
above-average volume signals directional institutional flow resuming.
Entry within 15-30 seconds of expansion start, before 1-min systems react.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA; returns array with leading NaN for warmup bars."""
    n = len(arr)
    result = np.full(n, np.nan)
    if period > n:
        return result
    alpha = 2.0 / (period + 1)
    # Seed with simple mean of first `period` bars
    result[period - 1] = np.mean(arr[:period])
    for i in range(period, n):
        result[i] = alpha * arr[i] + (1.0 - alpha) * result[i - 1]
    return result


def _ffill_nan(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """Forward-fill NaN values; any leading NaN filled with fill_value."""
    result = arr.copy()
    last_valid = fill_value
    for i in range(len(result)):
        if np.isnan(result[i]):
            result[i] = last_valid
        else:
            last_valid = result[i]
    return result


class Strategy(BaseStrategy):
    name = "multi_ema_ribbon_squeeze"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip first-5-min noise
    session_end_minutes = 920     # 15:20 IST — avoid EOD illiquidity
    max_lookback = 120            # 600s = 10 min warmup (covers ema_60 + buffer)
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Ribbon squeeze threshold as fraction of EMA level (e.g. 0.0006 = 0.06%)
            TunableParam("squeeze_threshold", 0.0006, 0.0003, 0.0012),
            # Volume must be at least this multiple of 20-bar average
            TunableParam("vol_ratio_min", 1.1, 0.8, 1.6),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot data (forward-fill nulls in Polars before numpy) ──
        close = (
            spot_df["close"]
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .to_numpy()
            .astype(np.float64)
        )
        volume = (
            spot_df["volume"]
            .fill_null(0)
            .to_numpy()
            .astype(np.float64)
        )
        time_min = spot_df["time_minutes"].to_numpy()

        squeeze_threshold = params.get("squeeze_threshold", 0.0006)
        vol_ratio_min = params.get("vol_ratio_min", 1.1)

        # ── EMA ribbon: 12 bars (60s), 36 bars (3 min), 60 bars (5 min) ──
        ema_fast = _ffill_nan(_ema(close, 12), fill_value=close[0] if n > 0 else 0.0)
        ema_mid  = _ffill_nan(_ema(close, 36), fill_value=close[0] if n > 0 else 0.0)
        ema_slow = _ffill_nan(_ema(close, 60), fill_value=close[0] if n > 0 else 0.0)

        # ── Ribbon squeeze: all EMAs within squeeze_threshold fraction of each other ──
        ema_max = np.maximum(ema_fast, np.maximum(ema_mid, ema_slow))
        ema_min = np.minimum(ema_fast, np.minimum(ema_mid, ema_slow))
        # Guard against zero division (shouldn't happen on NIFTY data)
        ema_min_safe = np.where(ema_min > 0, ema_min, 1.0)
        ribbon_spread = (ema_max - ema_min) / ema_min_safe
        squeeze = ribbon_spread < squeeze_threshold

        # Require squeeze on PREVIOUS bar (lag by 1) — prevents same-bar noise entries
        squeeze_prev = np.empty(n, dtype=bool)
        squeeze_prev[0] = False
        squeeze_prev[1:] = squeeze[:-1]

        # ── Volume: ratio to 20-bar rolling mean ──
        vol_sma = np.full(n, 1.0)
        for i in range(20, n):
            window = volume[i - 20:i]
            mean_v = np.mean(window)
            vol_sma[i] = mean_v if mean_v > 0 else 1.0
        # Fill warmup period with the first valid value
        if n > 20:
            vol_sma[:20] = vol_sma[20]

        vol_ok = volume > (vol_ratio_min * vol_sma)

        # ── Price position relative to ribbon ──
        above_ribbon = close > ema_max   # bullish breakout
        below_ribbon = close < ema_min   # bearish breakdown

        # ── Session and warmup filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed_up = np.arange(n) >= 60   # need at least 60 bars (5 min) for ema_slow

        # ── Entry signals ──
        buy_ce = in_session & warmed_up & squeeze_prev & above_ribbon & vol_ok
        buy_pe = in_session & warmed_up & squeeze_prev & below_ribbon & vol_ok

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # 4 option pts stop: ~8 spot pts — valid breakout should not retrace this far
            stop_points=np.full(n, 5),
            # 7 option pts target: ~14 spot pts = ~50% of first expansion leg (20-30 spot pts)
            target_points=np.full(n, 8),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds = 18 × 5s bars
            max_trades_per_day=self.max_trades_per_day,
        )
