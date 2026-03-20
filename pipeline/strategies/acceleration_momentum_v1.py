"""Acceleration Momentum Strategy (acceleration_momentum_v1)

Trades NIFTY ATM options based on the second derivative of price momentum.
When NIFTY's 1-minute ROC is itself accelerating (ROC_accel > 0 for 3+ bars
and roc_accel_slope > 0), institutional order flow is in the "jerk phase" —
algorithmic participation is growing, not just sustained. We enter during this
early acceleration window before the momentum burst peaks.

Converted from: trading_strategies/unique_strategies_all/Strategy_167.json
Original: equity stocks 1-min, 10-30 min hold, ROC(10) + ROC of ROC.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _linreg_slope(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling linear regression slope over `window` bars."""
    n = len(arr)
    result = np.zeros(n)
    if window < 2:
        return result
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    ss_xx = ((x - x_mean) ** 2).sum()
    if ss_xx == 0:
        return result
    for i in range(window - 1, n):
        y = arr[i - window + 1: i + 1]
        y_mean = y.mean()
        result[i] = ((x - x_mean) * (y - y_mean)).sum() / ss_xx
    return result


class Strategy(BaseStrategy):
    name = "acceleration_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min pre-open noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 60             # 5-min warmup: enough for ROC_12 + vol_sma20

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum ROC_accel to count as "accelerating" (in % terms)
            TunableParam("accel_threshold", 0.03, 0.01, 0.10),
            # VWAP trend filter: 1 = on (must be above VWAP for CE), 0 = off
            TunableParam("vwap_filter_on", 1.0, 0.0, 1.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        accel_threshold = params.get("accel_threshold", 0.03)
        vwap_filter_on = params.get("vwap_filter_on", 1.0) >= 0.5

        # ── Pull arrays (forward-fill NaN in Polars before numpy) ──────────
        close = (
            spot_df["close"].fill_null(strategy="forward").fill_null(0.0).to_numpy().astype(float)
        )
        high = (
            spot_df["high"].fill_null(strategy="forward").fill_null(0.0).to_numpy().astype(float)
        )
        low = (
            spot_df["low"].fill_null(strategy="forward").fill_null(0.0).to_numpy().astype(float)
        )
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── ROC_12: 1-minute momentum (12 bars × 5s = 60s) ─────────────────
        # Original: ROC(10) on 1-min = 10-min; compressed to 1-min because
        # our hold is 30-90s — we ride the 1-min momentum, not 10-min context.
        roc_12 = np.zeros(n)
        for i in range(12, n):
            if close[i - 12] > 0:
                roc_12[i] = (close[i] - close[i - 12]) / close[i - 12] * 100.0

        # ── ROC_accel: 30-second change in 1-min momentum ───────────────────
        # roc_12[t] - roc_12[t-6]: positive means momentum is growing
        # 6 bars = 30s ≈ 1/3 of our 90s max hold (original ratio maintained)
        roc_accel = np.zeros(n)
        for i in range(6, n):
            roc_accel[i] = roc_12[i] - roc_12[i - 6]

        # ── ROC_accel_slope: 15-second trend of acceleration (3 bars) ───────
        # Confirms acceleration is rising, not a single-bar spike
        roc_accel_slope = _linreg_slope(roc_accel, 3)

        # ── Session VWAP (cumulative from open, reset each day) ─────────────
        vwap = np.zeros(n)
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = int(day_id[i])
            tp = (high[i] + low[i] + close[i]) / 3.0
            v = max(volume[i], 0.0)
            cum_tp_vol += tp * v
            cum_vol += v
            vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0 else close[i]

        # ── Volume SMA(20): 100-second volume baseline ───────────────────────
        vol_sma20 = np.zeros(n)
        for i in range(20, n):
            vol_sma20[i] = np.mean(volume[i - 20: i])
        # Early bars: use raw volume so we don't suppress all signals
        vol_sma20[:20] = volume[:20]

        # ── Consecutive acceleration bar counter (3-bar confirmation) ────────
        # consec_bull[i] = how many consecutive bars roc_accel > threshold
        # consec_bear[i] = how many consecutive bars roc_accel < -threshold
        consec_bull = np.zeros(n, dtype=np.int32)
        consec_bear = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            consec_bull[i] = consec_bull[i - 1] + 1 if roc_accel[i] > accel_threshold else 0
            consec_bear[i] = consec_bear[i - 1] + 1 if roc_accel[i] < -accel_threshold else 0

        # ── Session filter ───────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── VWAP alignment filters ───────────────────────────────────────────
        if vwap_filter_on:
            vwap_bull = close > vwap
            vwap_bear = close < vwap
        else:
            vwap_bull = np.ones(n, dtype=bool)
            vwap_bear = np.ones(n, dtype=bool)

        # ── Volume filter ────────────────────────────────────────────────────
        vol_ok = volume > vol_sma20

        # ── Entry signals ────────────────────────────────────────────────────
        # buy_ce: bullish — 1-min momentum positive + accelerating for 15s + slope rising
        buy_ce = (
            in_session
            & (roc_12 > 0)
            & (consec_bull >= 3)
            & (roc_accel_slope > 0)
            & vwap_bull
            & vol_ok
        )

        # buy_pe: bearish — 1-min momentum negative + decelerating/accelerating down
        buy_pe = (
            in_session
            & (roc_12 < 0)
            & (consec_bear >= 3)
            & (roc_accel_slope < 0)
            & vwap_bear
            & vol_ok
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # 3 pts stop: ~6 NIFTY spot pts; acceleration signal wrong if retraces this fast
            stop_points=np.full(n, 3.0),
            # 6 pts target: ~12 NIFTY spot pts; ~50% of expected 15-25 pt continuation (1:2 R:R)
            target_points=np.full(n, 6.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
