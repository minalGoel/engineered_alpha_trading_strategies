"""
intraday_trend_channel_v1 — NIFTY Linear Regression Channel Pullback

Mechanism:
    On NIFTY, institutional TWAP algorithms executing large directional baskets create
    smooth micro-trends at 3-minute resolution. When a 3-minute linear regression channel
    carries a positive slope and price pulls back to the lower 2-sigma boundary, it signals
    a temporary deceleration within an ongoing buying program. The regression midline acts
    as a dynamic magnet: VWAP-tracking algorithms and delta-hedging market makers re-enter
    when price deviates from the regression fit, creating a 10-15 NIFTY spot point snap-back
    toward the midline within 30-90 seconds.

Original: Strategy_170.json — 60-bar 1-min linreg channel on NIFTY200 stocks, hold 15-45 min
Adapted:  36-bar 5s linreg channel on NIFTY index, hold 15-90 s
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_linreg(close: np.ndarray, n: int):
    """Compute rolling linear regression over last n bars.

    Returns:
        linreg_val  — fitted value at the current (last) bar of each window
        slope       — OLS beta (points per 5-second bar)
        res_std     — std dev of residuals (channel half-width denominator)
    """
    T = len(close)
    linreg_val = np.full(T, np.nan)
    slope = np.zeros(T)
    res_std = np.full(T, 1.0)

    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_ss = ((x - x_mean) ** 2).sum()  # sum of squared deviations

    for i in range(n - 1, T):
        y = close[i - n + 1: i + 1]
        y_mean = y.mean()
        xy_cov = ((x - x_mean) * (y - y_mean)).sum()
        b = xy_cov / x_ss
        a = y_mean - b * x_mean
        linreg_val[i] = a + b * (n - 1)   # fitted value at last (current) bar
        slope[i] = b
        residuals = y - (a + b * x)
        s = float(np.std(residuals))
        res_std[i] = s if s > 0.01 else 0.01

    return linreg_val, slope, res_std


class Strategy(BaseStrategy):
    name = "intraday_trend_channel_v1"
    underlying = "NIFTY"
    session_start_minutes = 560    # 09:20 IST
    session_end_minutes = 925      # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 72              # 6-min warmup (36 regression + 12 slope stability + buffer)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum slope (pts/bar) to qualify as a valid trend channel
            TunableParam("slope_threshold", 0.10, 0.02, 0.50),
            # Channel position thresholds (0-1 scale)
            TunableParam("channel_pos_low", 0.15, 0.05, 0.30),
            TunableParam("channel_pos_high", 0.85, 0.70, 0.95),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        slope_thr = float(params.get("slope_threshold", 0.10))
        ch_pos_low = float(params.get("channel_pos_low", 0.15))
        ch_pos_high = float(params.get("channel_pos_high", 0.85))

        # ── Rolling 36-bar (3-minute) linear regression ──────────────────────
        LR_N = 36
        linreg_val, lr_slope, lr_std = _rolling_linreg(close, LR_N)

        # Channel boundaries (2-sigma bands)
        linreg_upper = linreg_val + 2.0 * lr_std
        linreg_lower = linreg_val - 2.0 * lr_std

        # Normalized channel position [0, 1]; guard against zero-width channel
        channel_width = linreg_upper - linreg_lower
        channel_width = np.where(channel_width < 0.1, 0.1, channel_width)
        channel_pos = (close - linreg_lower) / channel_width

        # ── Slope stability: all last 12 bars (1 min) must have same slope sign ──
        STABLE_N = 12
        slope_up_stable = np.zeros(n, dtype=bool)
        slope_dn_stable = np.zeros(n, dtype=bool)
        for i in range(STABLE_N - 1, n):
            window = lr_slope[i - STABLE_N + 1: i + 1]
            slope_up_stable[i] = bool(np.all(window > 0))
            slope_dn_stable[i] = bool(np.all(window < 0))

        # ── Valid bars (regression warmed up) ────────────────────────────────
        valid = ~np.isnan(linreg_val)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes) &
            (time_min < self.session_end_minutes)
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        # Buy CE: uptrend channel, price near lower 2-sigma boundary (pullback to support)
        buy_ce = (
            in_session &
            valid &
            slope_up_stable &
            (lr_slope > slope_thr) &
            (channel_pos < ch_pos_low) &
            (close >= linreg_lower)           # not broken below channel
        )

        # Buy PE: downtrend channel, price near upper 2-sigma boundary (rejection from resistance)
        buy_pe = (
            in_session &
            valid &
            slope_dn_stable &
            (lr_slope < -slope_thr) &
            (channel_pos > ch_pos_high) &
            (close <= linreg_upper)           # not broken above channel
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 3 pts (~6 NIFTY spot points) — channel breakdown signal
            # Target: 5 pts (~10 NIFTY spot points) — midline snap-back
            stop_points=np.full(n, 3.0),
            target_points=np.full(n, 5.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,            # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
