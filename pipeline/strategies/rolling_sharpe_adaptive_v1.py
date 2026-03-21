"""rolling_sharpe_adaptive_v1 — EMA/VWAP momentum with rolling signal quality filter.

Converted from: trading_strategies/unique_strategies_all/Strategy_313.json
Original: NIFTY 50 constituent stocks, 1-min bars, EMA(9/21) + VWAP + rolling trade Sharpe.

Adaptation:
- Universe: NIFTY index (5-second bars)
- EMA periods compressed to 2-min / 5-min (not 12x scaled) to match 30-90s hold time
- Rolling Sharpe of trade outcomes → Rolling signal accuracy of EMA direction over last 3 min
  (compute() has no live PnL; rolling bar-direction accuracy preserves the anti-chop regime filter)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average; returns NaN until period-1 bars have elapsed."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < period:
        return out
    alpha = 2.0 / (period + 1.0)
    out[period - 1] = np.mean(arr[:period])
    for i in range(period, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _compute_session_vwap(
    close: np.ndarray, volume: np.ndarray, day_id: np.ndarray
) -> np.ndarray:
    """Cumulative session VWAP, resets each new day_id."""
    n = len(close)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_v += volume[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


def _compute_signal_quality(
    ema_diff_sign: np.ndarray, bar_ret_sign: np.ndarray, lookback: int
) -> np.ndarray:
    """Rolling fraction of bars where EMA direction correctly predicted next bar's direction.

    ema_diff_sign[t] = sign(ema_fast[t] - ema_slow[t]) — the 'signal' at bar t
    bar_ret_sign[t]  = sign(close[t] - close[t-1])     — what actually happened at bar t
    We want: was ema_diff_sign[t-1] == bar_ret_sign[t]? (did last bar's signal predict this bar?)
    neutral = 0.5 (no regime information)
    """
    n = len(ema_diff_sign)
    quality = np.full(n, 0.5)
    # correct[t] = 1 if ema_diff_sign[t-1] predicted bar_ret_sign[t], else 0, NaN if no signal
    correct = np.full(n, np.nan)
    for t in range(1, n):
        sig = ema_diff_sign[t - 1]
        if sig != 0.0:
            correct[t] = 1.0 if sig == bar_ret_sign[t] else 0.0

    for i in range(lookback, n):
        window = correct[i - lookback : i]
        valid = window[~np.isnan(window)]
        if len(valid) >= lookback // 2:
            quality[i] = float(np.mean(valid))
    return quality


class Strategy(BaseStrategy):
    name = "rolling_sharpe_adaptive_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup (EMA-60 needs 60 bars + signal quality 36 bars)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum rolling signal accuracy (regime quality gate)
            TunableParam("signal_quality_threshold", 0.52, 0.48, 0.62),
            # Minimum |VWAP distance| in bps to confirm directional bias
            TunableParam("vwap_distance_bps", 5.0, 2.0, 15.0),
            TunableParam("stop_pts", 5.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN in Polars before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].to_numpy().astype(int)
        time_min = spot_df["time_minutes"].to_numpy().astype(int)

        # ── Parameters ──
        sq_thresh = float(params.get("signal_quality_threshold", 0.52))
        vwap_dist_bps = float(params.get("vwap_distance_bps", 5.0))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # ── Indicators ──
        # EMA(24) = 2-min fast; EMA(60) = 5-min slow (not 12x; calibrated to 30-90s hold)
        ema_fast = _compute_ema(close, 24)
        ema_slow = _compute_ema(close, 60)

        # Session VWAP (cumulative, reset per day)
        vwap = _compute_session_vwap(close, volume, day_id)

        # Volume SMA (30 bars = 2.5 min)
        vol_sma = np.zeros(n)
        for i in range(30, n):
            vol_sma[i] = np.mean(volume[i - 30 : i])

        # EMA diff sign (1 = bullish, -1 = bearish, 0 = undefined / NaN)
        ema_finite = np.isfinite(ema_fast) & np.isfinite(ema_slow)
        ema_diff_sign = np.where(ema_finite, np.sign(ema_fast - ema_slow), 0.0)

        # Bar return sign (1 = up, -1 = down, 0 = flat)
        bar_ret_sign = np.zeros(n)
        bar_ret_sign[1:] = np.sign(close[1:] - close[:-1])

        # Rolling signal quality over 36 bars (3 min)
        signal_quality = _compute_signal_quality(ema_diff_sign, bar_ret_sign, lookback=36)

        # VWAP distance in bps (positive = above VWAP)
        vwap_dist = np.where(vwap > 0.0, (close - vwap) / vwap * 10000.0, 0.0)

        # ── Session mask ──
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )

        # ── Regime and confirmation masks ──
        good_regime = signal_quality > sq_thresh
        has_volume = volume > vol_sma
        has_ema = ema_finite

        # ── Entry signals ──
        # Buy CE: bullish EMA alignment + above VWAP + good trending regime + volume surge
        buy_ce = (
            in_session
            & has_ema
            & good_regime
            & has_volume
            & (ema_fast > ema_slow)
            & (vwap_dist > vwap_dist_bps)
        )

        # Buy PE: bearish EMA alignment + below VWAP + good trending regime + volume surge
        buy_pe = (
            in_session
            & has_ema
            & good_regime
            & has_volume
            & (ema_fast < ema_slow)
            & (vwap_dist < -vwap_dist_bps)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
