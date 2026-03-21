"""
vol_of_vol_signal_v1 — Volatility-of-Volatility Regime Classifier
===================================================================
Mechanism:
  On NIFTY, the 2-minute ATR's own instability (vol-of-vol) signals whether
  the index is in a stable-drift or regime-transition state. Low vol-of-vol
  z-score → stable regime → RSI extremes revert within 30-60s. Rising
  vol-of-vol z-score → institutional TWAP or macro trigger disrupting
  equilibrium → EMA crossover confirms directional flow shift lasting 60-90s.
  Very high vol-of-vol → chaotic, no trades.

Converted from: trading_strategies/unique_strategies_all/Strategy_220.json
Original: 1-min equity, ATR(14)/vol-of-vol, 10-45 min hold.
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed ATR."""
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = hl if hl >= hc and hl >= lc else (hc if hc >= lc else lc)
    atr = np.zeros(n)
    if period <= n:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation (population std)."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window - 1, n):
        result[i] = np.std(arr[i - window + 1: i + 1])
    return result


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gain[1: period + 1])
    avg_loss[period] = np.mean(loss[1: period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (initialised at first bar)."""
    n = len(close)
    ema = np.zeros(n)
    k = 2.0 / (period + 1)
    ema[0] = close[0]
    for i in range(1, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    """Vol-of-vol regime classifier for 5-second NIFTY options."""

    name = "vol_of_vol_signal_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — need 174 bars (14.5 min) warmup after 09:15
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 180            # 15 min buffer (atr_24=24 + vov_30=30 + vov_z_window=120 = 174 bars)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vov_stable_thresh", 0.5, 0.0, 1.2),
            TunableParam("vov_transition_thresh", 2.0, 1.2, 3.0),
            TunableParam("rsi_oversold", 32.0, 22.0, 40.0),
            TunableParam("rsi_overbought", 68.0, 60.0, 78.0),
            TunableParam("stop_stable", 3.0, 2.0, 5.0),
            TunableParam("target_stable", 5.0, 3.5, 8.0),
            TunableParam("stop_transition", 4.0, 3.0, 7.0),
            TunableParam("target_transition", 7.0, 5.0, 12.0),
        ]

    def compute(self, spot_df: pl.DataFrame, option_df: pl.DataFrame, vix_df: pl.DataFrame, params: dict) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill NaN before converting to numpy
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # Replace any remaining NaN (e.g., at index 0) with neutral values
        high = np.where(np.isnan(high), 0.0, high)
        low = np.where(np.isnan(low), 0.0, low)
        close = np.where(np.isnan(close), 0.0, close)

        # --- Parameters ---
        vov_stable = float(params.get("vov_stable_thresh", 0.5))
        vov_trans = float(params.get("vov_transition_thresh", 2.0))
        rsi_os = float(params.get("rsi_oversold", 32.0))
        rsi_ob = float(params.get("rsi_overbought", 68.0))
        stop_s = float(params.get("stop_stable", 3.0))
        tgt_s = float(params.get("target_stable", 5.0))
        stop_t = float(params.get("stop_transition", 4.0))
        tgt_t = float(params.get("target_transition", 7.0))

        # --- Compute indicators ---

        # ATR(24): 2-minute true range — captures current micro-volatility
        atr_24 = _compute_atr(high, low, close, 24)

        # Vol-of-vol: rolling std of ATR over 30 bars (2.5 min)
        vov_30 = _rolling_std(atr_24, 30)

        # Z-score of vol-of-vol using 10-minute rolling baseline
        vov_mean = np.zeros(n)
        vov_std_arr = np.ones(n)
        for i in range(120, n):
            window = vov_30[i - 120: i]
            vov_mean[i] = np.mean(window)
            s = np.std(window)
            vov_std_arr[i] = s if s > 1e-8 else 1e-8
        vov_z = (vov_30 - vov_mean) / vov_std_arr

        # RSI(24): 2-minute RSI for stable-regime mean-reversion signals
        rsi_24 = _compute_rsi(close, 24)

        # EMA(12) and EMA(36): 1-min / 3-min for transitioning-regime breakout
        ema_12 = _compute_ema(close, 12)
        ema_36 = _compute_ema(close, 36)

        # --- Regime classification ---
        # Require warmup: atr_24(24) + vov_30(30) + vov_z(120) = 174 bars minimum
        warmed = np.arange(n) >= 174

        stable_regime = warmed & (vov_z < vov_stable)
        transitioning_regime = warmed & (vov_z >= vov_stable) & (vov_z < vov_trans)
        # volatile_regime: vov_z >= vov_trans → no trades

        # Session filter
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # --- Entry signals ---

        # Stable regime: mean reversion on RSI extremes
        buy_ce_stable = in_session & stable_regime & (rsi_24 < rsi_os)
        buy_pe_stable = in_session & stable_regime & (rsi_24 > rsi_ob)

        # Transitioning regime: breakout on EMA crossover
        ema_cross_up = np.zeros(n, dtype=bool)
        ema_cross_down = np.zeros(n, dtype=bool)
        ema_cross_up[1:] = (ema_12[1:] > ema_36[1:]) & (ema_12[:-1] <= ema_36[:-1])
        ema_cross_down[1:] = (ema_12[1:] < ema_36[1:]) & (ema_12[:-1] >= ema_36[:-1])

        buy_ce_trans = in_session & transitioning_regime & ema_cross_up
        buy_pe_trans = in_session & transitioning_regime & ema_cross_down

        buy_ce = buy_ce_stable | buy_ce_trans
        buy_pe = buy_pe_stable | buy_pe_trans

        # Regime-adaptive stops and targets
        # Stable: tighter stop (3pts) smaller target (5pts) — short reversion bounce
        # Transitioning: wider stop (4pts) larger target (7pts) — directional move continuation
        stop_arr = np.where(
            stable_regime, stop_s,
            np.where(transitioning_regime, stop_t, 0.0)
        )
        target_arr = np.where(
            stable_regime, tgt_s,
            np.where(transitioning_regime, tgt_t, 0.0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_arr,
            target_points=target_arr,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
