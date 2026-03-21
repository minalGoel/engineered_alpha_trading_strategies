"""VIX Bucket Strategy v1 — VIX-regime-adaptive directional strategy for NIFTY options.

Regime classification using India VIX:
  CALM     (VIX < 13):  Bollinger Band mean-reversion at 3-min resolution
  NORMAL   (13–18):     EMA(9-min) x EMA(21-min) crossover momentum
  ELEVATED (18–24):     Supertrend(10-min ATR, 2.5x) trend-following with volume surge
  CRISIS   (VIX >= 24): No trading — flow is too unpredictable

Each regime activates a different directional sub-strategy tuned to the microstructure
behaviour that India VIX implies on the NIFTY order book.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


# ─────────────────────────── helpers ────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average (Wilder-style, seed = first value)."""
    n = len(arr)
    out = np.zeros(n)
    k = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """RSI with Wilder's smoothing. Returns 50.0 for warmup bars."""
    n = len(close)
    out = np.full(n, 50.0)
    if period >= n:
        return out
    gains = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        d = close[i] - close[i - 1]
        if d > 0:
            gains[i] = d
        else:
            losses[i] = -d
    avg_g = np.mean(gains[1 : period + 1])
    avg_l = np.mean(losses[1 : period + 1])
    rs = avg_g / avg_l if avg_l > 0.0 else 100.0
    out[period] = 100.0 - 100.0 / (1.0 + rs) if avg_l > 0.0 else 100.0
    for i in range(period + 1, n):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs = avg_g / avg_l if avg_l > 0.0 else 100.0
        out[i] = 100.0 - 100.0 / (1.0 + rs) if avg_l > 0.0 else 100.0
    return out


def _rolling_mean_std(arr: np.ndarray, window: int):
    """Rolling mean and std. Pre-warmup bars seeded with first valid value / 0."""
    n = len(arr)
    means = np.full(n, arr[0] if n > 0 else 0.0)
    stds = np.zeros(n)
    for i in range(window - 1, n):
        w = arr[i - window + 1 : i + 1]
        means[i] = np.mean(w)
        stds[i] = np.std(w)
    return means, stds


def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR using Wilder's smoothing (same as Supertrend standard)."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.zeros(n)
    if period < n:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _supertrend(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int,
    multiplier: float,
):
    """Supertrend direction bands. Returns (is_bullish, is_bearish) bool arrays."""
    atr = _wilder_atr(high, low, close, period)
    n = len(close)
    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr
    upper = upper_basic.copy()
    lower = lower_basic.copy()
    direction = np.ones(n)  # 1=bull, -1=bear
    for i in range(1, n):
        # Upper band — only tighten (or reset on break)
        if upper_basic[i] < upper[i - 1] or close[i - 1] > upper[i - 1]:
            upper[i] = upper_basic[i]
        else:
            upper[i] = upper[i - 1]
        # Lower band — only widen (or reset on break)
        if lower_basic[i] > lower[i - 1] or close[i - 1] < lower[i - 1]:
            lower[i] = lower_basic[i]
        else:
            lower[i] = lower[i - 1]
        # Flip direction
        if close[i] > upper[i - 1]:
            direction[i] = 1
        elif close[i] < lower[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]
    return direction == 1, direction == -1


def _vwap_daily(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Cumulative intraday VWAP, reset per day_id."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_vol = 0.0
    prev_day = -999999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        vol_i = max(float(volume[i]), 1.0)
        cum_pv += close[i] * vol_i
        cum_vol += vol_i
        vwap[i] = cum_pv / cum_vol
    return vwap


def _rolling_vol_mean(volume: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean of volume, seeded with current value before warmup."""
    n = len(volume)
    out = volume.copy().astype(np.float64)
    for i in range(window, n):
        out[i] = np.mean(volume[i - window : i])
    return out


# ─────────────────────────── strategy ───────────────────────────────────────

class Strategy(BaseStrategy):
    name = "vix_bucket_strategy_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST
    session_end_minutes = 910     # 15:10 IST
    max_trades_per_day = 8
    max_lookback = 360            # 30 min — needed for EMA(252) + Supertrend(120) warmup

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_calm",      13.0, 10.0, 16.0),
            TunableParam("vix_elevated",  18.0, 15.0, 22.0),
            TunableParam("vix_crisis",    24.0, 20.0, 30.0),
            TunableParam("rsi_oversold",  30.0, 20.0, 40.0),
            TunableParam("rsi_overbought", 70.0, 60.0, 80.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN in Polars before numpy) ──────────
        close  = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high   = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low    = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        day_id = spot_df["day_id"].fill_null(0).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────────
        vix_calm      = params.get("vix_calm",      13.0)
        vix_elevated  = params.get("vix_elevated",  18.0)
        vix_crisis    = params.get("vix_crisis",    24.0)
        rsi_oversold  = params.get("rsi_oversold",  30.0)
        rsi_overbought = params.get("rsi_overbought", 70.0)

        # ── VIX level (aligned to spot bars via backward join) ────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── VIX regime masks ───────────────────────────────────────────────────
        is_calm     = vix_close < vix_calm
        is_normal   = (vix_close >= vix_calm)     & (vix_close < vix_elevated)
        is_elevated = (vix_close >= vix_elevated) & (vix_close < vix_crisis)
        # is_crisis = vix_close >= vix_crisis → no trading (signals stay False)

        # ── VWAP (universal directional context, all regimes) ─────────────────
        vwap = _vwap_daily(close, volume, day_id)

        # ── CALM: Bollinger Band(36) + RSI(24) ────────────────────────────────
        bb_mean, bb_std = _rolling_mean_std(close, 36)
        bb_std = np.where(bb_std < 0.01, 0.01, bb_std)
        bb_upper = bb_mean + 1.5 * bb_std
        bb_lower = bb_mean - 1.5 * bb_std

        rsi_24 = _rsi(close, 24)

        # Single-bar uptick / downtick confirmation
        ret_1 = np.zeros(n)
        ret_1[1:] = close[1:] - close[:-1]

        calm_buy_ce = (
            is_calm
            & (close < bb_lower)
            & (close < vwap)
            & (rsi_24 < rsi_oversold)
            & (ret_1 > 0)
        )
        calm_buy_pe = (
            is_calm
            & (close > bb_upper)
            & (close > vwap)
            & (rsi_24 > rsi_overbought)
            & (ret_1 < 0)
        )

        # ── NORMAL: EMA(108) x EMA(252) crossover ─────────────────────────────
        # 108 bars = 9 min; 252 bars = 21 min — same calendar time as original
        ema_fast = _ema(close, 108)
        ema_slow = _ema(close, 252)

        # Detect crossover: fast crosses slow on this bar vs previous bar
        ema_fast_prev = np.empty_like(ema_fast)
        ema_fast_prev[0] = ema_fast[0]
        ema_fast_prev[1:] = ema_fast[:-1]

        ema_slow_prev = np.empty_like(ema_slow)
        ema_slow_prev[0] = ema_slow[0]
        ema_slow_prev[1:] = ema_slow[:-1]

        cross_bull = (ema_fast > ema_slow) & (ema_fast_prev <= ema_slow_prev)
        cross_bear = (ema_fast < ema_slow) & (ema_fast_prev >= ema_slow_prev)

        normal_buy_ce = is_normal & cross_bull & (close > vwap)
        normal_buy_pe = is_normal & cross_bear & (close < vwap)

        # ── ELEVATED: Supertrend(120, 2.5) + volume surge ─────────────────────
        # 120 bars = 10 min; multiplier 2.5 (vs original 4.0 on 1-min bars)
        st_bull, st_bear = _supertrend(high, low, close, 120, 2.5)

        vol_ma_20 = _rolling_vol_mean(volume, 20)
        vol_surge = volume > 1.5 * vol_ma_20

        elevated_buy_ce = is_elevated & st_bull & (close > vwap) & vol_surge
        elevated_buy_pe = is_elevated & st_bear & (close < vwap) & vol_surge

        # ── Session filter ─────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Combined signals ───────────────────────────────────────────────────
        buy_ce = in_session & (calm_buy_ce | normal_buy_ce | elevated_buy_ce)
        buy_pe = in_session & (calm_buy_pe | normal_buy_pe | elevated_buy_pe)

        # ── Regime-dependent stops / targets (option premium points) ──────────
        # Default to Normal regime params; overwrite per active regime
        stop_pts   = np.full(n, 4.0)
        target_pts = np.full(n, 6.0)
        # Calm: tighter stop/target (small moves, fast reversion)
        stop_pts   = np.where(is_calm, 3.0, stop_pts)
        target_pts = np.where(is_calm, 4.0, target_pts)
        # Elevated: wider stop/target (larger moves, dealer hedging momentum)
        stop_pts   = np.where(is_elevated, 6.0, stop_pts)
        target_pts = np.where(is_elevated, 9.0, target_pts)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_pts,
            target_points=target_pts,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
