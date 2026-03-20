"""
momentum_score_composite_v1 — Multi-indicator composite momentum score on NIFTY 5-second bars.

Mechanism: On NIFTY, when 5 independent momentum signals (2-min RSI velocity, MACD histogram
acceleration, 1-min ROC, ADX directional bias, volume-price direction) simultaneously align,
multiple algorithmic strategy families are all positioned in the same direction. This
multi-indicator confluence at the 1-3 minute timescale signals institutional order flow has
reached critical mass, with remaining unfilled volume driving continuation for 30-90 seconds.

Original: composite momentum score on NIFTY200 stocks, 1-min bars, 15-45 min hold.
Converted: same composite logic at 5s resolution on NIFTY index, 30-90s hold.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


# ─────────────────────────────────────────────
# Helper indicators (all operate on numpy arrays)
# ─────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI. Returns 50 for bars before warmup completes."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 2:
        return rsi
    delta = np.empty(n)
    delta[0] = 0.0
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    # Seed with SMA over first period
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gain[1: period + 1])
    avg_loss[period] = np.mean(loss[1: period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder ATR."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.zeros(n)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average (expanding until period reached)."""
    n = len(arr)
    out = np.zeros(n)
    cumsum = np.cumsum(arr)
    for i in range(n):
        if i < period:
            out[i] = cumsum[i] / (i + 1)
        else:
            out[i] = (cumsum[i] - cumsum[i - period]) / period
    return out


def _adx_dir(
    high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (adx, plus_di - minus_di direction sign × capped ADX strength)."""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        h_diff = high[i] - high[i - 1]
        l_diff = low[i - 1] - low[i]
        plus_dm[i] = h_diff if (h_diff > l_diff and h_diff > 0) else 0.0
        minus_dm[i] = l_diff if (l_diff > h_diff and l_diff > 0) else 0.0
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    # Wilder smoothing
    atr_s = np.zeros(n)
    plus_s = np.zeros(n)
    minus_s = np.zeros(n)
    if n > period:
        atr_s[period] = np.sum(tr[1: period + 1])
        plus_s[period] = np.sum(plus_dm[1: period + 1])
        minus_s[period] = np.sum(minus_dm[1: period + 1])
        for i in range(period + 1, n):
            atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
            plus_s[i] = plus_s[i - 1] - plus_s[i - 1] / period + plus_dm[i]
            minus_s[i] = minus_s[i - 1] - minus_s[i - 1] / period + minus_dm[i]
    safe_atr = np.where(atr_s > 0, atr_s, 1.0)
    plus_di = 100.0 * plus_s / safe_atr
    minus_di = 100.0 * minus_s / safe_atr
    dx = np.where(
        (plus_di + minus_di) > 0,
        100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di),
        0.0,
    )
    adx = np.zeros(n)
    start2 = 2 * period
    if n > start2:
        adx[start2] = np.mean(dx[period: start2 + 1])
        for i in range(start2 + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    dir_sign = np.sign(plus_di - minus_di)
    adx_score = dir_sign * np.minimum(adx, 40.0) / 2.0  # range ≈ [-20, +20]
    return adx_score


# ─────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────

class Strategy(BaseStrategy):
    """
    Composite momentum score: 5 sub-signals each normalised to [-20, +20] and summed.
    Entry when composite > threshold (bullish) or < -threshold (bearish) with 3-bar
    persistence (15s) and VWAP alignment. Target NIFTY ATM options, 30-90s hold.
    """

    name = "momentum_score_composite_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip opening noise + indicator warmup
    session_end_minutes = 920     # 15:20 IST — avoid low-liquidity close
    max_trades_per_day = 6
    max_lookback = 240            # 20-min warmup for slowest indicator (ADX 2×period=48)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("composite_threshold", 55.0, 40.0, 75.0),
            TunableParam("persistence_bars",     3.0,   2.0,  6.0),
            TunableParam("vix_max",             25.0,  18.0, 30.0),
            TunableParam("stop_pts",             4.0,   2.0,  8.0),
            TunableParam("target_pts",           7.0,   4.0, 14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill nulls/NaN before to_numpy) ──
        close  = spot_df["close"].fill_nan(None).fill_null(strategy="forward").to_numpy()
        high   = spot_df["high"].fill_nan(None).fill_null(strategy="forward").to_numpy()
        low    = spot_df["low"].fill_nan(None).fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(strategy="forward").cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        # ── Parameters ──
        threshold   = params.get("composite_threshold", 55.0)
        persist_n   = int(round(params.get("persistence_bars", 3.0)))
        vix_max     = params.get("vix_max", 25.0)
        stop_pts    = params.get("stop_pts", 4.0)
        target_pts  = params.get("target_pts", 7.0)

        # ── VIX (aligned to spot bars) ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ─────────────────────────────────────────────
        # Sub-signal 1: RSI(24) — 2-min price velocity
        # Score: clip((RSI - 50) / 2.5, -20, 20)
        # ─────────────────────────────────────────────
        rsi_24   = _rsi(close, 24)
        rsi_score = np.clip((rsi_24 - 50.0) / 2.5, -20.0, 20.0)

        # ─────────────────────────────────────────────
        # Sub-signal 2: MACD histogram (EMA24, EMA52, signal EMA18)
        # Normalised by ATR(24) to make it scale-independent
        # ─────────────────────────────────────────────
        ema24      = _ema(close, 24)
        ema52      = _ema(close, 52)
        macd_line  = ema24 - ema52
        macd_sig   = _ema(macd_line, 18)
        macd_hist  = macd_line - macd_sig
        atr24      = _atr(high, low, close, 24)
        safe_atr   = np.where(atr24 > 0.1, atr24, 0.1)
        macd_score = np.clip(macd_hist / safe_atr * 20.0, -20.0, 20.0)

        # ─────────────────────────────────────────────
        # Sub-signal 3: ROC(12) — 1-min rate of change
        # NIFTY typical 1-min move ±0.05-0.2% → scale ×10000 → [-20, 20]
        # ─────────────────────────────────────────────
        prev12      = np.concatenate([close[:12], close[:-12]])
        safe_prev   = np.where(prev12 > 0, prev12, 1.0)
        roc_12      = (close - prev12) / safe_prev
        roc_score   = np.clip(roc_12 * 10000.0, -20.0, 20.0)

        # ─────────────────────────────────────────────
        # Sub-signal 4: ADX(24) directional bias
        # sign(+DI - -DI) × min(ADX, 40) / 2  → [-20, +20]
        # ─────────────────────────────────────────────
        adx_score = _adx_dir(high, low, close, 24)

        # ─────────────────────────────────────────────
        # Sub-signal 5: Volume-price direction momentum
        # (vol - SMA(vol,60)) / SMA(vol,60) × sign(bar direction) → clip ×20
        # ─────────────────────────────────────────────
        vol_sma60   = _sma(volume, 60)
        safe_vsma   = np.where(vol_sma60 > 0, vol_sma60, 1.0)
        bar_dir     = np.sign(close - np.concatenate([[close[0]], close[:-1]]))
        vol_rel     = (volume - vol_sma60) / safe_vsma
        vol_score   = np.clip(vol_rel * 20.0 * bar_dir, -20.0, 20.0)

        # ─────────────────────────────────────────────
        # Composite score [-100, +100]
        # ─────────────────────────────────────────────
        composite = rsi_score + macd_score + roc_score + adx_score + vol_score

        # ─────────────────────────────────────────────
        # Intraday VWAP (cumulative per session day)
        # ─────────────────────────────────────────────
        typical_price = (high + low + close) / 3.0
        vwap     = np.zeros(n)
        cum_pv   = 0.0
        cum_vol  = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_pv  = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            cum_pv  += typical_price[i] * max(volume[i], 0.0)
            cum_vol += max(volume[i], 0.0)
            vwap[i]  = cum_pv / cum_vol if cum_vol > 0 else close[i]

        # ─────────────────────────────────────────────
        # Persistence filter: composite held above/below
        # (threshold - 10) for last persist_n bars
        # ─────────────────────────────────────────────
        persist_thresh = threshold - 10.0
        bull_persist   = np.zeros(n, dtype=bool)
        bear_persist   = np.zeros(n, dtype=bool)
        for i in range(persist_n - 1, n):
            window = composite[i - persist_n + 1: i + 1]
            bull_persist[i] = bool(np.all(window > persist_thresh))
            bear_persist[i] = bool(np.all(window < -persist_thresh))

        # ─────────────────────────────────────────────
        # Entry signals
        # ─────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok     = vix_close < vix_max

        buy_ce = (
            in_session
            & vix_ok
            & (composite > threshold)
            & bull_persist
            & (close > vwap)
        )
        buy_pe = (
            in_session
            & vix_ok
            & (composite < -threshold)
            & bear_persist
            & (close < vwap)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
