"""
bollinger_squeeze_breakout_v82 — NIFTY 5-second Bollinger Squeeze Breakout

Mechanism:
On NIFTY, intraday consolidation phases see order book depth build on both sides,
compressing realized volatility until Bollinger Bands (3-min) narrow inside Keltner
Channels (3-min). When price presses above the upper BB while the squeeze is active,
aggressive buyers have begun absorbing resting offers and algorithmic momentum
strategies cascade in — producing a 15-30 spot point directional push in 30-90 seconds.

Original: bollinger_squeeze_breakout_v82 (Strategy_112.json), 15-min equity bars,
NIFTY200 universe, hold 75-225 min.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


# ── helpers ──────────────────────────────────────────────────────────────────

def _rolling_sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    cs = np.cumsum(arr)
    out[period - 1:] = (cs[period - 1:] - np.concatenate([[0.0], cs[:-(period)]])) / period
    return out


def _rolling_std(arr: np.ndarray, period: int) -> np.ndarray:
    """Population-based rolling std (ddof=1 for sample)."""
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        out[i] = np.std(arr[i - period + 1: i + 1], ddof=1)
    return out


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) < period:
        return out
    alpha = 2.0 / (period + 1)
    out[period - 1] = float(np.mean(arr[:period]))
    for i in range(period, len(arr)):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    return _ema(tr, period)


# ── strategy ─────────────────────────────────────────────────────────────────

class Strategy(BaseStrategy):
    name = "bollinger_squeeze_breakout_v82"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 72             # 6-min warmup (72 × 5s) — 2× lookback for stable ATR/STDDEV

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("keltner_mult", 1.5, 1.0, 2.5),
            TunableParam("bb_mult", 2.0, 1.5, 2.5),
            TunableParam("stop_pts", 5.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 12.0),
            TunableParam("vix_max", 25.0, 18.0, 35.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── extract spot OHLCV (forward-fill before numpy) ────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        high  = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        low   = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()

        # ── parameters ───────────────────────────────────────────────────
        keltner_mult = params.get("keltner_mult", 1.5)
        bb_mult      = params.get("bb_mult", 2.0)
        stop_pts     = params.get("stop_pts", 5.0)
        target_pts   = params.get("target_pts", 8.0)
        vix_max      = params.get("vix_max", 25.0)

        period = 36  # 3-minute lookback — see mechanism comment above

        # ── Bollinger Bands (STDDEV-based) ───────────────────────────────
        sma    = _rolling_sma(close, period)
        stddev = _rolling_std(close, period)
        bb_upper = sma + bb_mult * stddev
        bb_lower = sma - bb_mult * stddev

        # ── Keltner Channels (ATR-based) ─────────────────────────────────
        ema_line = _ema(close, period)
        atr      = _atr(high, low, close, period)
        kc_upper = ema_line + keltner_mult * atr
        kc_lower = ema_line - keltner_mult * atr

        # ── NaN → neutral before boolean masks ───────────────────────────
        # For BB bands: NaN means insufficient history → treat as no-squeeze
        bb_upper = np.where(np.isnan(bb_upper), close + 1e6, bb_upper)
        bb_lower = np.where(np.isnan(bb_lower), close - 1e6, bb_lower)
        kc_upper = np.where(np.isnan(kc_upper), close + 1e6, kc_upper)
        kc_lower = np.where(np.isnan(kc_lower), close - 1e6, kc_lower)

        # ── VIX filter ───────────────────────────────────────────────────
        vix_ok = np.ones(n, dtype=bool)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()
            vix_ok = vix_close < vix_max

        # ── squeeze condition: BB inside Keltner ─────────────────────────
        squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        # ── session filter ───────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── entry signals ────────────────────────────────────────────────
        # Squeeze active AND price pressing above upper BB → bullish breakout
        buy_ce = in_session & vix_ok & squeeze & (close > bb_upper)
        # Squeeze active AND price pressing below lower BB → bearish breakout
        buy_pe = in_session & vix_ok & squeeze & (close < bb_lower)

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
