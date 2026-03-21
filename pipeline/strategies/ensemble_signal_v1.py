"""ensemble_signal_v1 — Multi-indicator consensus strategy for NIFTY index options.

Thesis: On NIFTY, when at least 3 of 4 distinct technical sub-signals (VWAP deviation,
EMA crossover, Bollinger Band position, 2-min RSI) simultaneously agree on direction,
it identifies a high-conviction 'trend-pullback-oversold' microstructure state where
multiple algorithmic participant types layer orders simultaneously. Hold 30-90 seconds.

Original: equity ensemble on NIFTY 50 constituents, 1-min bars, 15-60 min hold.
Adapted to: NIFTY index, 5-second bars, 30-90 second hold.
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Compute EMA with standard 2/(period+1) smoothing factor."""
    k = 2.0 / (period + 1)
    out = np.empty(len(arr))
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def _rsi_wilder(arr: np.ndarray, period: int) -> np.ndarray:
    """Compute RSI using Wilder smoothing. Returns array of same length, first `period` bars = 50.0."""
    n = len(arr)
    diff = np.empty(n)
    diff[0] = 0.0
    diff[1:] = arr[1:] - arr[:-1]

    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)

    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi

    avg_gain = np.mean(gain[1 : period + 1])
    avg_loss = np.mean(loss[1 : period + 1])

    for i in range(period, n):
        if i > period:
            avg_gain = (avg_gain * (period - 1) + gain[i]) / period
            avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return rsi


def _session_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Compute session-cumulative VWAP, resetting at each new day_id."""
    n = len(close)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_vol = 0.0
    current_day = day_id[0]

    for i in range(n):
        if day_id[i] != current_day:
            cum_pv = 0.0
            cum_vol = 0.0
            current_day = day_id[i]
        v = float(volume[i]) if volume[i] > 0 else 1.0
        cum_pv += close[i] * v
        cum_vol += v
        vwap[i] = cum_pv / cum_vol

    return vwap


class Strategy(BaseStrategy):
    name = "ensemble_signal_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — allow 15-min warmup for EMA60/RSI24
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 120            # 120 bars = 10 min warmup (covers EMA60 and BB60)
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum VWAP deviation fraction to activate Sub1 (0.1% default ≈ 24 NIFTY pts)
            TunableParam("vwap_threshold", 0.001, 0.0005, 0.003),
            # RSI thresholds for Sub4
            TunableParam("rsi_low", 35.0, 25.0, 45.0),
            TunableParam("rsi_high", 65.0, 55.0, 75.0),
            # Minimum ensemble score to fire (default 3 of 4 sub-signals)
            TunableParam("min_score", 3.0, 2.0, 4.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before converting to numpy
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(np.float64)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.int64)
        day_id = spot_df["day_id"].to_numpy().astype(np.int32)
        time_min = spot_df["time_minutes"].to_numpy().astype(np.int32)

        vwap_thresh = params.get("vwap_threshold", 0.001)
        rsi_low = params.get("rsi_low", 35.0)
        rsi_high = params.get("rsi_high", 65.0)
        min_score = int(round(params.get("min_score", 3.0)))

        # ── Sub1: Session VWAP deviation ──────────────────────────────────────
        # +1 if price below VWAP by >= vwap_threshold (expect bullish reversion)
        # -1 if price above VWAP by >= vwap_threshold (expect bearish reversion)
        vwap = _session_vwap(close, volume, day_id)
        dev = (close - vwap) / np.where(vwap > 0, vwap, 1.0)
        sub1 = np.where(dev < -vwap_thresh, 1, np.where(dev > vwap_thresh, -1, 0)).astype(np.int8)

        # ── Sub2: EMA crossover (1-min vs 5-min) ─────────────────────────────
        # +1 if EMA(12) > EMA(60) — 1-min trend above 5-min trend (bullish)
        # -1 otherwise (bearish or neutral)
        ema12 = _ema(close, 12)
        ema60 = _ema(close, 60)
        sub2 = np.where(ema12 > ema60, 1, -1).astype(np.int8)

        # ── Sub3: Bollinger Band breakout (5-min, 2σ) ────────────────────────
        # +1 if close > upper BB (strong bullish momentum breakout)
        # -1 if close < lower BB (strong bearish momentum breakout)
        # 0  if within bands (neutral)
        bb_period = 60
        sma60 = np.zeros(n)
        bb_std = np.zeros(n)
        for i in range(bb_period, n):
            window = close[i - bb_period : i]
            sma60[i] = np.mean(window)
            bb_std[i] = np.std(window)
        upper_bb = sma60 + 2.0 * bb_std
        lower_bb = sma60 - 2.0 * bb_std
        sub3 = np.where(
            close > upper_bb, 1, np.where(close < lower_bb, -1, 0)
        ).astype(np.int8)
        sub3[:bb_period] = 0  # zero out warmup bars

        # ── Sub4: 2-min RSI for short-term exhaustion ────────────────────────
        # +1 if RSI(24) < rsi_low  (oversold, expect bounce up → bullish)
        # -1 if RSI(24) > rsi_high (overbought, expect fade down → bearish)
        # 0  if neutral
        rsi24 = _rsi_wilder(close, 24)
        sub4 = np.where(rsi24 < rsi_low, 1, np.where(rsi24 > rsi_high, -1, 0)).astype(np.int8)

        # ── Ensemble score ────────────────────────────────────────────────────
        score = sub1.astype(np.int16) + sub2.astype(np.int16) + sub3.astype(np.int16) + sub4.astype(np.int16)

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry with 2-bar persistence (10s confirmation) ──────────────────
        # Require ensemble score >= min_score on 2 consecutive bars to filter noise
        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if in_session[i]:
                if score[i] >= min_score and score[i - 1] >= min_score:
                    buy_ce[i] = True
                if score[i] <= -min_score and score[i - 1] <= -min_score:
                    buy_pe[i] = True

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # Stop: 4 pts = ~8 NIFTY spot pts; if momentum fails within 30s, thesis broken
            stop_points=np.full(n, 4.0),
            # Target: 7 pts = ~14 NIFTY spot pts; ~60% of expected 15-25 pt continuation
            target_points=np.full(n, 7.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90s = 18 bars at 5s; mid-range of 30-90s hold
            max_trades_per_day=self.max_trades_per_day,
        )
