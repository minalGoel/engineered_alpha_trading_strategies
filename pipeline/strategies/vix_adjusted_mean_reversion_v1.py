"""VIX-Adjusted VWAP Mean Reversion (NIFTY, 5-second bars)

Mechanism: NIFTY deviates from session VWAP when directional flow overwhelms market
makers. Institutional VWAP-benchmarked algorithms fade these deviations once they
become statistically significant — but the significance threshold scales with VIX.
Low VIX (< 14): z-score ~1.0 triggers the reverting force. High VIX (> 20): z-score
~2.5 needed. We enter when the deviation crosses the VIX-adaptive threshold AND the
z-score has already started reverting, then ride the 30-90s institutional fading impulse.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vix_adjusted_mean_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    max_lookback = 240            # 20 min warmup for z-score stddev to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_oversold",   40.0, 25.0, 50.0),
            TunableParam("rsi_overbought", 60.0, 50.0, 75.0),
            TunableParam("vix_cap",        25.0, 20.0, 30.0),
            TunableParam("stop_pts",        3.0,  2.0,  8.0),
            TunableParam("target_pts",      6.0,  4.0, 15.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        close    = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume   = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        rsi_oversold   = params.get("rsi_oversold",   40.0)
        rsi_overbought = params.get("rsi_overbought", 60.0)
        vix_cap        = params.get("vix_cap",        25.0)
        stop_pts       = params.get("stop_pts",        3.0)
        target_pts     = params.get("target_pts",      6.0)

        # ── VIX close aligned to spot bars ─────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── VIX-adaptive threshold: clip(1.0 + (vix - 12) * 0.1, 0.8, 2.5) ───
        adaptive_thresh = np.clip(1.0 + (vix_close - 12.0) * 0.1, 0.8, 2.5)

        # ── Session VWAP (cumulative per calendar day, reset at day boundary) ──
        vwap = _compute_session_vwap(close, volume, day_id)

        # ── VWAP deviation z-score with 240-bar (20 min) rolling stddev ────────
        dev    = close - vwap
        zscore = _rolling_zscore(dev, lookback=240)

        # ── Z-score delta: positive = zscore rising (reversion started if we
        #    were below VWAP), negative = falling (reversion if above VWAP) ──────
        zscore_delta = np.zeros(n)
        zscore_delta[1:] = zscore[1:] - zscore[:-1]

        # ── RSI(24) ≈ 2-minute RSI on spot close ───────────────────────────────
        rsi = _compute_rsi(close, period=24)

        # ── Session + VIX filters ───────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok     = vix_close < vix_cap
        warmed_up  = zscore != 0.0   # zscore == 0 while stddev window not yet full

        # ── Entry: Buy CE (bullish — NIFTY below VWAP, expect reversion up) ────
        buy_ce = (
            in_session
            & vix_ok
            & warmed_up
            & (zscore < -adaptive_thresh)          # below VIX-scaled lower band
            & (rsi < rsi_oversold)                 # fast momentum exhausted
            & (zscore_delta > 0.0)                 # z-score already turning up
        )

        # ── Entry: Buy PE (bearish — NIFTY above VWAP, expect reversion down) ──
        buy_pe = (
            in_session
            & vix_ok
            & warmed_up
            & (zscore > adaptive_thresh)           # above VIX-scaled upper band
            & (rsi > rsi_overbought)               # fast momentum exhausted
            & (zscore_delta < 0.0)                 # z-score already turning down
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,             # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _compute_session_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Cumulative VWAP per session day, reset at each day boundary."""
    n = len(close)
    vwap = np.empty(n)
    cum_tp_vol = 0.0
    cum_vol    = 0.0
    current_day = -1
    for i in range(n):
        if day_id[i] != current_day:
            current_day = day_id[i]
            cum_tp_vol  = 0.0
            cum_vol     = 0.0
        v = volume[i] if volume[i] > 0 else 0.0
        cum_tp_vol += close[i] * v
        cum_vol    += v
        vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0.0 else close[i]
    return vwap


def _rolling_zscore(series: np.ndarray, lookback: int) -> np.ndarray:
    """Z-score of series[i] relative to rolling stddev over series[i-lookback:i]."""
    n      = len(series)
    result = np.zeros(n)
    for i in range(lookback, n):
        window = series[i - lookback:i]
        s = np.std(window)
        if s > 0.0:
            result[i] = series[i] / s
    return result


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI. Returns 50.0 during warmup period."""
    n   = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)

    avg_gain        = np.zeros(n)
    avg_loss        = np.zeros(n)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    for i in range(period, n):
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs     = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi
