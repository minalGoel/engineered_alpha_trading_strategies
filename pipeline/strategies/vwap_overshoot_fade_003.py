"""vwap_overshoot_fade_003 — VWAP Overshoot Fade with VIX Regime + 5-min RSI Exhaustion.

Converted from: trading_strategies/unique_strategies_all/Strategy_137.json

Core thesis:
    On NIFTY, in moderate-VIX sessions (12-24), a 0.5% VWAP deviation driven by retail
    stop-sweeps and short-horizon algo triggers exhausts its order queue after 5 minutes
    (RSI-60 extreme). A green/red reversal bar confirms the micro-pivot has already begun.
    VWAP-benchmarked programs, delta-hedging MMs, and trapped trend-followers converge as
    reversion buyers/sellers over the next 15-90 seconds.

Differentiation from v001/v002:
    - v001: 0.3% deviation, RSI(24)=2-min, no VIX, early session only
    - v002: 0.6% deviation, RSI(36)=3-min, Bollinger bands, no VIX, early session only
    - v003: 0.5% deviation, RSI(60)=5-min (slowest exhaustion), VIX 12-24 regime filter,
            candlestick confirmation (pivot bar), extended session to 14:30
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI. Returns array length n with neutral 50.0 for warmup bars."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)

    avg_gain = np.mean(gains[1:period + 1])
    avg_loss = np.mean(losses[1:period + 1])

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    return rsi


def _compute_vwap(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                  volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Cumulative typical-price VWAP, resets on each new day_id."""
    n = len(close)
    vwap = np.zeros(n)
    typical = (high + low + close) / 3.0
    cum_pv = 0.0
    cum_v = 0.0

    for i in range(n):
        if i == 0 or day_id[i] != day_id[i - 1]:
            cum_pv = typical[i] * float(volume[i])
            cum_v = float(volume[i])
        else:
            cum_pv += typical[i] * float(volume[i])
            cum_v += float(volume[i])
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else typical[i]

    return vwap


class Strategy(BaseStrategy):
    """VWAP Overshoot Fade #003 — VIX regime + 5-min RSI exhaustion + candlestick pivot."""

    name = "vwap_overshoot_fade_003"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST
    session_end_minutes = 870     # 14:30 IST (VIX filter handles quality throughout day)
    max_trades_per_day = 6
    max_lookback = 120            # 10-min warmup ensures RSI(60) is stable

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_threshold", 0.005, 0.003, 0.009),
            TunableParam("rsi_oversold",        22.0,  15.0, 30.0),
            TunableParam("rsi_overbought",       78.0,  70.0, 85.0),
            TunableParam("vix_low",              12.0,  10.0, 16.0),
            TunableParam("vix_high",             24.0,  18.0, 28.0),
            TunableParam("stop_pts",              4.0,   2.0,  7.0),
            TunableParam("target_pts",            6.0,   4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot columns (forward-fill NaN before numpy conversion) ──
        close  = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high   = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low    = spot_df["low"].fill_null(strategy="forward").to_numpy()
        open_  = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(np.float64)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──
        vwap_dev_threshold = params.get("vwap_dev_threshold", 0.005)
        rsi_oversold       = params.get("rsi_oversold",        22.0)
        rsi_overbought     = params.get("rsi_overbought",       78.0)
        vix_low            = params.get("vix_low",              12.0)
        vix_high           = params.get("vix_high",             24.0)
        stop_pts           = params.get("stop_pts",              4.0)
        target_pts         = params.get("target_pts",            6.0)

        # ── Cumulative VWAP (resets per day) ──
        vwap = _compute_vwap(close, high, low, volume, day_id)

        # VWAP deviation: (close - vwap) / vwap
        vwap_dev = np.where(vwap > 0.0, (close - vwap) / vwap, 0.0)

        # ── RSI(60) = 5-minute exhaustion (original RSI(5) on 1-min = same 5-min window) ──
        rsi_60 = _compute_rsi(close, 60)

        # ── VIX close aligned to spot bars ──
        vix_close = np.full(n, 15.0)   # neutral default (inside regime)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime", strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Candlestick confirmation: current 5s bar has already pivoted ──
        is_green = close > open_   # bullish pivot bar — long entry
        is_red   = close < open_   # bearish pivot bar — short entry

        # ── Regime and session filters ──
        vix_in_regime = (vix_close >= vix_low) & (vix_close <= vix_high)
        in_session    = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── Entry signals ──
        # Buy CE: NIFTY ≥ 0.5% below VWAP, 5-min RSI oversold, green pivot bar, moderate VIX
        buy_ce = (
            in_session
            & vix_in_regime
            & (vwap_dev < -vwap_dev_threshold)
            & (rsi_60 < rsi_oversold)
            & is_green
        )

        # Buy PE: NIFTY ≥ 0.5% above VWAP, 5-min RSI overbought, red pivot bar, moderate VIX
        buy_pe = (
            in_session
            & vix_in_regime
            & (vwap_dev > vwap_dev_threshold)
            & (rsi_60 > rsi_overbought)
            & is_red
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
