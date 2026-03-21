"""
RSI Extreme Reversion — NIFTY/BANKNIFTY 5-second index options.

Original equity strategy: RSI(7) < 15 or > 85 on 1-min NIFTY50 stocks with
VWAP deviation + volume confirmation. Holds 5-20 min for midline reversion.

Converted: RSI(24) on 5s spot bars, thresholds 20/80, VWAP + relative volume
filters. Targets the 30-90 second reversion after an extreme RSI flush.
ATM CE on bullish cross, ATM PE on bearish cross.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed RSI. Returns array of same length as close, neutral 50.0 during warmup."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.diff(close)  # length n-1
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    # Seed with simple average of first `period` changes
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Intraday VWAP reset at each new day_id."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(n):
        if i == 0 or day_id[i] != day_id[i - 1]:
            cum_pv = close[i] * volume[i]
            cum_v = float(volume[i])
        else:
            cum_pv += close[i] * volume[i]
            cum_v += float(volume[i])
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean; values before warmup are set to arr mean (neutral)."""
    n = len(arr)
    out = np.full(n, np.nanmean(arr) if n > 0 else 1.0)
    cum = 0.0
    for i in range(n):
        cum += arr[i]
        if i >= window:
            cum -= arr[i - window]
            out[i] = cum / window
        elif i >= 1:
            out[i] = cum / (i + 1)
    return out


class Strategy(BaseStrategy):
    name = "rsi_extreme_reversion_v1"
    underlying = "BOTH"
    session_start_minutes = 565   # 09:25 IST
    session_end_minutes = 900     # 15:00 IST
    max_trades_per_day = 8
    max_lookback = 144            # RSI(24) + vol SMA(120) = 144 bars warmup

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_oversold",   20.0, 12.0, 28.0),
            TunableParam("rsi_overbought", 80.0, 72.0, 88.0),
            TunableParam("vwap_dist_bps",  15.0,  8.0, 30.0),
            TunableParam("vol_mult",        1.3,  1.0,  2.0),
            TunableParam("vix_max",        22.0, 16.0, 30.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before numpy) ──────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        volume_raw = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].fill_null(0).to_numpy().astype(int)
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy().astype(int)

        # ── Parameters ───────────────────────────────────────────────────────
        rsi_oversold   = params.get("rsi_oversold",   20.0)
        rsi_overbought = params.get("rsi_overbought", 80.0)
        vwap_thresh    = params.get("vwap_dist_bps",  15.0)
        vol_mult       = params.get("vol_mult",         1.3)
        vix_max        = params.get("vix_max",         22.0)

        # ── RSI(24) on spot close ─────────────────────────────────────────────
        rsi = _compute_rsi(close, 24)

        # ── Intraday VWAP ─────────────────────────────────────────────────────
        vwap = _compute_vwap(close, volume_raw, day_id)
        vwap_safe = np.where(vwap > 0.0, vwap, close)
        vwap_dist = (close - vwap_safe) / vwap_safe * 10000.0  # bps

        # ── Relative volume (rolling 120-bar SMA) ────────────────────────────
        vol_sma = _rolling_mean(volume_raw, 120)
        vol_sma_safe = np.where(vol_sma > 0.0, vol_sma, 1.0)
        rel_vol = volume_raw / vol_sma_safe

        # ── VIX (aligned to spot bars) ───────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy().astype(float)

        # ── Session mask ─────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── RSI cross signals ─────────────────────────────────────────────────
        # Shift RSI by 1 to get previous bar values; neutral (50) for first bar
        prev_rsi = np.empty(n)
        prev_rsi[0] = 50.0
        prev_rsi[1:] = rsi[:-1]

        # Bullish: prior bar oversold, current bar exits oversold zone
        rsi_cross_up = (prev_rsi < rsi_oversold) & (rsi >= rsi_oversold)

        # Bearish: prior bar overbought, current bar exits overbought zone
        rsi_cross_dn = (prev_rsi > rsi_overbought) & (rsi <= rsi_overbought)

        # ── Entry signals ─────────────────────────────────────────────────────
        buy_ce = (
            in_session
            & rsi_cross_up
            & (vwap_dist < -vwap_thresh)   # price still below VWAP (reversion pull active)
            & (rel_vol > vol_mult)          # above-average participation during flush
            & (vix_close < vix_max)         # calm-enough volatility regime
        )

        buy_pe = (
            in_session
            & rsi_cross_dn
            & (vwap_dist > vwap_thresh)    # price still above VWAP
            & (rel_vol > vol_mult)
            & (vix_close < vix_max)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 3),
            target_points=np.full(n, 6),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,         # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
