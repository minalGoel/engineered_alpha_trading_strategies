"""VWAP + RSI + Volume Hybrid Mean-Reversion Strategy.

On NIFTY, when the index deviates 20+ spot points below session VWAP with RSI(36)
below 35 (3-minute oversold exhaustion) and volume spiking 1.5x the 20-minute average,
VWAP-benchmarked institutional buy programs activate. We enter on the first uptick,
capturing the initial mean-reversion snap before 1-minute participants react.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "vwap_rsi_volume_hybrid_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 300            # 25 min warmup — needed for vol SMA(240) to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_pts",   20.0, 10.0, 50.0),
            TunableParam("rsi_oversold",   35.0, 25.0, 45.0),
            TunableParam("rsi_overbought", 65.0, 55.0, 75.0),
            TunableParam("vol_threshold",   1.5,  1.2,  2.5),
            TunableParam("stop_pts",        5.0,  3.0, 10.0),
            TunableParam("target_pts",      8.0,  5.0, 15.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN before numpy) ──────────────────
        close = (
            spot_df["close"]
            .fill_nan(None)
            .forward_fill()
            .fill_null(strategy="forward")
            .to_numpy()
            .astype(np.float64)
        )
        volume = (
            spot_df["volume"]
            .cast(pl.Float64)
            .fill_nan(None)
            .forward_fill()
            .fill_null(0.0)
            .to_numpy()
            .astype(np.float64)
        )
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        # ── Parameters ───────────────────────────────────────────────────────
        vwap_dev_pts   = float(params.get("vwap_dev_pts",   20.0))
        rsi_oversold   = float(params.get("rsi_oversold",   35.0))
        rsi_overbought = float(params.get("rsi_overbought", 65.0))
        vol_threshold  = float(params.get("vol_threshold",   1.5))
        stop_pts       = float(params.get("stop_pts",        5.0))
        target_pts     = float(params.get("target_pts",      8.0))

        # ── Indicators ───────────────────────────────────────────────────────
        vwap      = _compute_vwap(close, volume, day_id, n)
        rsi       = _compute_rsi(close, 36, n)        # 3-min RSI at 5s bars
        vol_ratio = _compute_vol_ratio(volume, 240, n) # 20-min rolling vol average

        # VWAP distance in spot points
        vwap_dist = close - vwap

        # Uptick / downtick confirmation (1-bar)
        uptick   = np.zeros(n, dtype=bool)
        downtick = np.zeros(n, dtype=bool)
        uptick[1:]   = close[1:] > close[:-1]
        downtick[1:] = close[1:] < close[:-1]

        # VIX filter (avoid extreme-vol days)
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        vix_ok = vix_close < 25.0

        # ── Session / warmup masks ────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        warmed_up  = np.arange(n) >= self.max_lookback
        active     = in_session & warmed_up & vix_ok

        # ── Signals ──────────────────────────────────────────────────────────
        # buy CE: below VWAP + RSI oversold + volume spike + uptick (reversion starting)
        buy_ce = (
            active
            & (vwap_dist < -vwap_dev_pts)
            & (rsi < rsi_oversold)
            & (vol_ratio > vol_threshold)
            & uptick
        )

        # buy PE: above VWAP + RSI overbought + volume spike + downtick
        buy_pe = (
            active
            & (vwap_dist > vwap_dev_pts)
            & (rsi > rsi_overbought)
            & (vol_ratio > vol_threshold)
            & downtick
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,
            max_trades_per_day=self.max_trades_per_day,
        )


# ── Helper functions ──────────────────────────────────────────────────────────

def _compute_vwap(
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
    n: int,
) -> np.ndarray:
    """Session-cumulative VWAP, reset each day."""
    vwap = np.full(n, np.nan)
    cum_pv = 0.0
    cum_v  = 0.0
    prev_day = -999999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv   = 0.0
            cum_v    = 0.0
            prev_day = day_id[i]
        v = volume[i] if volume[i] > 0 else 1.0
        cum_pv += close[i] * v
        cum_v  += v
        vwap[i] = cum_pv / cum_v
    return vwap


def _compute_rsi(close: np.ndarray, period: int, n: int) -> np.ndarray:
    """Wilder's RSI with given period. Default 50.0 during warmup."""
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    gains  = np.zeros(n)
    losses = np.zeros(n)
    for i in range(1, n):
        diff = close[i] - close[i - 1]
        if diff > 0:
            gains[i]  = diff
        else:
            losses[i] = -diff

    # Seed with simple average over first `period` changes
    avg_gain = float(np.mean(gains[1:period + 1]))
    avg_loss = float(np.mean(losses[1:period + 1]))
    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder's smoothing for subsequent bars
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i])  / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


def _compute_vol_ratio(volume: np.ndarray, period: int, n: int) -> np.ndarray:
    """Ratio of current bar volume to rolling SMA(period). Default 1.0 during warmup."""
    vol_ratio = np.ones(n)
    for i in range(period, n):
        avg_vol = float(np.mean(volume[i - period:i]))
        if avg_vol > 0.0:
            vol_ratio[i] = volume[i] / avg_vol
    return vol_ratio
