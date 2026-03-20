"""Williams %R Range-Extreme Reversion — NIFTY 5-second index options.

Thesis: When NIFTY is pushed to the extreme of its 3-minute high-low range (Williams %R < -88
or > -12) with elevated volume, aggressive sellers/buyers have cleared the near-side order book.
Passive resting orders and delta-hedging from gamma-short market makers then drive a swift
reversion toward mid-range. We capture the first 30-90 seconds of this rebound.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Compute Williams %R with given period. Returns values in [-100, 0]. Neutral = -50."""
    n = len(close)
    wr = np.full(n, -50.0)
    for i in range(period - 1, n):
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        rng = hh - ll
        if rng > 0:
            wr[i] = ((hh - close[i]) / rng) * -100.0
    return wr


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average; returns NaN for the warmup period."""
    n = len(arr)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        result[i] = np.mean(arr[i - period + 1:i + 1])
    return result


def _vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
          volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP — resets at each new day_id. Returns close when volume == 0."""
    n = len(close)
    typical = (high + low + close) / 3.0
    vwap_arr = np.empty(n)
    cum_tpv = 0.0
    cum_vol = 0.0
    prev_day = -999999
    for i in range(n):
        if day_id[i] != prev_day:
            cum_tpv = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        cum_tpv += typical[i] * volume[i]
        cum_vol += volume[i]
        vwap_arr[i] = cum_tpv / cum_vol if cum_vol > 0 else close[i]
    return vwap_arr


def _consecutive_extreme(condition: np.ndarray) -> np.ndarray:
    """Count consecutive bars where condition is True (resets on False)."""
    n = len(condition)
    count = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if condition[i]:
            count[i] = count[i - 1] + 1
        else:
            count[i] = 0
    return count


class Strategy(BaseStrategy):
    name = "williams_r_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 60             # 5 minutes warmup (need 36 bars for WR + 24 for vol SMA)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("wr_oversold",        -88.0,  -95.0, -80.0),
            TunableParam("wr_overbought",       -12.0,  -20.0,  -5.0),
            TunableParam("vol_ratio_threshold",   1.5,    1.1,   3.0),
            TunableParam("vwap_band",             0.015,  0.005, 0.03),
            TunableParam("vix_max",              25.0,   18.0,  35.0),
            TunableParam("stop_pts",              4.0,    2.0,   8.0),
            TunableParam("target_pts",            7.0,    4.0,  14.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # --- Extract spot arrays (forward-fill NaN before numpy) ---
        high   = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low    = spot_df["low"].fill_null(strategy="forward").to_numpy()
        close  = spot_df["close"].fill_null(strategy="forward").to_numpy()
        open_  = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id   = spot_df["day_id"].to_numpy()

        # --- Parameters ---
        wr_oversold    = params.get("wr_oversold",        -88.0)
        wr_overbought  = params.get("wr_overbought",       -12.0)
        vol_ratio_thr  = params.get("vol_ratio_threshold",  1.5)
        vwap_band      = params.get("vwap_band",            0.015)
        vix_max        = params.get("vix_max",             25.0)
        stop_pts       = params.get("stop_pts",             4.0)
        target_pts     = params.get("target_pts",           7.0)

        # --- Williams %R (36 bars = 3 minutes) ---
        wr = _williams_r(high, low, close, 36)

        # --- Williams %R slope over 3 bars (15 seconds) ---
        wr_slope = np.zeros(n)
        wr_slope[3:] = wr[3:] - wr[:-3]

        # --- Volume ratio: volume / SMA(volume, 24) ---
        vol_sma = _sma(volume, 24)
        vol_ratio = np.zeros(n)
        valid_vol = vol_sma > 0
        vol_ratio[valid_vol] = volume[valid_vol] / vol_sma[valid_vol]

        # --- Session VWAP ---
        vwap_arr = _vwap(high, low, close, volume, day_id)

        # --- VIX (join asof) ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- Filters ---
        in_session = (
            (time_min >= self.session_start_minutes) &
            (time_min < self.session_end_minutes)
        )

        near_vwap = (
            (close >= vwap_arr * (1.0 - vwap_band)) &
            (close <= vwap_arr * (1.0 + vwap_band))
        )

        low_vix   = vix_close < vix_max
        vol_spike = vol_ratio > vol_ratio_thr

        # --- "Not stuck in extreme" filter (skip if ≥5 consecutive bars at extreme) ---
        wr_in_oversold   = wr < wr_oversold
        wr_in_overbought = wr > wr_overbought
        consec_oversold   = _consecutive_extreme(wr_in_oversold)
        consec_overbought = _consecutive_extreme(wr_in_overbought)

        not_stuck_oversold   = consec_oversold   < 5
        not_stuck_overbought = consec_overbought < 5

        # --- Entry signals ---
        buy_ce = (
            in_session &
            wr_in_oversold &
            vol_spike &
            near_vwap &
            low_vix &
            (wr_slope > 0) &        # WR turning up (reversion starting)
            (close >= open_) &      # bar closing bullishly
            not_stuck_oversold
        )

        buy_pe = (
            in_session &
            wr_in_overbought &
            vol_spike &
            near_vwap &
            low_vix &
            (wr_slope < 0) &        # WR turning down (reversion starting)
            (close <= open_) &      # bar closing bearishly
            not_stuck_overbought
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds
            max_trades_per_day=self.max_trades_per_day,
        )
