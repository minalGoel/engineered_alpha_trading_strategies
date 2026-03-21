"""VWAP Band Squeeze Breakout Strategy (vwap_band_squeeze_v1)

Mechanism: On NIFTY, TWAP/VWAP execution algorithms from FIIs and domestic institutions
create intraday volatility compressions where the index churns within a tight corridor
around session VWAP. When VWAP band width drops to the 20th percentile of its recent
10-minute distribution, institutional orders are temporarily balanced and directional
flow is coiled. Once price closes above/below the band with volume confirmation, NIFTY
typically moves 15-25 spot points in the next 30-90 seconds. At 5-second resolution we
capture the breakout bar itself, entering before slower 1-min strategies see the signal.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_vwap(
    close: np.ndarray, volume: np.ndarray, day_id: np.ndarray
) -> np.ndarray:
    """Cumulative session VWAP, resets at each new day_id."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = day_id[0] - 1  # force reset on first bar
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_v += volume[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling population std (ddof=1); 0.0 for bars with insufficient history."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window - 1, n):
        result[i] = np.std(arr[i - window + 1 : i + 1], ddof=1)
    return result


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean; 0.0 for bars with insufficient history."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window - 1, n):
        result[i] = np.mean(arr[i - window + 1 : i + 1])
    return result


def _rolling_percentile_rank(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling percentile rank (0-100) of current value within past window values.
    Returns 50.0 (neutral) where history is insufficient."""
    n = len(arr)
    result = np.full(n, 50.0)
    for i in range(window - 1, n):
        window_vals = arr[i - window + 1 : i + 1]
        result[i] = float(np.sum(window_vals <= arr[i])) / window * 100.0
    return result


class Strategy(BaseStrategy):
    name = "vwap_band_squeeze_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 240            # 20-min warmup: 60-bar std + 120-bar percentile + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("squeeze_pctile",  20.0, 10.0, 30.0),  # band-width percentile threshold
            TunableParam("vol_ratio_thresh", 1.4,  1.1,  2.0),  # volume breakout multiplier
            TunableParam("band_k",           2.0,  1.5,  2.5),  # band width multiplier (sigma)
            TunableParam("stop_pts",         5.0,  2.0,  7.0),  # option premium points stop
            TunableParam("target_pts",       8.0,  4.0, 12.0),  # option premium points target
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays, forward-fill NaN before converting ──────────────
        close   = spot_df["close"].forward_fill().to_numpy()
        volume  = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id  = spot_df["day_id"].to_numpy()

        squeeze_pctile   = params.get("squeeze_pctile",   20.0)
        vol_ratio_thresh = params.get("vol_ratio_thresh",  1.4)
        band_k           = params.get("band_k",            2.0)
        stop_pts         = params.get("stop_pts",          5.0)
        target_pts       = params.get("target_pts",        8.0)

        # ── VIX filter (moderate-volatility regime) ──────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Session-cumulative VWAP (resets each day) ────────────────────────
        vwap = _compute_vwap(close, volume, day_id)

        # ── VWAP band std over 60 bars (5 min) ───────────────────────────────
        dev = close - vwap
        dev_std = _rolling_std(dev, 60)
        dev_std = np.where(dev_std < 1e-6, 1e-6, dev_std)  # avoid zero std

        upper = vwap + band_k * dev_std
        lower = vwap - band_k * dev_std

        # Band width as % of VWAP (normalised)
        bw = np.where(vwap > 0.0, (upper - lower) / vwap * 100.0, 0.0)

        # ── 120-bar rolling percentile rank of band width (10 min) ───────────
        bw_pctile = _rolling_percentile_rank(bw, 120)

        # ── Volume ratio: current bar vs 36-bar mean (3 min) ─────────────────
        vol_sma = _rolling_mean(volume, 36)
        vol_ratio = np.where(vol_sma > 0.0, volume / vol_sma, 0.0)

        # ── Composite filters ────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok     = (vix_close >= 12.0) & (vix_close <= 22.0)
        squeeze    = bw_pctile < squeeze_pctile
        vol_ok     = vol_ratio > vol_ratio_thresh
        warmed_up  = np.arange(n) >= 240  # need 240 bars for reliable percentile

        base_filter = in_session & vix_ok & squeeze & vol_ok & warmed_up

        # ── Entry signals ─────────────────────────────────────────────────────
        # Bullish: close breaks ABOVE upper band with midline confirmation
        buy_ce = base_filter & (close > upper) & (close > vwap)

        # Bearish: close breaks BELOW lower band with midline confirmation
        buy_pe = base_filter & (close < lower) & (close < vwap)

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
