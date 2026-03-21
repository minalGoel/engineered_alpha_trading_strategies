"""vwap_band_reversion_v1 — VWAP ±2SD Band Mean Reversion on NIFTY

Mechanism:
    On NIFTY, when the index spot extends to ±2 rolling standard deviations from
    session VWAP, institutional participants benchmarked to VWAP (domestic MFs, FIIs
    executing index rebalancing) begin absorbing aggressively — they're getting fills
    2 sigma better than their benchmark. A 5-second bar that closes back inside the 2SD
    band on elevated volume signals absorption has overwhelmed directional flow. The index
    reverts toward the 1SD band within 60-90 seconds.

Entry:
    buy_ce: prior close <= lower 2SD band, current close > lower band, vol spike
    buy_pe: prior close >= upper 2SD band, current close < upper band, vol spike

Exit:
    stop:   4 option points (~8 spot pts) — absorption failed if extends further
    target: 7 option points (~14 spot pts) — captures ~50-65% of band-to-band reversion
    time:   18 bars (90 seconds)
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Intraday cumulative VWAP, reset at each calendar day boundary."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    current_day = day_id[0]
    for i in range(n):
        if day_id[i] != current_day:
            cum_pv = 0.0
            cum_v = 0.0
            current_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_v += volume[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation (population std)."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window, n):
        result[i] = np.std(arr[i - window: i])
    return result


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window, n):
        result[i] = np.mean(arr[i - window: i])
    return result


class Strategy(BaseStrategy):
    name = "vwap_band_reversion_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — skip first 15 min of VWAP SD instability
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 240            # 20 min warmup for stable VWAP SD (120 bars SD + buffer)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("band_mult", 2.0, 1.5, 2.5),
            TunableParam("vol_ratio_thresh", 1.5, 1.2, 2.5),
            TunableParam("vix_max", 22.0, 18.0, 28.0),
            TunableParam("stop_pts", 4.0, 3.0, 7.0),
            TunableParam("target_pts", 7.0, 5.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays, forward-fill NaN ──────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy() \
            if hasattr(spot_df["volume"].fill_null(0), "cast") \
            else spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ─────────────────────────────────────────────────────────
        band_mult = params.get("band_mult", 2.0)
        vol_ratio_thresh = params.get("vol_ratio_thresh", 1.5)
        vix_max = params.get("vix_max", 22.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── VIX close (aligned to spot bars) ──────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Volume as float array ──────────────────────────────────────────────
        vol = volume.astype(float)

        # ── Cumulative intraday VWAP ───────────────────────────────────────────
        vwap = _compute_vwap(close, vol, day_id)

        # ── Rolling 10-min (120-bar) VWAP deviation stddev ────────────────────
        dev = close - vwap
        sd_120 = _rolling_std(dev, 120)
        # Floor to avoid trivially narrow bands during low-volatility periods
        sd_120 = np.where(sd_120 < 0.5, 0.5, sd_120)

        upper_band = vwap + band_mult * sd_120
        lower_band = vwap - band_mult * sd_120

        # ── Volume spike: vs 10-min (120-bar) rolling average ─────────────────
        vol_avg = _rolling_mean(vol, 120)
        vol_avg = np.where(vol_avg < 1.0, 1.0, vol_avg)
        vol_ratio = vol / vol_avg

        # ── Session and regime filters ─────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < vix_max

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: prior bar closed at/below lower 2SD band; current bar closes back above it
        # buy_pe: prior bar closed at/above upper 2SD band; current bar closes back below it
        buy_ce = np.zeros(n, dtype=bool)
        buy_pe = np.zeros(n, dtype=bool)

        for i in range(1, n):
            if not in_session[i] or not vix_ok[i]:
                continue
            if vol_ratio[i] < vol_ratio_thresh:
                continue
            # Reversal from lower band → bullish
            if close[i - 1] <= lower_band[i - 1] and close[i] > lower_band[i]:
                buy_ce[i] = True
            # Reversal from upper band → bearish
            if close[i - 1] >= upper_band[i - 1] and close[i] < upper_band[i]:
                buy_pe[i] = True

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
