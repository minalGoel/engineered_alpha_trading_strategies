"""wavelet_decomposition_v1 — Multi-scale denoising momentum for NIFTY 5-second options.

Mechanism: NIFTY's 5-second price series has multi-scale structure: sub-10s microstructure
noise, 20–80s institutional momentum bursts, and session-level drift. A Haar wavelet level-3
approximation (= 8-bar SMA, 40s) strips the tick noise, leaving the "tradeable frequency
band." When raw NIFTY price crosses above the denoised trend line with the denoised slope
actively rising and sufficient medium-frequency energy, it signals a genuine momentum burst —
not noise — that sustains over the next 30–90 seconds.

Stop: 4 pts — ~8 NIFTY spot points; if momentum reverses within seconds of crossover, thesis
is wrong. Target: 7 pts — wavelet-flagged momentum bursts extend 15–25 spot pts (7–12 opt pts);
7 captures the lower bound with 1:1.75 R:R.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean; NaN for first (window-1) bars."""
    n = len(arr)
    result = np.full(n, np.nan)
    running = 0.0
    valid_count = 0
    for i in range(n):
        val = arr[i]
        if not np.isnan(val):
            running += val
            valid_count += 1
        if i >= window:
            old = arr[i - window]
            if not np.isnan(old):
                running -= old
                valid_count -= 1
        if i >= window - 1 and valid_count == window:
            result[i] = running / window
    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling std (population); 0.0 for first (window-1) bars."""
    n = len(arr)
    result = np.zeros(n)
    for i in range(window - 1, n):
        result[i] = np.std(arr[i - window + 1 : i + 1])
    return result


def _session_vwap(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Cumulative session VWAP, reset each day."""
    n = len(close)
    vwap = np.full(n, np.nan)
    cum_pv = 0.0
    cum_vol = 0.0
    prev_day = -1
    for i in range(n):
        d = day_id[i]
        if d != prev_day:
            cum_pv = 0.0
            cum_vol = 0.0
            prev_day = d
        tp = (high[i] + low[i] + close[i]) / 3.0
        v = volume[i] if not np.isnan(volume[i]) else 0.0
        cum_pv += tp * v
        cum_vol += v
        if cum_vol > 0:
            vwap[i] = cum_pv / cum_vol
    return vwap


class Strategy(BaseStrategy):
    name = "wavelet_decomposition_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — 15 min warmup after open
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 120            # 10 min warmup (120 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # Minimum denoised slope (bps/20s) to confirm trend is actively moving
            TunableParam("slope_threshold_bps", 3.0, 1.0, 8.0),
            # Minimum detail energy (bps) to filter out flat/dead-market crossovers
            TunableParam("min_energy_bps", 2.0, 0.5, 6.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 7.0, 4.0, 14.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays ──────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy().astype(float)
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        slope_thresh = params.get("slope_threshold_bps", 3.0)
        energy_thresh = params.get("min_energy_bps", 2.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── Wavelet level-3 Haar approximation = 8-bar SMA (40s denoised) ───
        # Zeros out level-1 detail (tick noise, 5–10s) and level-2 detail (10–20s),
        # retaining the 40s+ medium-frequency trend.
        denoised = _rolling_mean(close, 8)

        # ── Denoised slope over 4 bars (20s), in bps ────────────────────────
        denoised_slope = np.full(n, np.nan)
        for i in range(4, n):
            d_now = denoised[i]
            d_prev = denoised[i - 4]
            if not np.isnan(d_now) and not np.isnan(d_prev) and d_prev > 0:
                denoised_slope[i] = (d_now - d_prev) / d_prev * 10_000.0

        # ── Detail energy: std of raw-vs-denoised divergence (16 bars = 80s) ─
        # Approximates level-2/3 wavelet detail coefficient energy.
        # High value = active medium-frequency oscillation; low = dead market.
        raw_vs_denoised = np.zeros(n)
        for i in range(n):
            if not np.isnan(denoised[i]) and close[i] > 0:
                raw_vs_denoised[i] = (close[i] - denoised[i]) / close[i] * 10_000.0
        detail_energy = _rolling_std(raw_vs_denoised, 16)

        # ── Session VWAP (no scaling needed — cumulative, resets each day) ──
        vwap = _session_vwap(open_, high, low, close, volume, day_id)

        # ── Safe arrays (NaN → neutral) ──────────────────────────────────────
        denoised_safe = np.where(np.isnan(denoised), close, denoised)
        slope_safe = np.where(np.isnan(denoised_slope), 0.0, denoised_slope)
        vwap_safe = np.where(np.isnan(vwap), close, vwap)

        # ── Crossover detection ───────────────────────────────────────────────
        # cross_up:   was close < denoised, now close >= denoised
        # cross_down: was close > denoised, now close <= denoised
        curr_above = close >= denoised_safe
        prev_above = np.empty(n, dtype=bool)
        prev_above[0] = curr_above[0]
        prev_above[1:] = curr_above[:-1]

        cross_up = (~prev_above) & curr_above
        cross_down = prev_above & (~curr_above)

        # ── Session & warmup filters ─────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        has_warmup = np.arange(n) >= 20  # 100s minimum before trusting denoised slope

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: raw price crosses ABOVE denoised trend
        #         + denoised actively rising (slope > threshold)
        #         + meaningful detail energy (not flat market)
        #         + above session VWAP (institutional buy-side context)
        buy_ce = (
            in_session
            & has_warmup
            & cross_up
            & (slope_safe > slope_thresh)
            & (detail_energy > energy_thresh)
            & (close > vwap_safe)
        )

        # buy_pe: raw price crosses BELOW denoised trend
        #         + denoised actively falling
        #         + meaningful detail energy
        #         + below session VWAP (institutional sell-side context)
        buy_pe = (
            in_session
            & has_warmup
            & cross_down
            & (slope_safe < -slope_thresh)
            & (detail_energy > energy_thresh)
            & (close < vwap_safe)
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
