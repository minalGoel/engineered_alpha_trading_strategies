"""range_day_band_fade_003 — NIFTY range-day VWAP mean reversion.

Mechanism: On NIFTY range sessions (ADX < 20, flat 5-min VWAP slope), VWAP-benchmarked
institutional algos aggressively defend prices that deviate ~0.05% from session VWAP.
When 1-minute MFI simultaneously hits extreme (≤22 or ≥78), short-term directional
pressure is exhausted. We enter at the exhaustion bar (confirmed by a bullish/bearish
close) inside the 20-minute Bollinger bands, fading the move back toward VWAP.

Converted from: trading_strategies/unique_strategies_all/Strategy_132.json
Original: range_day_band_fade_003 — nifty200 equity range-day VWAP fade, 1-min bars.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


# ── Helper functions ──────────────────────────────────────────────────────────


def _compute_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  volume: np.ndarray, day_ids: np.ndarray) -> np.ndarray:
    """Cumulative session VWAP, reset each trading day."""
    n = len(close)
    vwap = np.zeros(n)
    typ = (high + low + close) / 3.0
    for d in np.unique(day_ids):
        idx = np.where(day_ids == d)[0]
        vol = volume[idx].astype(float)
        vol = np.where(vol == 0, 1e-10, vol)
        cum_tpv = np.cumsum(typ[idx] * vol)
        cum_v = np.cumsum(vol)
        vwap[idx] = cum_tpv / cum_v
    return vwap


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 period: int) -> np.ndarray:
    """Wilder-smoothed ADX. Returns 50.0 (neutral) before warmup completes."""
    n = len(close)
    tr = np.zeros(n)
    pdm = np.zeros(n)
    mdm = np.zeros(n)
    for i in range(1, n):
        h, l, c_prev = high[i], low[i], close[i - 1]
        tr[i] = max(h - l, abs(h - c_prev), abs(l - c_prev))
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0

    # Wilder smoothed sums (seeded at period)
    atr_s = np.zeros(n)
    pdm_s = np.zeros(n)
    mdm_s = np.zeros(n)
    if n > period:
        atr_s[period] = np.sum(tr[1:period + 1])
        pdm_s[period] = np.sum(pdm[1:period + 1])
        mdm_s[period] = np.sum(mdm[1:period + 1])
        for i in range(period + 1, n):
            atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
            pdm_s[i] = pdm_s[i - 1] - pdm_s[i - 1] / period + pdm[i]
            mdm_s[i] = mdm_s[i - 1] - mdm_s[i - 1] / period + mdm[i]

    dx = np.zeros(n)
    for i in range(period, n):
        if atr_s[i] > 0:
            pdi = 100.0 * pdm_s[i] / atr_s[i]
            mdi = 100.0 * mdm_s[i] / atr_s[i]
            dsum = pdi + mdi
            if dsum > 0:
                dx[i] = 100.0 * abs(pdi - mdi) / dsum

    adx = np.full(n, 50.0)  # neutral before warmup
    smooth_start = 2 * period
    if n > smooth_start:
        adx[smooth_start] = np.mean(dx[period:smooth_start + 1])
        for i in range(smooth_start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


def _compute_bb(close: np.ndarray, period: int, mult: float
                ) -> tuple[np.ndarray, np.ndarray]:
    """Bollinger upper and lower bands. Forward-fills warmup zeros."""
    n = len(close)
    bb_upper = np.zeros(n)
    bb_lower = np.zeros(n)
    for i in range(period - 1, n):
        w = close[i - period + 1:i + 1]
        m = float(np.mean(w))
        s = float(np.std(w))
        bb_upper[i] = m + mult * s
        bb_lower[i] = m - mult * s
    # Forward-fill leading zeros
    for i in range(1, n):
        if bb_upper[i] == 0.0:
            bb_upper[i] = bb_upper[i - 1]
        if bb_lower[i] == 0.0:
            bb_lower[i] = bb_lower[i - 1]
    return bb_upper, bb_lower


def _compute_mfi(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                 volume: np.ndarray, period: int) -> np.ndarray:
    """Money Flow Index. Returns 50.0 (neutral) before warmup."""
    n = len(close)
    typ = (high + low + close) / 3.0
    mf_pos = np.zeros(n)
    mf_neg = np.zeros(n)
    for i in range(1, n):
        mf = typ[i] * float(volume[i])
        if typ[i] > typ[i - 1]:
            mf_pos[i] = mf
        elif typ[i] < typ[i - 1]:
            mf_neg[i] = mf
    mfi = np.full(n, 50.0)
    for i in range(period, n):
        pos = float(np.sum(mf_pos[i - period + 1:i + 1]))
        neg = float(np.sum(mf_neg[i - period + 1:i + 1]))
        total = pos + neg
        if total > 0:
            mfi[i] = 100.0 * pos / total
    return mfi


def _compute_atr_ema(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     period: int) -> np.ndarray:
    """Wilder-smoothed ATR."""
    n = len(close)
    atr = np.zeros(n)
    for i in range(1, n):
        tr_i = max(high[i] - low[i],
                   abs(high[i] - close[i - 1]),
                   abs(low[i] - close[i - 1]))
        if i < period:
            atr[i] = tr_i
        elif i == period:
            atr[i] = float(np.mean([
                max(high[j] - low[j],
                    abs(high[j] - close[j - 1]),
                    abs(low[j] - close[j - 1]))
                for j in range(1, period + 1)
            ]))
        else:
            atr[i] = (atr[i - 1] * (period - 1) + tr_i) / period
    return atr


# ── Strategy ─────────────────────────────────────────────────────────────────


class Strategy(BaseStrategy):
    """NIFTY range-day VWAP fade with MFI exhaustion trigger.

    Enters when NIFTY deviates from session VWAP on a confirmed range day
    (ADX < 20, flat VWAP slope) and 1-minute MFI shows exhaustion (≤22 or ≥78).
    The 20-minute Bollinger bands act as a safety filter — we never fade a genuine
    breakout beyond the bands.
    """

    name = "range_day_band_fade_003"
    underlying = "NIFTY"
    session_start_minutes = 615   # 10:15 IST
    session_end_minutes = 855     # 14:15 IST
    max_trades_per_day = 6
    max_lookback = 300            # 25 min warmup — ADX(120) needs 2×120 = 240 bars

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_threshold", 20.0, 12.0, 26.0),
            TunableParam("vwap_dev_threshold", 0.0005, 0.0003, 0.0012),
            TunableParam("mfi_low", 22.0, 15.0, 30.0),
            TunableParam("mfi_high", 78.0, 70.0, 85.0),
            TunableParam("vwap_slope_threshold", 0.06, 0.03, 0.12),
            TunableParam("stop_pts", 3.0, 2.5, 7.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df: pl.DataFrame, option_df: pl.DataFrame,
                vix_df: pl.DataFrame, params: dict) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN before numpy) ────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        open_p = spot_df["open"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_ids = spot_df["day_id"].to_numpy()

        # ── Parameters ────────────────────────────────────────────────────
        adx_thr = float(params.get("adx_threshold", 20.0))
        vwap_dev_thr = float(params.get("vwap_dev_threshold", 0.0005))
        mfi_low_thr = float(params.get("mfi_low", 22.0))
        mfi_high_thr = float(params.get("mfi_high", 78.0))
        slope_thr = float(params.get("vwap_slope_threshold", 0.06))
        stop_pts = float(params.get("stop_pts", 3.0))
        target_pts = float(params.get("target_pts", 6.0))

        # ── Indicators ────────────────────────────────────────────────────

        # 1. Session VWAP (cumulative, reset each day)
        vwap = _compute_vwap(high, low, close, volume, day_ids)

        # 2. ADX(120) = 10-minute ADX — range day context filter
        adx = _compute_adx(high, low, close, period=120)

        # 3. Bollinger Bands SMA(240) ± 2.2σ — 20-minute stretch filter
        bb_upper, bb_lower = _compute_bb(close, period=240, mult=2.2)

        # 4. VWAP slope over 60 bars (5 min), normalized by ATR(20)
        atr20 = _compute_atr_ema(high, low, close, period=20)
        vwap_slope = np.zeros(n)
        for i in range(60, n):
            vwap_slope[i] = abs(vwap[i] - vwap[i - 60])
        vwap_slope_norm = np.where(atr20 > 0, vwap_slope / atr20, 99.0)

        # 5. MFI(12) = 1-minute MFI — fast exhaustion trigger
        mfi = _compute_mfi(high, low, close, volume, period=12)

        # ── VWAP deviation (ratio) ─────────────────────────────────────────
        vwap_safe = np.where(vwap > 0, vwap, close)
        vwap_dev = (close - vwap_safe) / vwap_safe

        # ── Filters ───────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )
        has_warmup = np.arange(n) >= self.max_lookback
        range_day = adx < adx_thr                         # low-ADX range regime
        flat_vwap = vwap_slope_norm < slope_thr           # VWAP not trending
        bullish_bar = close > open_p                       # confirmation bar
        bearish_bar = close < open_p

        # ── Entry signals ─────────────────────────────────────────────────

        # Buy CE (bullish): price below VWAP, inside BB, MFI oversold, bullish bar
        buy_ce = (
            in_session
            & has_warmup
            & range_day
            & flat_vwap
            & (vwap_dev <= -vwap_dev_thr)
            & (close > bb_lower)            # not a breakdown
            & (mfi <= mfi_low_thr)          # selling exhausted
            & bullish_bar                   # first reversal tick
        )

        # Buy PE (bearish): price above VWAP, inside BB, MFI overbought, bearish bar
        buy_pe = (
            in_session
            & has_warmup
            & range_day
            & flat_vwap
            & (vwap_dev >= vwap_dev_thr)
            & (close < bb_upper)            # not a breakout
            & (mfi >= mfi_high_thr)         # buying exhausted
            & bearish_bar                   # first fade tick
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,              # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
