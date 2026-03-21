"""ORB with ATR-Scaled Stops/Targets — NIFTY 5s index options.

Original: equity ORB on NIFTY50 stocks (1-min bars), ATR(14) scaled exits.
Converted: NIFTY index ORB (09:15-09:30 range), ATR(168) scaled stops/targets
in option premium points. Entry window 09:30-10:30 IST.

Key insight: decouple ORB entry signal (price structure) from trade management
(current-session volatility via ATR). High-vol days get wider stops/targets;
low-vol days use tighter management to protect small gains.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = hl if hl >= hc and hl >= lc else (hc if hc >= lc else lc)
    return tr


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR smoothing."""
    n = len(high)
    tr = _compute_true_range(high, low, close)
    atr = np.zeros(n)
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """ADX using Wilder smoothing."""
    n = len(high)
    adx = np.zeros(n)
    if n < period * 2 + 1:
        return adx

    tr = _compute_true_range(high, low, close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0.0) else 0.0
        minus_dm[i] = down if (down > up and down > 0.0) else 0.0

    # Wilder smoothing
    alpha = 1.0 / period
    sth = np.sum(tr[1: period + 1])
    spdm = np.sum(plus_dm[1: period + 1])
    sndm = np.sum(minus_dm[1: period + 1])

    di_plus_arr = np.zeros(n)
    di_minus_arr = np.zeros(n)
    dx_arr = np.zeros(n)

    for i in range(period + 1, n):
        sth = sth * (1.0 - alpha) + tr[i]
        spdm = spdm * (1.0 - alpha) + plus_dm[i]
        sndm = sndm * (1.0 - alpha) + minus_dm[i]
        if sth > 0.0:
            di_plus_arr[i] = 100.0 * spdm / sth
            di_minus_arr[i] = 100.0 * sndm / sth
        di_sum = di_plus_arr[i] + di_minus_arr[i]
        dx_arr[i] = 100.0 * abs(di_plus_arr[i] - di_minus_arr[i]) / di_sum if di_sum > 0.0 else 0.0

    # Smooth DX into ADX
    start = period * 2
    if start < n:
        adx[start] = np.mean(dx_arr[period: start])
        for i in range(start + 1, n):
            adx[i] = adx[i - 1] * (1.0 - alpha) + dx_arr[i] * alpha
    return adx


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_ids: np.ndarray) -> np.ndarray:
    """Session VWAP, reset each day."""
    n = len(close)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -999999
    for i in range(n):
        if day_ids[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = day_ids[i]
        v = volume[i]
        cum_pv += close[i] * v
        cum_v += v
        vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]
    return vwap


def _compute_sma(arr: np.ndarray, period: int) -> np.ndarray:
    n = len(arr)
    sma = np.zeros(n)
    for i in range(period - 1, n):
        sma[i] = np.mean(arr[i - period + 1: i + 1])
    return sma


class Strategy(BaseStrategy):
    """ORB with ATR-Scaled Stops/Targets on NIFTY.

    Entry: ORB breakout/breakdown (09:15-09:30 range), entry window 09:30-10:30.
    Stops/targets: adaptive to ATR(168) = 14-min volatility at 5s bars.
    Filters: ADX(36) > 18, volume ratio > 1.2, close above/below VWAP.
    """

    name = "orb_atr_scaled_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — after ORB window closes
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 360            # 30 min warmup for ATR(168) + ADX(36) to stabilise

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_threshold", 18.0, 10.0, 30.0),
            TunableParam("vol_ratio_threshold", 1.2, 0.8, 2.0),
            TunableParam("close_range_threshold", 0.70, 0.55, 0.85),
            TunableParam("stop_atr_mult", 2.5, 1.5, 4.0),
            TunableParam("target_atr_mult", 5.0, 3.0, 8.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before numpy) ──────────────
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        day_ids = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── 1. Opening Range (09:15–09:30) per session day ───────────────────
        # 555 = 09:15 IST, 570 = 09:30 IST (minutes from midnight)
        orb_high = np.full(n, np.nan)
        orb_low = np.full(n, np.nan)
        for day in np.unique(day_ids):
            day_mask = day_ids == day
            orb_mask = day_mask & (time_min >= 555) & (time_min < 570)
            if orb_mask.any():
                day_orb_high = float(np.max(high[orb_mask]))
                day_orb_low = float(np.min(low[orb_mask]))
                orb_high[day_mask] = day_orb_high
                orb_low[day_mask] = day_orb_low

        orb_valid = (~np.isnan(orb_high)) & (~np.isnan(orb_low))

        # ── 2. ATR(168) = 14-min volatility window at 5s ─────────────────────
        # 168 bars × 5s = 840s = 14 min — exact time-equivalent of ATR(14, 1-min)
        atr_168 = _compute_atr(high, low, close, 168)

        # ── 3. ADX(36) = 3-min directional strength ──────────────────────────
        # Compressed from 14-min (original) to 3-min: matches 30-120s hold window
        adx_36 = _compute_adx(high, low, close, 36)

        # ── 4. Session VWAP (cumulative, resets each day) ─────────────────────
        vwap = _compute_vwap(close, volume, day_ids)

        # ── 5. Volume ratio vs 3-min SMA (36 bars) ───────────────────────────
        vol_sma = _compute_sma(volume, 36)
        vol_ratio = np.where(vol_sma > 0.0, volume / vol_sma, 1.0)

        # ── 6. Bar range position: where did close land within bar? ──────────
        bar_range = high - low
        bar_range = np.where(bar_range > 0.0, bar_range, 1.0)
        close_in_range = (close - low) / bar_range  # 0.0=bottom, 1.0=top

        # ── 7. ATR-adaptive stops/targets in option premium points ────────────
        # ATR_168 ≈ 2-5 NIFTY spot pts/bar; × 0.5 (delta) ≈ option pt equivalent
        option_atr = np.nan_to_num(atr_168 * 0.5, nan=2.0)
        stop_atr_mult = params.get("stop_atr_mult", 2.5)
        target_atr_mult = params.get("target_atr_mult", 5.0)
        # clip: stop 3-8 pts, target 6-14 pts (consistent with 30-120s hold sizes)
        stop_pts = np.clip(option_atr * stop_atr_mult, 3.0, 8.0)
        target_pts = np.clip(option_atr * target_atr_mult, 6.0, 14.0)

        # ── 8. Threshold parameters ───────────────────────────────────────────
        adx_thresh = params.get("adx_threshold", 18.0)
        vol_thresh = params.get("vol_ratio_threshold", 1.2)
        cr_thresh = params.get("close_range_threshold", 0.70)

        # ── 9. Session and entry-window filters ───────────────────────────────
        after_orb = time_min >= 570          # 09:30 IST — ORB window closed
        entry_window = time_min <= 630       # 10:30 IST — first hour only
        in_session = (
            (time_min >= self.session_start_minutes) &
            (time_min < self.session_end_minutes)
        )

        # ── 10. VIX filter (optional) ─────────────────────────────────────────
        vix_ok = np.ones(n, dtype=bool)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()
            vix_ok = (vix_close >= 12.0) & (vix_close <= 28.0)

        # ── 11. Entry signals ─────────────────────────────────────────────────
        base_filter = in_session & after_orb & entry_window & orb_valid & vix_ok & (adx_36 > adx_thresh) & (vol_ratio > vol_thresh)

        # Bullish: close breaks above ORB high, above VWAP, bullish bar close
        buy_ce = (
            base_filter &
            (close > orb_high) &
            (close > vwap) &
            (close_in_range > cr_thresh)
        )

        # Bearish: close breaks below ORB low, below VWAP, bearish bar close
        buy_pe = (
            base_filter &
            (close < orb_low) &
            (close < vwap) &
            (close_in_range < (1.0 - cr_thresh))
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=stop_pts,
            target_points=target_pts,
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
