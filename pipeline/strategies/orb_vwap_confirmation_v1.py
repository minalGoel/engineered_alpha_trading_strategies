"""
ORB + VWAP Dual Confirmation — NIFTY 5-second options strategy.

Mechanism: NIFTY's first 20 minutes (09:15-09:35) absorbs overnight institutional
positioning. When price breaks the ORB high AND is meaningfully above VWAP with the
ORB high itself above VWAP (structural alignment), VWAP-benchmarked institutional
algorithms are already net buyers who have pushed through the opening supply ceiling.
A 2-bar (10s) persistence filter eliminates stop-hunt spikes, selecting only breakouts
sustained by genuine order flow.

Converted from: trading_strategies/unique_strategies_all/Strategy_231.json
Original: 1-min ORB + VWAP on NIFTY50 constituents, hold 20-180 min.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam

# ── Time constants (minutes from midnight IST) ──
_ORB_START = 555   # 09:15 IST — opening range start
_ORB_END = 575     # 09:35 IST — opening range end (20-min = 240 bars)


def _compute_vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  volume: np.ndarray, day_ids: np.ndarray) -> np.ndarray:
    """Cumulative VWAP reset per day. Typical price = (H+L+C)/3."""
    n = len(close)
    vwap = np.zeros(n)
    unique_days = np.unique(day_ids)
    for d in unique_days:
        idx = np.where(day_ids == d)[0]
        tp = (high[idx] + low[idx] + close[idx]) / 3.0
        vol = volume[idx].astype(np.float64)
        cum_pv = np.cumsum(tp * vol)
        cum_v = np.cumsum(vol)
        safe_v = np.where(cum_v > 0, cum_v, 1.0)
        vwap[idx] = cum_pv / safe_v
    return vwap


def _compute_orb(high: np.ndarray, low: np.ndarray,
                 day_ids: np.ndarray, time_min: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ORB high/low over 09:15-09:35 window, broadcast to all bars of that day."""
    n = len(high)
    orb_high = np.full(n, np.nan)
    orb_low = np.full(n, np.nan)
    unique_days = np.unique(day_ids)
    for d in unique_days:
        idx = np.where(day_ids == d)[0]
        day_time = time_min[idx]
        orb_mask = (day_time >= _ORB_START) & (day_time < _ORB_END)
        if orb_mask.any():
            orb_h = high[idx][orb_mask].max()
            orb_l = low[idx][orb_mask].min()
        else:
            # Fallback: use first bar high/low if no ORB bars present
            orb_h = high[idx[0]] if len(idx) > 0 else 0.0
            orb_l = low[idx[0]] if len(idx) > 0 else 0.0
        orb_high[idx] = orb_h
        orb_low[idx] = orb_l
    return orb_high, orb_low


def _rolling_mean_same_day(volume: np.ndarray, day_ids: np.ndarray,
                            window: int = 12) -> np.ndarray:
    """Rolling mean of volume over `window` bars, only within the same day."""
    n = len(volume)
    vol_mean = volume.astype(np.float64).copy()
    for i in range(window, n):
        if day_ids[i] == day_ids[i - window]:
            vol_mean[i] = volume[i - window:i + 1].mean()
    return vol_mean


class Strategy(BaseStrategy):
    name = "orb_vwap_confirmation_v1"
    underlying = "NIFTY"
    session_start_minutes = 575   # 09:35 IST — enter only after ORB window completes
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 5
    max_lookback = 240            # 20-min ORB window warmup (240 × 5s)

    def tunable_params(self) -> list[TunableParam]:
        return [
            # VWAP deviation threshold in bps — ensures price is meaningfully above/below VWAP
            TunableParam("vwap_dev_bps", 3.0, 1.0, 10.0),
            # Volume ratio threshold — filters low-participation breakouts
            TunableParam("vol_ratio_threshold", 1.1, 0.8, 1.5),
            # VIX bounds — below 12 = too calm (narrow ORB, high false breakout rate),
            # above 22 = too jumpy (breakouts gap)
            TunableParam("vix_low", 12.0, 8.0, 16.0),
            TunableParam("vix_high", 22.0, 17.0, 30.0),
        ]

    def compute(self, spot_df: pl.DataFrame, option_df: pl.DataFrame,
                vix_df: pl.DataFrame, params: dict) -> OptionSignals:
        # ── Extract and forward-fill spot arrays ──
        spot_clean = spot_df.with_columns([
            pl.col("close").forward_fill(),
            pl.col("high").forward_fill(),
            pl.col("low").forward_fill(),
            pl.col("volume").fill_null(0),
        ])
        n = len(spot_clean)
        close = spot_clean["close"].to_numpy()
        high = spot_clean["high"].to_numpy()
        low = spot_clean["low"].to_numpy()
        volume = spot_clean["volume"].to_numpy().astype(np.float64)
        day_ids = spot_clean["day_id"].to_numpy()
        time_min = spot_clean["time_minutes"].to_numpy()

        # ── Parameters ──
        vwap_dev_threshold = params.get("vwap_dev_bps", 3.0)
        vol_threshold = params.get("vol_ratio_threshold", 1.1)
        vix_low = params.get("vix_low", 12.0)
        vix_high = params.get("vix_high", 22.0)

        # ── Opening Range Breakout (fixed 09:15-09:35 window) ──
        orb_high, orb_low = _compute_orb(high, low, day_ids, time_min)
        # Replace NaN with a large/small sentinel so comparisons are neutral
        orb_high = np.where(np.isnan(orb_high), 1e9, orb_high)
        orb_low = np.where(np.isnan(orb_low), 0.0, orb_low)

        # ── Cumulative VWAP per day ──
        vwap = _compute_vwap(high, low, close, volume, day_ids)

        # ── VWAP deviation in basis points ──
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        vwap_dev_bps = (close - safe_vwap) / safe_vwap * 10000.0

        # ── Volume ratio (current / 12-bar rolling mean within same day) ──
        vol_mean_12 = _rolling_mean_same_day(volume, day_ids, window=12)
        safe_vol_mean = np.where(vol_mean_12 > 0, vol_mean_12, 1.0)
        vol_ratio = volume / safe_vol_mean

        # ── VIX aligned to spot bars ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Structural alignment ──
        # Long: ORB high itself must be above VWAP (net buyers from open)
        orb_high_above_vwap = orb_high > vwap
        # Short: ORB low itself must be below VWAP (net sellers from open)
        orb_low_below_vwap = orb_low < vwap

        # ── 2-bar persistence filter (10 seconds) ──
        # Signal fires on the 2nd consecutive bar above ORB high / below ORB low
        above_orb_high = close > orb_high
        below_orb_low = close < orb_low

        persist_above = np.zeros(n, dtype=bool)
        persist_below = np.zeros(n, dtype=bool)
        # Both current and previous bar must be on the breakout side
        persist_above[1:] = above_orb_high[1:] & above_orb_high[:-1]
        persist_below[1:] = below_orb_low[1:] & below_orb_low[:-1]
        # Enforce same-day continuity (don't carry persistence across day boundary)
        same_day = np.zeros(n, dtype=bool)
        same_day[1:] = day_ids[1:] == day_ids[:-1]
        persist_above &= same_day
        persist_below &= same_day

        # ── Session and VIX filters ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = (vix_close >= vix_low) & (vix_close <= vix_high)
        vol_ok = vol_ratio > vol_threshold

        # ── Entry signals ──
        # buy_ce: ORB breakout up + VWAP alignment + structural + persistence + volume + VIX
        buy_ce = (
            in_session
            & persist_above
            & (vwap_dev_bps > vwap_dev_threshold)
            & orb_high_above_vwap
            & vol_ok
            & vix_ok
        )

        # buy_pe: ORB breakdown + VWAP alignment + structural + persistence + volume + VIX
        buy_pe = (
            in_session
            & persist_below
            & (vwap_dev_bps < -vwap_dev_threshold)
            & orb_low_below_vwap
            & vol_ok
            & vix_ok
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            # stop: 5 option pts (~10 NIFTY spot pts at delta 0.5)
            # If price retraces 10 spot pts the ORB level has been reclaimed → thesis invalid
            stop_points=np.full(n, 5.0),
            # target: 8 option pts (~16 NIFTY spot pts)
            # Conservative lower bound of NIFTY ORB first-leg extension (20-35 spot pts typical)
            target_points=np.full(n, 8.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,          # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
