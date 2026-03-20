"""
vwap_anchored_support_v1 — Previous Session VWAP Cross with Volume Confirmation

Mechanism: On NIFTY, the previous session's VWAP is the most-tracked institutional
benchmark. When price drifts to within 0.15% of this level, institutional order clusters
form (VWAP-lagging algos accumulate, VWAP-beating algos reduce). A clean cross confirmed
by a volume surge and neutral RSI (38-62) resolves into a 15-25 spot-point impulse within
30-90 seconds as the level resolves directionally.

Adapted from: equity anchored-VWAP-from-events strategy (Strategy_144.json)
Adaptation: replaced stock-event anchor with previous session VWAP (the natural
cross-session index institutional reference); compressed hold time from 20-90 min to
30-90 seconds; RSI lookback compressed to 3 min (not 12x scaled); vol SMA compressed
to 100 seconds (not 12x scaled).
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI using exponential smoothing."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)

    # Seed with simple mean of first `period` deltas
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    for i in range(period, n):
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


class Strategy(BaseStrategy):
    name = "vwap_anchored_support_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min gap-resolution noise
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 360            # 30 min warmup (needs RSI(36) + vol_sma(20) warm + prev-day data)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("distance_threshold", 0.0015, 0.0005, 0.003),
            TunableParam("volume_ratio_threshold", 1.3, 1.1, 2.0),
            TunableParam("rsi_low", 38.0, 30.0, 48.0),
            TunableParam("rsi_high", 62.0, 52.0, 72.0),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays (forward-fill NaN in Polars before numpy) ─────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        high = spot_df["high"].fill_null(strategy="forward").to_numpy()
        low = spot_df["low"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Params ────────────────────────────────────────────────────────────
        dist_thresh = params.get("distance_threshold", 0.0015)
        vol_thresh = params.get("volume_ratio_threshold", 1.3)
        rsi_low = params.get("rsi_low", 38.0)
        rsi_high = params.get("rsi_high", 62.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── 1. Previous session VWAP ──────────────────────────────────────────
        # Compute full-session VWAP (typical price weighted) for each day_id.
        # Then assign prev day's VWAP to each bar in the current day.
        typical = (high + low + close) / 3.0
        unique_days = np.unique(day_id)
        sorted_days = sorted(unique_days.tolist())

        day_vwap: dict[int, float] = {}
        for d in sorted_days:
            mask = day_id == d
            d_typical = typical[mask]
            d_vol = volume[mask]
            total_vol = float(np.sum(d_vol))
            if total_vol > 0:
                day_vwap[d] = float(np.sum(d_typical * d_vol) / total_vol)
            else:
                day_vwap[d] = float(np.mean(d_typical)) if len(d_typical) > 0 else np.nan

        # Map each day to its predecessor
        prev_day_map: dict[int, int] = {}
        for i in range(1, len(sorted_days)):
            prev_day_map[sorted_days[i]] = sorted_days[i - 1]

        # Assign prev_day_vwap to each bar; forward-fill any remaining NaN
        prev_vwap = np.full(n, np.nan)
        for i in range(n):
            d = int(day_id[i])
            if d in prev_day_map:
                prev_d = prev_day_map[d]
                prev_vwap[i] = day_vwap.get(prev_d, np.nan)

        # Forward-fill leading NaN (first session in dataset has no prev day)
        for i in range(1, n):
            if np.isnan(prev_vwap[i]):
                prev_vwap[i] = prev_vwap[i - 1]

        has_anchor = ~np.isnan(prev_vwap)

        # ── 2. Distance to prev-day VWAP ─────────────────────────────────────
        with np.errstate(invalid="ignore", divide="ignore"):
            dist = np.where(
                has_anchor & (prev_vwap > 0),
                (close - prev_vwap) / prev_vwap,
                np.nan,
            )

        # ── 3. RSI(36) — 3-minute RSI to detect neutral zone ─────────────────
        rsi = _compute_rsi(close, 36)

        # ── 4. Volume ratio — 20-bar SMA (~100 seconds) ──────────────────────
        vol_sma = np.zeros(n)
        for i in range(20, n):
            vol_sma[i] = np.mean(volume[i - 20:i])

        vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 1.0)

        # ── 5. VIX filter ─────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── 6. Crossover detection ─────────────────────────────────────────────
        # cross_above: was at/below anchor, now above AND within distance threshold
        # cross_below: was at/above anchor, now below AND within distance threshold
        cross_above = np.zeros(n, dtype=bool)
        cross_below = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if np.isnan(dist[i]) or np.isnan(dist[i - 1]):
                continue
            # Require the cross is small (not a gap-over)
            if abs(dist[i]) >= dist_thresh:
                continue
            if dist[i - 1] <= 0.0 and dist[i] > 0.0:
                cross_above[i] = True
            elif dist[i - 1] >= 0.0 and dist[i] < 0.0:
                cross_below[i] = True

        # ── 7. Combined filters ───────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        rsi_neutral = (rsi >= rsi_low) & (rsi <= rsi_high)
        vol_surge = vol_ratio >= vol_thresh
        low_vix = vix_close < 25.0

        # ── 8. Entry signals ──────────────────────────────────────────────────
        buy_ce = in_session & has_anchor & cross_above & vol_surge & rsi_neutral & low_vix
        buy_pe = in_session & has_anchor & cross_below & vol_surge & rsi_neutral & low_vix

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,           # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
