"""adaptive_threshold_v1 — Adaptive RSI Percentile Threshold with BB Squeeze

Original equity strategy: dynamic RSI thresholds using 120-bar rolling percentiles
on NIFTY50 constituents (1-min bars, 15-45 min hold).

Conversion: On NIFTY, FII-driven flows create persistent RSI regimes where static
30/70 thresholds fail. By calibrating overbought/oversold thresholds to the 90th/10th
percentile of RSI over the past 20 minutes (240 bars at 5s), we detect true regime
extremes. When RSI hits the dynamic threshold AND Bollinger Bands expand from a recent
micro-squeeze, VWAP-benchmarked algorithms are overextended and begin rebalancing,
producing a 10-20 spot point mean reversion within 30-90 seconds.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "adaptive_threshold_v1"
    underlying = "NIFTY"
    session_start_minutes = 575   # 09:35 IST — need 240-bar (20 min) percentile warmup
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 240            # 240 × 5s = 20 min

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_pctile_lo", 0.10, 0.03, 0.20),
            TunableParam("rsi_pctile_hi", 0.90, 0.80, 0.97),
            TunableParam("bb_squeeze_pctile", 0.20, 0.05, 0.35),
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

        # ── Extract arrays ────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        rsi_pctile_lo = params.get("rsi_pctile_lo", 0.10)
        rsi_pctile_hi = params.get("rsi_pctile_hi", 0.90)
        bb_squeeze_pctile = params.get("bb_squeeze_pctile", 0.20)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── VIX filter ────────────────────────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        vix_ok = vix_close < 25.0

        # ── RSI(12) — 1-minute RSI at 5s resolution ───────────────────────
        rsi = _compute_rsi(close, period=12)

        # ── Bollinger Band width (36 bars = 3 min) ────────────────────────
        bb_width = _compute_bb_width(close, period=36)

        # ── Rolling adaptive thresholds (240-bar = 20 min window) ─────────
        pctile_window = 240
        rsi_lo_thresh = np.full(n, np.nan)
        rsi_hi_thresh = np.full(n, np.nan)
        bb_squeeze_thresh = np.full(n, np.nan)

        for i in range(pctile_window, n):
            rsi_window = rsi[i - pctile_window:i]
            valid_rsi = rsi_window[~np.isnan(rsi_window)]
            if len(valid_rsi) >= 30:
                rsi_lo_thresh[i] = np.percentile(valid_rsi, rsi_pctile_lo * 100.0)
                rsi_hi_thresh[i] = np.percentile(valid_rsi, rsi_pctile_hi * 100.0)

            bb_window = bb_width[i - pctile_window:i]
            valid_bb = bb_window[~np.isnan(bb_window)]
            if len(valid_bb) >= 30:
                bb_squeeze_thresh[i] = np.percentile(valid_bb, bb_squeeze_pctile * 100.0)

        # ── Recent micro-squeeze: BB width was below threshold in last 10 bars ──
        had_squeeze = np.zeros(n, dtype=bool)
        for i in range(10, n):
            thresh = bb_squeeze_thresh[i]
            if not np.isnan(thresh):
                had_squeeze[i] = bool(np.any(bb_width[i - 10:i] < thresh))

        # ── RSI turn confirmation ─────────────────────────────────────────
        rsi_turn_up = np.zeros(n, dtype=bool)
        rsi_turn_down = np.zeros(n, dtype=bool)
        for i in range(1, n):
            if not np.isnan(rsi[i]) and not np.isnan(rsi[i - 1]):
                rsi_turn_up[i] = bool(rsi[i] > rsi[i - 1])
                rsi_turn_down[i] = bool(rsi[i] < rsi[i - 1])

        # ── Session filter ────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # Require valid adaptive thresholds (no signal before 240-bar warmup)
        has_thresholds = ~np.isnan(rsi_lo_thresh) & ~np.isnan(rsi_hi_thresh)

        # ── Entry signals ─────────────────────────────────────────────────
        # Bullish mean-reversion: RSI at dynamic oversold + micro-squeeze release + RSI turning up
        buy_ce = (
            in_session
            & has_thresholds
            & (rsi < rsi_lo_thresh)
            & had_squeeze
            & rsi_turn_up
            & vix_ok
        )

        # Bearish mean-reversion: RSI at dynamic overbought + micro-squeeze release + RSI turning down
        buy_pe = (
            in_session
            & has_thresholds
            & (rsi > rsi_hi_thresh)
            & had_squeeze
            & rsi_turn_down
            & vix_ok
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


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's RSI using exponential smoothing seed."""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    delta = np.diff(close)                        # length n-1
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Seed with simple average of first `period` changes
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _compute_bb_width(close: np.ndarray, period: int) -> np.ndarray:
    """Normalised Bollinger Band width: (upper - lower) / SMA = 4 * std / sma."""
    n = len(close)
    bb_width = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = close[i - period + 1: i + 1]
        sma = float(np.mean(window))
        if sma > 0.0:
            bb_width[i] = 4.0 * float(np.std(window)) / sma
    return bb_width
