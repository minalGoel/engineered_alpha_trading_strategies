"""Implied Vol Dislocation v1 — VIX vs 5-min intraday realized vol on NIFTY.

Mechanism: On NIFTY, when India VIX is significantly elevated relative to the
index's own rolling 5-minute realized volatility, options market makers are
pricing in more risk than actual price action warrants — they have accumulated
excess long-delta spot positions from delta-hedging short puts at inflated IV,
and as each bar passes without the feared volatility materializing, that
hedge-unwind pressure eases and creates a mild upward drift (buy CE when
ratio z-score is high and 1-min return turns positive). Conversely, when
5-minute realized vol sharply exceeds VIX, a stop-loss cascade is in progress
and the downward momentum tends to persist for another 15-45 seconds before
options repricing catches up (buy PE when RV > VIX and 1-min return is negative).

Original: implied_vol_dislocation_v1 — nifty50 stocks, 1-min bars, VIX/rv_20d
ratio > 1.3 (long) or < 0.8 (short). Hold 60-200 min, fires 3-5x per month.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


class Strategy(BaseStrategy):
    name = "implied_vol_dislocation_v1"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — 15-min warmup for rv_60 accumulation
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 360            # 30-min warmup: 240-bar z-score window + 60-bar RV + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 1.5, 1.0, 2.5),
            TunableParam("min_momentum", 0.0005, 0.0001, 0.002),
            TunableParam("stop_pts", 4.0, 2.0, 8.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # Forward-fill before numpy conversion
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        zscore_thresh = params.get("zscore_threshold", 1.5)
        min_mom = params.get("min_momentum", 0.0005)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── VIX close aligned to spot bars ───────────────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── 5-min rolling annualized realized vol ─────────────────────────────
        # Log returns for better vol estimation; clip to avoid log(0)
        safe_close = np.where(close > 0, close, 1.0)
        log_ret = np.zeros(n)
        log_ret[1:] = np.log(safe_close[1:] / safe_close[:-1])

        rv_60 = _rolling_rv(log_ret, lookback=60, n=n)

        # ── IV/RV ratio (VIX as decimal / 5-min RV as decimal) ────────────────
        iv = vix_close / 100.0
        iv_rv_ratio = iv / np.where(rv_60 > 0.001, rv_60, 0.001)

        # ── 20-min (240-bar) z-score of iv_rv_ratio ───────────────────────────
        ratio_zscore = _rolling_zscore(iv_rv_ratio, lookback=240, n=n)

        # ── 1-min (12-bar) momentum for directional confirmation ──────────────
        mom_12 = np.zeros(n)
        for i in range(12, n):
            if safe_close[i - 12] > 0:
                mom_12[i] = (close[i] - close[i - 12]) / close[i - 12]

        # ── Session filter ────────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: VIX >> intraday RV (fear premium excessive) + 1-min positive momentum
        # Thesis: hedge-unwind pressure easing → NIFTY upward drift
        buy_ce = (
            in_session
            & (ratio_zscore > zscore_thresh)
            & (mom_12 > min_mom)
        )

        # buy_pe: intraday RV >> VIX (unexpected vol burst, options lagging) + 1-min negative momentum
        # Thesis: stop-loss cascade in progress → directional momentum persists 15-45s
        buy_pe = (
            in_session
            & (ratio_zscore < -zscore_thresh)
            & (mom_12 < -min_mom)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=9,   # 45 seconds
            max_trades_per_day=self.max_trades_per_day,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling_rv(log_ret: np.ndarray, lookback: int, n: int) -> np.ndarray:
    """Rolling annualized realized vol from log returns on 5-second bars."""
    # bars_per_year = 252 trading days * 6.25 hours/day * 3600 s/hour / 5 s/bar
    annualizer = np.sqrt(252 * 6.25 * 3600 / 5)
    rv = np.zeros(n)
    for i in range(lookback, n):
        window = log_ret[i - lookback : i]
        rv[i] = np.std(window) * annualizer
    return rv


def _rolling_zscore(arr: np.ndarray, lookback: int, n: int) -> np.ndarray:
    """Rolling z-score over a lookback window; returns 0 when std is near zero."""
    result = np.zeros(n)
    for i in range(lookback, n):
        window = arr[i - lookback : i]
        mu = np.mean(window)
        sigma = np.std(window)
        if sigma > 1e-6:
            result[i] = (arr[i] - mu) / sigma
    return result
