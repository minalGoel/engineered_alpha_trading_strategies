"""VWAP Slope Momentum — NIFTY 5-second index options.

Mechanism: When NIFTY's session VWAP sustains an elevated linear-regression slope
(z-score > threshold persisting 25+ seconds), it reflects active institutional TWAP
execution. NIFTY futures market makers quote against VWAP benchmarks, so a persistently
steep VWAP slope forces their delta-hedging in the same direction, amplifying the initial
institutional push. The 5-bar persistence requirement (25 seconds) filters single-candle
noise and confirms the slope is institutionally sustained.

Hold time: 30-90 seconds (6-18 bars).
Stops: 4 option pts. Targets: 6 option pts.
"""
from __future__ import annotations
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_vwap(close: np.ndarray, volume: np.ndarray, day_id: np.ndarray) -> np.ndarray:
    """Session VWAP that resets at the start of each trading day."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_vol = 0.0
    cur_day = day_id[0]
    for i in range(n):
        if day_id[i] != cur_day:
            cum_pv = 0.0
            cum_vol = 0.0
            cur_day = day_id[i]
        cum_pv += close[i] * volume[i]
        cum_vol += volume[i]
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else close[i]
    return vwap


def _linear_regression_slope(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling linear regression slope over `window` bars."""
    n = len(values)
    slopes = np.zeros(n)
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    ss_xx = float(np.sum((x - x_mean) ** 2))
    for i in range(window - 1, n):
        y = values[i - window + 1: i + 1]
        y_mean = y.mean()
        ss_xy = float(np.sum((x - x_mean) * (y - y_mean)))
        slopes[i] = ss_xy / ss_xx if ss_xx > 0 else 0.0
    return slopes


def _rolling_zscore(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling z-score over `window` bars."""
    n = len(values)
    zscore = np.zeros(n)
    for i in range(window - 1, n):
        w = values[i - window + 1: i + 1]
        mu = w.mean()
        sigma = w.std()
        zscore[i] = (values[i] - mu) / sigma if sigma > 1e-10 else 0.0
    return zscore


class Strategy(BaseStrategy):
    name = "vwap_slope_momentum_v1"
    underlying = "NIFTY"
    session_start_minutes = 570    # 09:30 IST — allow 15 min for VWAP to stabilise
    session_end_minutes = 920      # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 300             # 25 min warmup: 240 (zscore) + 24 (slope) + buffer

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("slope_zscore_threshold", 1.5, 1.0, 2.5),
            TunableParam("stop_pts", 4.0, 2.0, 7.0),
            TunableParam("target_pts", 6.0, 4.0, 12.0),
        ]

    def compute(self, spot_df, option_df, vix_df, params) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill NaN in Polars before to_numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        day_id = spot_df["day_id"].fill_null(0).to_numpy()
        time_min = spot_df["time_minutes"].fill_null(0).to_numpy()

        threshold = params.get("slope_zscore_threshold", 1.5)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── VWAP (session-cumulative, resets each day) ──
        vwap = _compute_vwap(close, volume, day_id)

        # ── VWAP slope in bps-per-bar (24 bars = 2 min linear regression) ──
        # 24 bars chosen to match hold time (30-90s): gives a 2-min slope estimate
        # that detects early institutional acceleration without excessive lag.
        raw_slope = _linear_regression_slope(vwap, 24)
        vwap_slope_bps = np.where(close > 0, raw_slope / close * 10_000.0, 0.0)

        # ── Z-score over 240 bars (20 min) — normalises for time-of-day flattening ──
        vwap_slope_zscore = _rolling_zscore(vwap_slope_bps, 240)

        # ── Price relative to VWAP ──
        price_above_vwap = close > vwap
        price_below_vwap = close < vwap

        # ── VIX filter ──
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Session filter ──
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── 5-bar persistence filter (25 seconds) ──
        # Directly maps the original strategy's "5 consecutive bar" confirmation.
        # At 5-second bars, 5 bars = 25 seconds of sustained VWAP slope signal,
        # confirming institutional flow rather than a single-candle spike.
        bull_persist = np.zeros(n, dtype=bool)
        bear_persist = np.zeros(n, dtype=bool)
        for i in range(4, n):
            bull_persist[i] = bool(np.all(vwap_slope_zscore[i - 4: i + 1] > threshold))
            bear_persist[i] = bool(np.all(vwap_slope_zscore[i - 4: i + 1] < -threshold))

        # ── Entry signals ──
        vix_ok = vix_close < 24.0
        buy_ce = in_session & bull_persist & price_above_vwap & vix_ok
        buy_pe = in_session & bear_persist & price_below_vwap & vix_ok

        # ── Signal-based exit: slope momentum fades toward zero ──
        sell_ce = vwap_slope_zscore < 0.5
        sell_pe = vwap_slope_zscore > -0.5

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=sell_ce,
            sell_pe=sell_pe,
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,             # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
