"""robust_regression_v1 — Huber IRLS Trend Dip Entry on NIFTY 5-second bars.

Mechanism:
    Institutional TWAP/VWAP algos create persistent 3-7 minute micro-trends on NIFTY.
    Single spike bars from block trades distort OLS regression; Huber IRLS down-weights
    those outlier bars to reveal the genuine institutional flow slope. When the robust
    slope t-statistic > 2.0 (60-bar / 5-min window) and price dips below the regression
    line, we enter in the trend direction expecting the ongoing TWAP execution to absorb
    the imbalance and carry spot through the trend line within 30-90 seconds.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_huber_regression(
    close: np.ndarray,
    window: int = 60,
    epsilon: float = 1.35,
    n_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rolling Huber IRLS regression of close on bar_index.

    Returns:
        t_stat_arr   — slope t-statistic at each bar (0 before warmup)
        residual_arr — residual at last bar of each window (close - trend_line)
        mad_arr      — robust MAD of residuals (noise level)
    """
    n = len(close)
    t_stat_arr = np.zeros(n)
    residual_arr = np.zeros(n)
    mad_arr = np.ones(n)

    x = np.arange(window, dtype=float)

    for i in range(window - 1, n):
        y = close[i - window + 1: i + 1].copy()

        w = np.ones(window)
        slope_i = 0.0
        intercept_i = float(y.mean())
        sw = float(window)
        denom = 1.0

        for _ in range(n_iter):
            sw = float(w.sum())
            swx = float(np.dot(w, x))
            swx2 = float(np.dot(w, x * x))
            swy = float(np.dot(w, y))
            swxy = float(np.dot(w, x * y))

            denom = sw * swx2 - swx * swx
            if abs(denom) < 1e-10:
                slope_i = 0.0
                intercept_i = swy / max(sw, 1e-9)
                break

            slope_i = (sw * swxy - swx * swy) / denom
            intercept_i = (swy - slope_i * swx) / sw

            res = y - (intercept_i + slope_i * x)
            # MAD around median (robust std)
            med_res = float(np.median(res))
            mad = float(np.median(np.abs(res - med_res))) * 1.4826
            if mad < 1e-6:
                mad = 1e-6

            r_scaled = np.abs(res) / (epsilon * mad)
            w = np.where(r_scaled <= 1.0, 1.0, 1.0 / np.maximum(r_scaled, 1e-9))

        # Final residuals with converged slope/intercept
        res = y - (intercept_i + slope_i * x)
        residual_arr[i] = float(res[-1])

        med_res = float(np.median(res))
        mad_final = float(np.median(np.abs(res - med_res))) * 1.4826
        if mad_final < 1e-6:
            mad_final = 1e-6
        mad_arr[i] = mad_final

        # Variance of slope: sigma^2 * sw / denom  (WLS formula)
        sse = float(np.dot(w, res * res))
        sigma2 = sse / max(window - 2, 1)
        se_slope = float(np.sqrt(sigma2 * sw / max(abs(denom), 1e-12)))
        t_stat_arr[i] = slope_i / max(se_slope, 1e-12)

    return t_stat_arr, residual_arr, mad_arr


def _compute_vwap(
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Intraday VWAP, reset each day."""
    n = len(close)
    vwap = np.zeros(n)
    cum_pv = 0.0
    cum_v = 0.0
    prev_day = -1

    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_v = 0.0
            prev_day = int(day_id[i])
        cum_pv += float(close[i]) * float(volume[i])
        cum_v += float(volume[i])
        vwap[i] = cum_pv / max(cum_v, 1e-9)

    return vwap


class Strategy(BaseStrategy):
    name = "robust_regression_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — 5-min warmup after 09:15 open
    session_end_minutes = 925     # 15:25 IST
    max_lookback = 60             # 60 bars × 5s = 5-minute regression warmup
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("t_stat_threshold", 2.0, 1.5, 3.5),
            TunableParam("mad_limit", 2.0, 1.5, 3.0),
            TunableParam("stop_pts", 5.0, 2.0, 7.0),
            TunableParam("target_pts", 8.0, 4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # --- Extract spot arrays (forward-fill NaN before numpy) ---
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Float64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # --- Parameters ---
        t_thresh = float(params.get("t_stat_threshold", 2.0))
        mad_limit = float(params.get("mad_limit", 2.0))
        stop_pts = float(params.get("stop_pts", 5.0))
        target_pts = float(params.get("target_pts", 8.0))

        # --- VIX filter ---
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(
                    ["datetime", pl.col("close").alias("vix_close")]
                ).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # --- VWAP (session direction filter) ---
        vwap = _compute_vwap(close, volume, day_id)

        # --- Rolling Huber IRLS regression ---
        t_stat, residual, mad = _rolling_huber_regression(close, window=60)

        # --- Session + VIX mask ---
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < 25.0

        # --- Entry signals ---
        # Buy CE: confirmed robust uptrend (t > threshold)
        #         price dipped below trend line (residual < 0)
        #         dip not too extreme (residual > -mad_limit * mad)
        #         session trend agrees (close > vwap)
        buy_ce = (
            in_session
            & vix_ok
            & (t_stat > t_thresh)
            & (residual < 0.0)
            & (residual > -mad_limit * mad)
            & (close > vwap)
        )

        # Buy PE: confirmed robust downtrend (t < -threshold)
        #         price ripped above trend line (residual > 0)
        #         rip not too extreme
        #         session trend agrees (close < vwap)
        buy_pe = (
            in_session
            & vix_ok
            & (t_stat < -t_thresh)
            & (residual > 0.0)
            & (residual < mad_limit * mad)
            & (close < vwap)
        )

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
