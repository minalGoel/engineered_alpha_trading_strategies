"""maximum_entropy_v1 — Maximum Entropy Tail-Event Mean Reversion on NIFTY

Mechanism:
    On NIFTY, a 5-second return that falls beyond the 95th (or below the 5th)
    percentile of the past 5-minute return distribution is statistically anomalous —
    typically a stop-loss cascade or momentum-algo overshoot on thin liquidity.
    The Cornish-Fisher expansion adjusts the tail threshold for the current skewness
    of the return distribution (unlike a plain z-score which assumes normality),
    correctly identifying 'truly extreme' events. NIFTY's structural mean-reversion
    forces (VWAP-benchmarked algos, market-maker delta-hedging) fade these overshoots
    within 30-90 seconds. We enter at the close of the extreme bar, in the direction
    of VWAP reversion.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _rolling_mean_std_skew(arr: np.ndarray, window: int):
    """Compute rolling mean, std, skewness over a fixed lookback window.

    Returns three arrays of length len(arr). Values at indices < window are 0.
    """
    n = len(arr)
    roll_mean = np.zeros(n)
    roll_std = np.full(n, 1e-9)
    roll_skew = np.zeros(n)

    for i in range(window, n):
        w = arr[i - window:i]
        m = float(np.mean(w))
        s = float(np.std(w))
        roll_mean[i] = m
        if s > 1e-9:
            roll_std[i] = s
            roll_skew[i] = float(np.mean(((w - m) / s) ** 3))
        else:
            roll_std[i] = 1e-9
            roll_skew[i] = 0.0

    return roll_mean, roll_std, roll_skew


class Strategy(BaseStrategy):
    name = "maximum_entropy_v1"
    underlying = "NIFTY"
    # Start at 09:30 to allow 60-bar warmup from 09:20 open (120 bars = 10 min gap)
    session_start_minutes = 570   # 09:30 IST
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 120            # 120 bars × 5s = 10 min warmup
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            # z-score corresponding to the tail percentile (1.645 → P5/P95, 1.28 → P10/P90)
            TunableParam("tail_z", 1.645, 1.28, 2.33),
            # VIX ceiling: above this, extremes are common noise, not tradeable reversions
            TunableParam("vix_max", 22.0, 16.0, 28.0),
            TunableParam("stop_pts", 3.0, 2.0, 6.0),
            TunableParam("target_pts", 5.0, 3.0, 10.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict,
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Parameters ──────────────────────────────────────────────────────────
        tail_z = float(params.get("tail_z", 1.645))
        vix_max = float(params.get("vix_max", 22.0))
        stop_pts = float(params.get("stop_pts", 3.0))
        target_pts = float(params.get("target_pts", 5.0))

        # ── Spot data ────────────────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── 5-second returns ─────────────────────────────────────────────────────
        ret = np.zeros(n)
        denom = np.where(close[:-1] != 0.0, close[:-1], 1.0)
        ret[1:] = (close[1:] - close[:-1]) / denom

        # ── Rolling distribution (5-minute = 60 bars) ───────────────────────────
        WINDOW = 60
        roll_mean, roll_std, roll_skew = _rolling_mean_std_skew(ret, WINDOW)

        # ── Cornish-Fisher tail thresholds ───────────────────────────────────────
        # Adjusts the standard normal quantile by the rolling skewness, giving an
        # asymmetric tail that better matches the actual distribution than a plain z-score.
        # CF formula: z_adj = z + (skew/6) * (z^2 - 1)
        z_lo = -tail_z
        z_hi = +tail_z
        cf_lo = z_lo + (roll_skew / 6.0) * (z_lo ** 2 - 1.0)
        cf_hi = z_hi + (roll_skew / 6.0) * (z_hi ** 2 - 1.0)

        maxent_p5 = roll_mean + cf_lo * roll_std     # P5 threshold
        maxent_p95 = roll_mean + cf_hi * roll_std    # P95 threshold

        # ── Session VWAP (cumulative, reset each day) ────────────────────────────
        vwap = np.zeros(n)
        cum_pv = 0.0
        cum_v = 0.0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_pv = 0.0
                cum_v = 0.0
                prev_day = int(day_id[i])
            cum_pv += close[i] * volume[i]
            cum_v += volume[i]
            vwap[i] = cum_pv / cum_v if cum_v > 0.0 else close[i]

        # ── VIX filter ───────────────────────────────────────────────────────────
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

        # ── Signal masks ─────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        low_vix = vix_close < vix_max
        has_dist = roll_std > 1e-9   # distribution is meaningful

        # Extreme DOWN return + below VWAP → mean reversion expected UP → buy CE
        buy_ce = in_session & low_vix & has_dist & (ret < maxent_p5) & (close < vwap)

        # Extreme UP return + above VWAP → mean reversion expected DOWN → buy PE
        buy_pe = in_session & low_vix & has_dist & (ret > maxent_p95) & (close > vwap)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,        # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
