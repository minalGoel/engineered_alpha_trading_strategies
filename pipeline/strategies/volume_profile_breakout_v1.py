"""Volume Profile Breakout — 5-second NIFTY index options.

Converted from: trading_strategies/unique_strategies_all/Strategy_74.json

Mechanism:
    NIFTY builds an intraday volume profile as the session progresses. The 70%
    value area (VAH/VAL) acts as the institutional fair-value zone where VWAP/TWAP
    algorithms recycle orders. When NIFTY breaks above VAH on two consecutive 5-second
    bars with a volume surge (>1.5× 20-min rolling average), resting sell orders at the
    ceiling have been absorbed and HFT/momentum algorithms cascade in, generating a
    15-25 spot-point burst in the first 30-60 seconds. We capture this initial burst.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_running_volume_profile(
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
    time_minutes: np.ndarray,
    bucket_size: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute running intraday volume profile per bar.

    Returns (poc, vah, val) arrays:
    - poc: Point of Control — 10-pt bucket with highest accumulated volume today
    - vah: Value Area High — upper bound of 70% volume zone
    - val: Value Area Low  — lower bound of 70% volume zone

    Profile accumulates from 09:15 (time_minutes >= 555). Resets each day.
    Returns close price as fallback when fewer than 3 buckets exist.
    """
    n = len(close)
    poc = close.copy()
    vah = close.copy()
    val = close.copy()

    current_day = -1
    bucket_vol: dict[float, float] = {}

    for i in range(n):
        d = int(day_id[i])
        if d != current_day:
            bucket_vol = {}
            current_day = d

        if time_minutes[i] >= 555 and volume[i] > 0:
            bucket = round(close[i] / bucket_size) * bucket_size
            bucket_vol[bucket] = bucket_vol.get(bucket, 0.0) + float(volume[i])

        if len(bucket_vol) < 3:
            # Not enough profile data — use close as neutral fallback
            poc[i] = close[i]
            vah[i] = close[i]
            val[i] = close[i]
            continue

        # POC = bucket with highest accumulated volume
        poc_level = max(bucket_vol, key=lambda k: bucket_vol[k])
        poc[i] = poc_level

        # 70% value area: accumulate highest-volume buckets until 70% of total
        total_vol = sum(bucket_vol.values())
        target_70 = 0.70 * total_vol
        sorted_buckets = sorted(bucket_vol.items(), key=lambda x: -x[1])
        cum = 0.0
        va_prices: list[float] = []
        for price, bvol in sorted_buckets:
            cum += bvol
            va_prices.append(price)
            if cum >= target_70:
                break

        vah[i] = max(va_prices)
        val[i] = min(va_prices)

    return poc, vah, val


class Strategy(BaseStrategy):
    name = "volume_profile_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 585   # 09:45 IST — 30-min wait for profile to develop
    session_end_minutes = 870     # 14:30 IST — avoid end-of-day noise
    max_trades_per_day = 5
    max_lookback = 240            # 20-min warmup for vol_ratio baseline

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_ratio_threshold", 1.5, 1.1, 2.5),
            TunableParam("vix_max", 24.0, 18.0, 30.0),
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

        # Forward-fill NaN before converting to numpy
        close = spot_df.select(pl.col("close").forward_fill()).to_series().to_numpy()
        volume = (
            spot_df.select(pl.col("volume").fill_null(0)).to_series().to_numpy().astype(float)
        )
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        vol_ratio_thr = params.get("vol_ratio_threshold", 1.5)
        vix_max_val = params.get("vix_max", 24.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 7.0)

        # ── VIX (aligned to spot bars via asof join) ────────────────────────
        vix_close = np.full(n, 15.0)
        if vix_df is not None and not vix_df.is_empty():
            vix_joined = spot_df.select("datetime").join_asof(
                vix_df.select(["datetime", pl.col("close").alias("vix_close")]).sort("datetime"),
                on="datetime",
                strategy="backward",
            )
            vix_close = vix_joined["vix_close"].fill_null(15.0).to_numpy()

        # ── Running intraday volume profile ─────────────────────────────────
        # bucket_size=10 pt gives 10-20 buckets across NIFTY's typical 100-200 pt daily range
        poc, vah, val = _compute_running_volume_profile(
            close, volume, day_id, time_min, bucket_size=10.0
        )

        # ── Rolling 240-bar (20-min) volume average — O(n) running sum ──────
        vol_avg = np.ones(n)
        running_sum = 0.0
        for i in range(n):
            running_sum += volume[i]
            if i >= 240:
                running_sum -= volume[i - 240]
            count = min(i + 1, 240)
            vol_avg[i] = running_sum / count if running_sum > 0 else 1.0

        vol_ratio = np.where(vol_avg > 0, volume / vol_avg, 0.0)

        # ── 2-bar breakout confirmation ──────────────────────────────────────
        above_vah = close > vah
        below_val = close < val

        above_vah_prev = np.zeros(n, dtype=bool)
        below_val_prev = np.zeros(n, dtype=bool)
        above_vah_prev[1:] = above_vah[:-1]
        below_val_prev[1:] = below_val[:-1]

        # ── Bar-over-bar direction (momentum confirmation) ───────────────────
        ret_1 = np.zeros(n)
        ret_1[1:] = close[1:] - close[:-1]

        # ── Filters ─────────────────────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok = vix_close < vix_max_val

        # ── Signals ─────────────────────────────────────────────────────────
        # buy_ce: NIFTY breaks above VAH on volume surge, 2-bar confirmation
        buy_ce = (
            in_session
            & vix_ok
            & above_vah
            & above_vah_prev
            & (vol_ratio >= vol_ratio_thr)
            & (ret_1 > 0)
        )

        # buy_pe: NIFTY breaks below VAL on volume surge, 2-bar confirmation
        buy_pe = (
            in_session
            & vix_ok
            & below_val
            & below_val_prev
            & (vol_ratio >= vol_ratio_thr)
            & (ret_1 < 0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,   # 60 seconds = 12 bars × 5s
            max_trades_per_day=self.max_trades_per_day,
        )
