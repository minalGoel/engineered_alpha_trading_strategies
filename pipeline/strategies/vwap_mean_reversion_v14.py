"""
vwap_mean_reversion_v14 — EMA-Smoothed Volume Exhaustion + Inflection Entry

Mechanism: On NIFTY, when the index is 2.2+ sigma below session VWAP while
1-minute EMA-smoothed volume is 1.8x the 20-minute EMA baseline (confirming a
sustained 60-second sell cascade), the entry fires only on the first bar where
the z-score begins improving (zscore_delta > 0). This inflection marks the exact
bar where passive market-makers absorb the cascade's tail and the order book
starts replenishing from the bid side. Hold 30-90 seconds for 12-20 spot point
snap-back (~7 option pts at delta 0.5).

Key differentiators vs other VWAP variants:
- v12: flush-ongoing (bar_return < 0); single-bar raw volume
- v13: 2-bar z-score persistence; 60-min stddev; raw rel_volume
- v18: deviation momentum (rate of z-score change, not inflection)
- v19: range contraction microstructure (bar_range + close_vs_low)
- v14: EMA-smoothed volume ratio (1-min vs 20-min) + zscore_delta inflection
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average with alpha = 2 / (span + 1). NaN-safe."""
    alpha = 2.0 / (span + 1)
    out = np.full(len(arr), np.nan)
    started = False
    for i in range(len(arr)):
        v = arr[i]
        if np.isnan(v):
            if started:
                out[i] = out[i - 1]  # forward-fill NaN
            continue
        if not started:
            out[i] = v
            started = True
        else:
            out[i] = alpha * v + (1.0 - alpha) * out[i - 1]
    return out


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling population standard deviation using stride tricks (vectorized)."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n >= window:
        views = np.lib.stride_tricks.sliding_window_view(arr, window)
        out[window - 1:] = np.std(views, axis=1, ddof=0)
    return out


class Strategy(BaseStrategy):
    name = "vwap_mean_reversion_v14"
    underlying = "NIFTY"
    session_start_minutes = 560    # 09:20 IST
    session_end_minutes = 920      # 15:20 IST
    max_trades_per_day = 6
    max_lookback = 480             # 40 min warmup (420 stddev bars + buffer)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_threshold", 2.2, 1.8, 3.0),
            TunableParam("vol_ratio_threshold", 1.8, 1.3, 2.5),
            TunableParam("vix_max", 28.0, 20.0, 35.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Parameters ──────────────────────────────────────────────────────
        zscore_thr = float(params.get("zscore_threshold", 2.2))
        vol_ratio_thr = float(params.get("vol_ratio_threshold", 1.8))
        vix_max = float(params.get("vix_max", 28.0))

        # ── Spot columns ────────────────────────────────────────────────────
        close = (
            spot_df["close"]
            .fill_null(strategy="forward")
            .to_numpy()
            .astype(np.float64)
        )
        volume = (
            spot_df["volume"]
            .fill_null(0)
            .cast(pl.Float64)
            .to_numpy()
            .astype(np.float64)
        )
        day_id = spot_df["day_id"].to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── VIX (join asof to spot bars) ─────────────────────────────────────
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

        # ── Cumulative VWAP (reset per session day) ──────────────────────────
        vwap = np.zeros(n)
        cum_tpv = 0.0
        cum_vol = 0.0
        prev_day = day_id[0] - 1
        for i in range(n):
            if day_id[i] != prev_day:
                cum_tpv = 0.0
                cum_vol = 0.0
                prev_day = day_id[i]
            cum_tpv += close[i] * volume[i]
            cum_vol += volume[i]
            vwap[i] = cum_tpv / cum_vol if cum_vol > 0.0 else close[i]

        # ── VWAP z-score (35-min rolling stddev) ────────────────────────────
        raw_std = _rolling_std(close, 420)
        vwap_std = np.where(
            np.isnan(raw_std), 1.0, np.maximum(raw_std, 0.5)
        )
        zscore = (close - vwap) / vwap_std

        # ── EMA-smoothed volume ratio ────────────────────────────────────────
        vol_ema_12 = _ema(volume, 12)     # ~1-min smoothed
        vol_ema_240 = _ema(volume, 240)   # ~20-min baseline

        # Replace NaN from EMA warmup with neutral values
        vol_ema_12 = np.where(np.isnan(vol_ema_12), 0.0, vol_ema_12)
        vol_ema_240 = np.where(np.isnan(vol_ema_240), 1.0, vol_ema_240)

        smoothed_rel_vol = vol_ema_12 / np.maximum(vol_ema_240, 1.0)

        # ── Z-score delta (inflection detector) ─────────────────────────────
        zscore_delta = np.zeros(n)
        zscore_delta[1:] = zscore[1:] - zscore[:-1]
        # bar 0 stays 0 (neutral — can't compute change without prior bar)

        # ── Session filter ───────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ────────────────────────────────────────────────────
        # buy_ce: deep negative z-score + sustained high volume + z-score inflecting up
        buy_ce = (
            in_session
            & (zscore < -zscore_thr)
            & (smoothed_rel_vol > vol_ratio_thr)
            & (zscore_delta > 0.0)
            & (vix_close < vix_max)
        )

        # buy_pe: deep positive z-score + sustained high volume + z-score inflecting down
        buy_pe = (
            in_session
            & (zscore > zscore_thr)
            & (smoothed_rel_vol > vol_ratio_thr)
            & (zscore_delta < 0.0)
            & (vix_close < vix_max)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, 5.0),
            target_points=np.full(n, 7.0),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,
            max_trades_per_day=self.max_trades_per_day,
        )
