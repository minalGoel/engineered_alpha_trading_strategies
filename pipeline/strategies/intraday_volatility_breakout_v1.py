"""intraday_volatility_breakout_v1 — ATR Expansion Breakout on NIFTY

Mechanism: When NIFTY's 1-minute ATR (12 bars at 5s) exceeds its 5-minute baseline ATR
(60 bars at 5s) by 2x or more, institutional order flow or macro news has shifted
microstructure into a high-volatility regime. Options market makers short gamma respond by
delta-hedging in the direction of the move, amplifying the burst for 30-90 seconds. We enter
in the direction of the triggering bar (confirmed by relative volume) to capture the first
leg of this institutional feedback loop.

Original: Strategy_42.json — intraday ATR expansion on FnO stocks (1-min bars, 10-30 min hold)
Converted: NIFTY 5s bars, 30-90s hold, ATR lookbacks compressed to match hold time
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Compute True Range series."""
    n = len(high)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    return tr


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean; values for i < window are 0.0."""
    n = len(arr)
    out = np.zeros(n)
    for i in range(window, n):
        out[i] = np.mean(arr[i - window: i])
    return out


def _session_vwap(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    day_id: np.ndarray,
) -> np.ndarray:
    """Session cumulative VWAP — resets at each new day_id."""
    n = len(close)
    vwap = np.empty(n)
    cum_pv = 0.0
    cum_vol = 0.0
    prev_day = -1
    for i in range(n):
        if day_id[i] != prev_day:
            cum_pv = 0.0
            cum_vol = 0.0
            prev_day = day_id[i]
        typ = (high[i] + low[i] + close[i]) / 3.0
        cum_pv += typ * volume[i]
        cum_vol += volume[i]
        vwap[i] = cum_pv / cum_vol if cum_vol > 0 else close[i]
    return vwap


class Strategy(BaseStrategy):
    name = "intraday_volatility_breakout_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 6
    max_lookback = 120            # 10 min warmup — enough for ATR_60 + SMA_60 baselines

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_expansion_threshold", 2.0, 1.5, 3.0),
            TunableParam("rel_volume_threshold", 1.8, 1.2, 3.0),
            TunableParam("stop_pts", 5.0, 2.0, 8.0),
            TunableParam("target_pts", 8.0, 4.0, 14.0),
            TunableParam("vix_max", 26.0, 18.0, 35.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract spot arrays (forward-fill NaN before converting) ──────────
        high = spot_df["high"].fill_null(strategy="forward").to_numpy().astype(float)
        low = spot_df["low"].fill_null(strategy="forward").to_numpy().astype(float)
        close = spot_df["close"].fill_null(strategy="forward").to_numpy().astype(float)
        open_ = spot_df["open"].fill_null(strategy="forward").to_numpy().astype(float)
        volume = spot_df["volume"].fill_null(0).to_numpy().astype(float)
        time_min = spot_df["time_minutes"].to_numpy()
        day_id = spot_df["day_id"].to_numpy()

        # ── Parameters ───────────────────────────────────────────────────────
        exp_thresh = params.get("vol_expansion_threshold", 2.0)
        vol_thresh = params.get("rel_volume_threshold", 1.8)
        stop_pts = params.get("stop_pts", 5.0)
        target_pts = params.get("target_pts", 8.0)
        vix_max = params.get("vix_max", 26.0)

        # ── VIX (aligned to spot bars) ────────────────────────────────────────
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

        # ── True Range ────────────────────────────────────────────────────────
        tr = _true_range(high, low, close)

        # ── ATR_12: 1-minute realized vol (12 bars × 5s = 60s) ───────────────
        # Trigger indicator — compressed from original ATR_10 on 1-min bars.
        # 1-min lookback captures the FAST burst onset; longer would lag the regime shift.
        atr_12 = _rolling_mean(tr, 12)

        # ── ATR_60: 5-minute baseline vol (60 bars × 5s = 300s) ──────────────
        # Context filter — compressed from original ATR_60 on 1-min bars (60 min).
        # 5-min baseline adapts to intraday vol clusters rather than stale all-day averages.
        atr_60 = _rolling_mean(tr, 60)

        # ── Vol expansion ratio ───────────────────────────────────────────────
        vol_expansion = np.zeros(n)
        for i in range(60, n):
            if atr_60[i] > 0.0:
                vol_expansion[i] = atr_12[i] / atr_60[i]

        # ── Bar direction (sign of close - open) ─────────────────────────────
        bar_dir = np.sign(close - open_)

        # ── Relative volume vs 5-min baseline ────────────────────────────────
        vol_sma_60 = _rolling_mean(volume, 60)
        rel_vol = np.zeros(n)
        for i in range(60, n):
            if vol_sma_60[i] > 0.0:
                rel_vol[i] = volume[i] / vol_sma_60[i]

        # ── Session VWAP ──────────────────────────────────────────────────────
        vwap = _session_vwap(high, low, close, volume, day_id)

        # ── Session and VIX filters ───────────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        vix_ok = vix_close < vix_max

        # ── Entry signals ─────────────────────────────────────────────────────
        # buy_ce: bullish ATR expansion — vol surges, bar is up, price above VWAP
        buy_ce = (
            in_session
            & vix_ok
            & (vol_expansion > exp_thresh)
            & (bar_dir > 0)
            & (rel_vol > vol_thresh)
            & (close > vwap)
        )

        # buy_pe: bearish ATR expansion — vol surges, bar is down, price below VWAP
        buy_pe = (
            in_session
            & vix_ok
            & (vol_expansion > exp_thresh)
            & (bar_dir < 0)
            & (rel_vol > vol_thresh)
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
            time_stop_bars=18,          # 90 seconds — vol expansion plays out fast or not at all
            max_trades_per_day=self.max_trades_per_day,
        )
