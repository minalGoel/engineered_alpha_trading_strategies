"""multi_timeframe_confirmation_v1 — Multi-timeframe EMA trend + RSI pullback entry on NIFTY.

Signal: buy CE when EMA(60) > EMA(180) (5-min uptrend) AND RSI(12) crosses from below 30
to above 30 (1-min micro-pullback ending) AND close > VWAP.
Mirror conditions for buy PE.

Adaptation: original traded top-30 FnO stocks on 1-min bars using EMA_9/21 on 5-min bars
as trend filter and RSI(7) on 1-min as entry trigger. Converted to NIFTY index at 5-second
bars: EMAs compressed to 5-min/15-min windows (NOT blind 12x); RSI scaled to 1-min
equivalent at 5s resolution (12 bars = 60s); ATR stops replaced with fixed option
premium points calibrated to NIFTY micro-move magnitudes.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. Seed = first value; fills forward from bar 0."""
    out = np.empty(len(arr), dtype=np.float64)
    if len(arr) == 0:
        return out
    k = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i] * k + out[i - 1] * (1.0 - k)
    return out


def _rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothed RSI. Returns 50.0 before warmup."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[period] = np.mean(gain[1 : period + 1])
    avg_loss[period] = np.mean(loss[1 : period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss[period:] > 0, avg_gain[period:] / avg_loss[period:], np.inf)
        rsi[period:] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


class Strategy(BaseStrategy):
    """Multi-timeframe EMA trend + RSI micro-pullback entry on NIFTY.

    Higher-TF filter: EMA(60) vs EMA(180) = 5-min vs 15-min trend alignment.
    Lower-TF trigger: RSI(12) = 1-min pullback detection (12 bars × 5s = 60s).
    Directional confirmation: close vs session VWAP.
    """

    name = "multi_timeframe_confirmation_v1"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min for EMA warmup
    session_end_minutes = 920     # 15:20 IST
    max_lookback = 180            # 15-min warmup for EMA(180) to stabilize
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_oversold",   30.0, 20.0, 40.0),
            TunableParam("rsi_overbought", 70.0, 60.0, 80.0),
            TunableParam("stop_pts",        3.0,  2.0,  6.0),
            TunableParam("target_pts",      5.0,  3.0,  8.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Extract arrays (forward-fill before numpy) ──
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = spot_df["volume"].fill_null(0).cast(pl.Int64).to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Tunable thresholds ──
        rsi_oversold  = params.get("rsi_oversold",   30.0)
        rsi_overbought = params.get("rsi_overbought", 70.0)
        stop_pts  = params.get("stop_pts",  3.0)
        target_pts = params.get("target_pts", 5.0)

        # ── Session VWAP (cumulative per day, resets at session open) ──
        df_vwap = spot_df.with_columns(
            (pl.col("close") * pl.col("volume").cast(pl.Float64)).alias("_pv"),
        ).with_columns(
            pl.col("_pv").cum_sum().over("day_id").alias("_cum_pv"),
            pl.col("volume").cast(pl.Float64).cum_sum().over("day_id").alias("_cum_v"),
        ).with_columns(
            (pl.col("_cum_pv") / pl.col("_cum_v").clip(lower_bound=1.0)).alias("_vwap"),
        )
        vwap = (
            df_vwap["_vwap"]
            .fill_null(strategy="forward")
            .fill_null(float(close[0]) if n > 0 else 0.0)
            .to_numpy()
        )

        # ── EMA indicators (5-min and 15-min context filters) ──
        # EMA(60)  = 5 min at 5s bars  — fast context filter
        # EMA(180) = 15 min at 5s bars — slow context filter
        ema_fast = _ema(close, 60)
        ema_slow = _ema(close, 180)

        # ── RSI(12) — 1-minute pullback detector ──
        # 12 bars × 5s = 60s = 1 minute; detects the micro-pullback within the 5-min trend
        rsi_arr = _rsi(close, 12)

        # ── RSI crossing detection ──
        # Crossing: RSI was outside the threshold on the PREVIOUS bar, now inside
        rsi_prev = np.roll(rsi_arr, 1)
        rsi_prev[0] = 50.0   # neutral seed for bar 0

        # RSI crosses from below oversold back above (pullback ending, buy CE)
        rsi_cross_up = (rsi_prev < rsi_oversold) & (rsi_arr >= rsi_oversold)

        # RSI crosses from above overbought back below (mini-rally exhaustion, buy PE)
        rsi_cross_down = (rsi_prev > rsi_overbought) & (rsi_arr <= rsi_overbought)

        # ── Session filter ──
        in_session = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)

        # ── EMA trend direction ──
        uptrend   = ema_fast > ema_slow   # 5-min EMA above 15-min EMA
        downtrend = ema_fast < ema_slow   # 5-min EMA below 15-min EMA

        # ── VWAP directional confirmation ──
        above_vwap = close > vwap
        below_vwap = close < vwap

        # ── Entry signals ──
        # BUY CE: 5-min uptrend + RSI crosses from oversold back up + price above VWAP
        buy_ce = in_session & uptrend & rsi_cross_up & above_vwap

        # BUY PE: 5-min downtrend + RSI crosses from overbought back down + price below VWAP
        buy_pe = in_session & downtrend & rsi_cross_down & below_vwap

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,              # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
