"""momentum_exhaustion_vix_reversal — 5-second NIFTY index options strategy.

Thesis: On NIFTY, a sharp vertical price move (RSI(6) at extreme in 30s) with
a volume surge >3x the 10-minute baseline while VIX is rising signals that
directional flow is hitting absorption. Market makers short gamma begin
delta-hedging in the opposite direction within 1-3 bars, creating a fast
mean reversion. We enter on the first confirming reversal tick.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Simple (non-Wilder) RSI for fast exhaustion detection."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi
    delta = np.diff(close)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    for i in range(period, n):
        avg_gain = np.mean(gains[i - period:i])
        avg_loss = np.mean(losses[i - period:i])
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def _sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average using cumsum for efficiency."""
    n = len(arr)
    result = np.full(n, np.nan)
    if n < period:
        return result
    cumsum = np.cumsum(arr)
    result[period - 1] = cumsum[period - 1] / period
    result[period:] = (cumsum[period:] - cumsum[:n - period]) / period
    return result


class Strategy(BaseStrategy):
    name = "momentum_exhaustion_vix_reversal"
    underlying = "NIFTY"
    session_start_minutes = 565   # 09:25 IST — skip first 10 min noise
    session_end_minutes = 920     # 15:20 IST — flatten 10 min before close
    max_trades_per_day = 6
    max_lookback = 144            # 120 bars = 10 min for volume SMA baseline

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_oversold",    20.0,  10.0, 30.0),
            TunableParam("rsi_overbought",  80.0,  70.0, 90.0),
            TunableParam("vol_multiplier",   3.0,   2.0,  5.0),
            TunableParam("stop_pts",         4.0,   2.0,  8.0),
            TunableParam("target_pts",       6.0,   4.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Raw arrays ──────────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        volume = (
            spot_df["volume"]
            .fill_null(0)
            .cast(pl.Float64)
            .to_numpy()
        )
        time_min = spot_df["time_minutes"].to_numpy()

        # ── Parameters ──────────────────────────────────────────────────────
        rsi_oversold   = params.get("rsi_oversold",   20.0)
        rsi_overbought = params.get("rsi_overbought", 80.0)
        vol_multiplier = params.get("vol_multiplier",  3.0)
        stop_pts       = params.get("stop_pts",        4.0)
        target_pts     = params.get("target_pts",      6.0)

        # ── Indicator 1: RSI(6) — 30-second exhaustion detector ────────────
        rsi6 = _compute_rsi(close, 6)

        # ── Indicator 2: Volume climax — current bar > 3x 10-min average ───
        vol_sma120 = _sma(volume, 120)
        # Fill leading NaN with the volume itself (no climax signal until warm)
        vol_baseline = np.where(np.isnan(vol_sma120), volume, vol_sma120)
        # Guard against zero baseline
        vol_baseline = np.where(vol_baseline == 0.0, 1.0, vol_baseline)
        vol_climax = volume > vol_multiplier * vol_baseline

        # ── Indicator 3: VIX slope — VIX rising over last 30 seconds ───────
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

        vix_rising = np.zeros(n, dtype=bool)
        vix_rising[6:] = vix_close[6:] > vix_close[:-6]

        # ── Directional confirmation: first tick of reversal ────────────────
        ret1 = np.zeros(n)
        ret1[1:] = close[1:] - close[:-1]

        # ── Session filter ──────────────────────────────────────────────────
        in_session = (
            (time_min >= self.session_start_minutes)
            & (time_min < self.session_end_minutes)
        )

        # ── Entry signals ───────────────────────────────────────────────────
        # buy_ce: oversold exhaustion bounce — RSI crashed, volume climax, VIX
        # rising (panic selloff absorbed), first uptick = reversal starting
        buy_ce = (
            in_session
            & (rsi6 < rsi_oversold)
            & vol_climax
            & vix_rising
            & (ret1 > 0)
        )

        # buy_pe: overbought exhaustion fade — RSI surged, volume climax, VIX
        # rising (euphoria hitting supply), first downtick = reversal starting
        buy_pe = (
            in_session
            & (rsi6 > rsi_overbought)
            & vol_climax
            & vix_rising
            & (ret1 < 0)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=24,           # 120 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
