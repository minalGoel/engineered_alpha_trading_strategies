"""rsi_connors_pullback_v41 — Connors RSI(6) Micro-Pullback Within Intraday Trend

Adapted from Connors RSI(2) strategy (original: FNO stocks, 15-min, SMA(200) trend filter).
Core thesis: NIFTY institutional TWAP/VWAP algos create persistent 20-45 min intraday trends.
Within these trends, RSI(6) < 25 signals micro-pullback exhaustion — the institutional buy side
resumes and NIFTY snaps back in trend direction within 15-60 seconds.
"""
from __future__ import annotations
import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder-smoothed RSI. Returns array of same length, pre-filled with 50.0."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n < period + 1:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]

    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Seed with simple average over first window
    avg_gain = np.mean(gains[1: period + 1])
    avg_loss = np.mean(losses[1: period + 1])

    if avg_loss == 0.0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    # Wilder smoothing for remaining bars
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    return rsi


def _compute_sma(close: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average; NaN before warmup."""
    n = len(close)
    sma = np.full(n, np.nan)
    if n < period:
        return sma
    cumsum = np.cumsum(close)
    # sma[i] = mean(close[i-period:i]) for i >= period
    sma[period:] = (cumsum[period:] - cumsum[:n - period]) / period
    return sma


class Strategy(BaseStrategy):
    name = "rsi_connors_pullback_v41"
    underlying = "NIFTY"
    session_start_minutes = 570   # 09:30 IST — allow 30-min trend to form before trading
    session_end_minutes = 920     # 15:20 IST
    max_trades_per_day = 8
    max_lookback = 400            # 400 bars ≈ 33 min warmup to seed SMA(360)

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_oversold",   25.0, 10.0, 35.0),
            TunableParam("rsi_overbought", 75.0, 65.0, 90.0),
            TunableParam("vix_max",        20.0, 14.0, 28.0),
            TunableParam("stop_pts",        3.0,  2.0,  6.0),
            TunableParam("target_pts",      5.0,  3.0, 10.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Parameters ──────────────────────────────────────────────────────────
        rsi_oversold   = params.get("rsi_oversold",   25.0)
        rsi_overbought = params.get("rsi_overbought", 75.0)
        vix_max        = params.get("vix_max",        20.0)
        stop_pts       = params.get("stop_pts",        3.0)
        target_pts     = params.get("target_pts",      5.0)

        # ── Spot close (forward-filled in Polars before numpy) ───────────────────
        close = spot_df.select(
            pl.col("close").forward_fill()
        ).to_series().to_numpy()

        time_min = spot_df["time_minutes"].to_numpy()

        # ── SMA(360): 30-min intraday trend filter ───────────────────────────────
        sma_360 = _compute_sma(close, 360)

        # ── RSI(6): 30-second exhaustion trigger ─────────────────────────────────
        rsi_6 = _compute_rsi(close, 6)

        # ── VIX: filter extremely volatile regimes ───────────────────────────────
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

        # ── Masks ────────────────────────────────────────────────────────────────
        in_session  = (time_min >= self.session_start_minutes) & (time_min < self.session_end_minutes)
        vix_ok      = vix_close < vix_max
        trend_valid = ~np.isnan(sma_360)   # warmup guard

        trend_up   = trend_valid & (close > sma_360)
        trend_down = trend_valid & (close < sma_360)

        # Buy CE: 30-min uptrend + 30s RSI oversold (micro-pullback exhausted)
        buy_ce = in_session & vix_ok & trend_up   & (rsi_6 < rsi_oversold)
        # Buy PE: 30-min downtrend + 30s RSI overbought (micro-bounce exhausted)
        buy_pe = in_session & vix_ok & trend_down & (rsi_6 > rsi_overbought)

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=12,             # 60 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
