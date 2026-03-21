"""rsi_reversal_v1 — RSI Exhaustion Reversion on NIFTY

Mechanism:
On NIFTY, sharp 2-minute directional moves push RSI(24) into extreme territory
(<30 or >70) as stop-loss cascades or momentum-chasing algos overshoot fair value.
When this exhaustion occurs while the index is displaced from its 30-minute EMA,
VWAP-benchmarked institutional algorithms and gamma-hedging desks push the index back
toward the intraday mean within 30-90 seconds.

Converted from: trading_strategies/unique_strategies_all/Strategy_4.json
Original: RSI(14)/EMA(200) on 1-min equity bars, hold 2-15 min.
Adaptation: RSI(24)=2min fast exhaustion trigger; EMA(360)=30min intraday anchor.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from pipeline.strategies.base import BaseStrategy, OptionSignals, TunableParam


def _compute_rsi(close: np.ndarray, period: int) -> np.ndarray:
    """Wilder RSI. Returns 50.0 for bars with insufficient history."""
    n = len(close)
    rsi = np.full(n, 50.0)
    if n <= period:
        return rsi

    delta = np.zeros(n)
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    # Seed with simple average over first period
    avg_gain[period] = np.mean(gain[1 : period + 1])
    avg_loss[period] = np.mean(loss[1 : period + 1])

    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period

    for i in range(period, n):
        if avg_loss[i] == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _compute_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average. Seeded at first bar close."""
    n = len(close)
    ema = np.empty(n)
    ema[0] = close[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "rsi_reversal_v1"
    underlying = "NIFTY"
    session_start_minutes = 560   # 09:20 IST — skip pre-open noise
    session_end_minutes = 925     # 15:25 IST
    max_trades_per_day = 8
    # EMA(360) needs 360 bars warmup = 30 min; add RSI(24) on top → 400 bars
    max_lookback = 400

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_oversold", 30.0, 20.0, 38.0),
            TunableParam("rsi_overbought", 70.0, 62.0, 80.0),
            TunableParam("vix_max", 25.0, 18.0, 30.0),
            TunableParam("stop_pts", 4.0, 2.0, 7.0),
            TunableParam("target_pts", 6.0, 3.0, 12.0),
        ]

    def compute(
        self,
        spot_df: pl.DataFrame,
        option_df: pl.DataFrame,
        vix_df: pl.DataFrame,
        params: dict[str, float],
    ) -> OptionSignals:
        n = len(spot_df)

        # ── Params ──────────────────────────────────────────────────────────
        rsi_oversold = params.get("rsi_oversold", 30.0)
        rsi_overbought = params.get("rsi_overbought", 70.0)
        vix_max = params.get("vix_max", 25.0)
        stop_pts = params.get("stop_pts", 4.0)
        target_pts = params.get("target_pts", 6.0)

        # ── Spot data ────────────────────────────────────────────────────────
        close = spot_df["close"].fill_null(strategy="forward").to_numpy()
        time_min = spot_df["time_minutes"].to_numpy()

        # ── RSI(24) — 2-minute fast exhaustion trigger ───────────────────────
        rsi = _compute_rsi(close, 24)

        # ── EMA(360) — 30-minute intraday trend anchor ───────────────────────
        ema360 = _compute_ema(close, 360)

        # ── VIX filter (panic regime exclusion) ──────────────────────────────
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

        # ── Session and regime filters ───────────────────────────────────────
        in_session = (time_min >= self.session_start_minutes) & (
            time_min < self.session_end_minutes
        )
        low_vix = vix_close < vix_max
        # EMA warmup guard: require at least 400 bars of history before trusting EMA
        warmed_up = np.zeros(n, dtype=bool)
        warmed_up[self.max_lookback :] = True

        base_filter = in_session & low_vix & warmed_up

        # ── Signals ──────────────────────────────────────────────────────────
        # Buy CE: RSI exhaustion selloff below 30-min EMA → expect reversion upward
        buy_ce = (
            base_filter
            & (rsi < rsi_oversold)
            & (close < ema360)
        )

        # Buy PE: RSI exhaustion rally above 30-min EMA → expect fade downward
        buy_pe = (
            base_filter
            & (rsi > rsi_overbought)
            & (close > ema360)
        )

        return OptionSignals(
            buy_ce=buy_ce,
            buy_pe=buy_pe,
            sell_ce=np.zeros(n, dtype=bool),
            sell_pe=np.zeros(n, dtype=bool),
            stop_points=np.full(n, stop_pts),
            target_points=np.full(n, target_pts),
            strike_offset=np.zeros(n, dtype=np.int32),
            time_stop_bars=18,          # 90 seconds max hold
            max_trades_per_day=self.max_trades_per_day,
        )
