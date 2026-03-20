"""Vol Surface Signal v1 — cursor_opus46max_085

Thesis: Extreme put-call IV skew (approximated via VIX z-score as proxy)
signals institutional over-hedging. Fade extreme fear (buy) when skew
starts unwinding. Fade complacency (sell) when skew starts steepening.
Since real option IV surface is unavailable, we use VIX behavior as proxy.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss > 0:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss > 0:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i + 1] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_085"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 1
    assumptions = ["Uses VIX z-score as proxy for option skew since real IV surface unavailable"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_z_long", default=2.0, low=1.5, high=3.0),
            TunableParam("vix_z_short", default=-1.5, low=-2.5, high=-1.0),
            TunableParam("rsi_long_thresh", default=40.0, low=30.0, high=50.0),
            TunableParam("rsi_short_thresh", default=65.0, low=55.0, high=75.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("breakeven_pct", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_z_long = params.get("vix_z_long", 2.0)
        vix_z_short = params.get("vix_z_short", -1.5)
        rsi_long = params.get("rsi_long_thresh", 40.0)
        rsi_short = params.get("rsi_short_thresh", 65.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        target_pct = params.get("target_pct", 0.005)
        be_pct = params.get("breakeven_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── VIX z-score over 120 bars (proxy for skew z) ──
        vix_z = np.zeros(n, dtype=np.float64)
        zw = 120
        for i in range(zw, n):
            window = vix[i - zw:i + 1]
            mu = np.mean(window)
            sd = np.std(window)
            if sd > 0:
                vix_z[i] = (vix[i] - mu) / sd

        # ── VIX rate of change over 30 bars (proxy for skew_rate) ──
        vix_rate = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            vix_rate[i] = vix[i] - vix[i - 30]

        # ── VIX declining 5 bars (skew unwinding) ──
        vix_declining_5 = np.zeros(n, dtype=np.bool_)
        for i in range(5, n):
            declining = True
            for j in range(1, 6):
                if vix[i - j + 1] >= vix[i - j]:
                    declining = False
                    break
            vix_declining_5[i] = declining

        # ── VIX rising 5 bars ──
        vix_rising_5 = np.zeros(n, dtype=np.bool_)
        for i in range(5, n):
            rising = True
            for j in range(1, 6):
                if vix[i - j + 1] <= vix[i - j]:
                    rising = False
                    break
            vix_rising_5[i] = rising

        # ── RSI(30) on close (longer RSI as in strategy) ──
        rsi = _compute_rsi(close, 30)

        vix_in_range = (vix >= 14.0) & (vix <= 25.0)

        long_entry = (
            (vix_z > vix_z_long) &
            (rsi < rsi_long) &
            vix_declining_5 &
            vix_in_range
        )

        short_entry = (
            (vix_z < vix_z_short) &
            (rsi > rsi_short) &
            vix_rising_5 &
            vix_in_range
        )

        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            breakeven_pct=be_pct,
            time_stop_bars=240,
        )
