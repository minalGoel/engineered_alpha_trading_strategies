"""Pullback to VWAP Trend Ride — gemini_8_of_20

Thesis: In a strong trend (ADX > 30), pullbacks to VWAP offer low-risk entries.
Ride the trend with trailing stop and exit if ADX weakens.
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


class Strategy(BaseStrategy):
    name = "pullback_to_vwap_trend_ride"
    is_long_only = False
    session_start = 630   # 10:30
    session_end = 900     # 15:00
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_min", default=30.0, low=20.0, high=40.0),
            TunableParam("adx_exit", default=25.0, low=15.0, high=35.0),
            TunableParam("vwap_proximity", default=0.001, low=0.0003, high=0.003),
            TunableParam("target_atr_mult", default=3.0, low=1.5, high=5.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.008),
            TunableParam("trailing_atr_mult", default=1.5, low=0.5, high=3.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_min = params.get("adx_min", 30.0)
        adx_exit = params.get("adx_exit", 25.0)
        vwap_prox = params.get("vwap_proximity", 0.001)
        tgt_atr = params.get("target_atr_mult", 3.0)
        stop_pct = params.get("stop_loss_pct", 0.004)
        trail_atr = params.get("trailing_atr_mult", 1.5)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=close[0] if n > 0 else 0.0)

        # ADX(14) with +DI / -DI
        atr14 = _compute_atr(high, low, close, 14)
        adx_period = 14

        plus_dm = np.zeros(n, dtype=np.float64)
        minus_dm = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            up = high[i] - high[i - 1]
            dn = low[i - 1] - low[i]
            plus_dm[i] = up if (up > dn and up > 0) else 0.0
            minus_dm[i] = dn if (dn > up and dn > 0) else 0.0

        smooth_plus = np.zeros(n, dtype=np.float64)
        smooth_minus = np.zeros(n, dtype=np.float64)
        if n >= adx_period + 1:
            smooth_plus[adx_period] = np.mean(plus_dm[1:adx_period + 1])
            smooth_minus[adx_period] = np.mean(minus_dm[1:adx_period + 1])
            for i in range(adx_period + 1, n):
                smooth_plus[i] = (smooth_plus[i - 1] * (adx_period - 1) + plus_dm[i]) / adx_period
                smooth_minus[i] = (smooth_minus[i - 1] * (adx_period - 1) + minus_dm[i]) / adx_period

        safe_atr = np.where(atr14 > 1e-10, atr14, 1e-10)
        plus_di = 100.0 * smooth_plus / safe_atr
        minus_di = 100.0 * smooth_minus / safe_atr
        dx = np.where((plus_di + minus_di) > 1e-10,
                       100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0.0)

        adx = np.zeros(n, dtype=np.float64)
        adx_start = 2 * adx_period
        if n > adx_start:
            adx[adx_start] = np.mean(dx[adx_period:adx_start + 1])
            for i in range(adx_start + 1, n):
                adx[i] = (adx[i - 1] * (adx_period - 1) + dx[i]) / adx_period

        # Close near VWAP
        safe_close = np.where(close > 1e-10, close, 1e-10)
        near_vwap = np.abs(close - vwap) / safe_close < vwap_prox

        # Bullish / bearish bar
        bullish_bar = close > opn
        bearish_bar = close < opn

        # Entries
        long_entry = ((adx > adx_min) & (plus_di > minus_di) &
                       near_vwap & bullish_bar)
        short_entry = ((adx > adx_min) & (minus_di > plus_di) &
                        near_vwap & bearish_bar)

        # Signal exit: ADX drops below threshold
        signal_exit_long = adx < adx_exit
        signal_exit_short = adx < adx_exit

        # Trailing stop expressed as pct from ATR
        # trail_pct ~ trail_atr * mean(atr14) / mean(close)
        mean_atr = np.mean(atr14[atr14 > 0]) if np.any(atr14 > 0) else 1.0
        mean_close = np.mean(close[close > 0]) if np.any(close > 0) else 1.0
        trail_pct = float(trail_atr * mean_atr / mean_close)
        trail_pct = max(trail_pct, 0.001)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_atr_mult=tgt_atr,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=stop_pct,
            time_stop_bars=180,
        )
