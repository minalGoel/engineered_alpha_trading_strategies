"""Stochastic Reversion v1 — cursor_opus46max_019

Thesis: Stochastic (14,3,3) crossover in extreme zones (<20, >80) with
VWAP proximity filter. Enter when %K crosses %D in oversold/overbought
zone with confirming bar direction.
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


def _sma(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    for i in range(period - 1, n):
        out[i] = np.mean(arr[i - period + 1:i + 1])
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_019"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("os_level", default=20.0, low=10.0, high=30.0),
            TunableParam("ob_level", default=80.0, low=70.0, high=90.0),
            TunableParam("vix_max", default=25.0, low=16.0, high=30.0),
            TunableParam("target_pct", default=0.0025, low=0.001, high=0.005),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        os_lvl = params.get("os_level", 20.0)
        ob_lvl = params.get("ob_level", 80.0)
        vix_max = params.get("vix_max", 25.0)
        tgt_pct = params.get("target_pct", 0.0025)
        stop_pct = params.get("stop_loss_pct", 0.002)

        n = len(df)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        close = df["close"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Stochastic (14,3,3) ──
        raw_k = np.zeros(n, dtype=np.float64)
        for i in range(13, n):
            hh = np.max(high[i - 13:i + 1])
            ll = np.min(low[i - 13:i + 1])
            denom = hh - ll
            if denom > 1e-10:
                raw_k[i] = (close[i] - ll) / denom * 100.0
            else:
                raw_k[i] = 50.0

        stoch_k = _sma(raw_k, 3)
        stoch_d = _sma(stoch_k, 3)

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.8 * avg_vol

        # ── Bullish / bearish bar ──
        bullish = close > opn
        bearish = close < opn

        # ── Cross detection ──
        cross_up = np.zeros(n, dtype=np.bool_)
        cross_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if stoch_k[i] > stoch_d[i] and stoch_k[i - 1] <= stoch_d[i - 1]:
                cross_up[i] = True
            if stoch_k[i] < stoch_d[i] and stoch_k[i - 1] >= stoch_d[i - 1]:
                cross_down[i] = True

        # ── Bars in extreme zone ──
        bars_in_extreme = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if stoch_k[i] < os_lvl or stoch_k[i] > ob_lvl:
                bars_in_extreme[i] = bars_in_extreme[i - 1] + 1
            else:
                bars_in_extreme[i] = 0

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 885)
        vix_ok = vix < vix_max

        # ── Entries ──
        long_entry = (
            (stoch_k < os_lvl) & (stoch_d < os_lvl)
            & cross_up & bullish
            & (close > vwap * 0.995)
            & (bars_in_extreme <= 10)
            & vol_ok & vix_ok & time_ok
        )
        short_entry = (
            (stoch_k > ob_lvl) & (stoch_d > ob_lvl)
            & cross_down & bearish
            & (close < vwap * 1.005)
            & (bars_in_extreme <= 10)
            & vol_ok & vix_ok & time_ok
        )

        # ── Signal exit: stoch_k re-enters extreme or crosses 50 ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if stoch_k[i] >= 50 and stoch_k[i - 1] < 50:
                sig_exit_long[i] = True
            if stoch_k[i] < os_lvl and stoch_k[i - 1] >= os_lvl:
                sig_exit_long[i] = True
            if stoch_k[i] <= 50 and stoch_k[i - 1] > 50:
                sig_exit_short[i] = True
            if stoch_k[i] > ob_lvl and stoch_k[i - 1] <= ob_lvl:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=25,
        )
