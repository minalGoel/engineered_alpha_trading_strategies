"""ADX Trend Strength v1 — cursor_opus46max_023

Thesis: ADX(14) crossing above 25 from below signals a nascent trend.
Enter when +DI > -DI (longs) or -DI > +DI (shorts) with DI spread > 10,
VWAP alignment, and ADX still rising as confirmation.
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


def _compute_adx_full(high, low, close, period):
    """Returns (adx, plus_di, minus_di) arrays."""
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return adx, plus_di, minus_di
    atr = _compute_atr(high, low, close, period)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    smooth_plus = np.zeros(n, dtype=np.float64)
    smooth_minus = np.zeros(n, dtype=np.float64)
    smooth_plus[period] = np.sum(plus_dm[1:period + 1])
    smooth_minus[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        smooth_plus[i] = smooth_plus[i - 1] - smooth_plus[i - 1] / period + plus_dm[i]
        smooth_minus[i] = smooth_minus[i - 1] - smooth_minus[i - 1] / period + minus_dm[i]
    for i in range(period, n):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * smooth_plus[i] / atr[i] / period
            minus_di[i] = 100.0 * smooth_minus[i] / atr[i] / period
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        denom = plus_di[i] + minus_di[i]
        if denom > 1e-10:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / denom
    if n >= 2 * period:
        adx[2 * period - 1] = np.mean(dx[period:2 * period])
        for i in range(2 * period, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx, plus_di, minus_di


class Strategy(BaseStrategy):
    name = "cursor_opus46max_023"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 885     # 14:45
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_cross_level", default=25.0, low=20.0, high=30.0),
            TunableParam("di_spread_min", default=10.0, low=5.0, high=15.0),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.004),
            TunableParam("trailing_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("trailing_activate", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_lvl = params.get("adx_cross_level", 25.0)
        di_spread = params.get("di_spread_min", 10.0)
        tgt_pct = params.get("target_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        trail_pct = params.get("trailing_pct", 0.002)
        trail_act = params.get("trailing_activate", 0.003)

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
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── ADX with DI lines ──
        adx, plus_di, minus_di = _compute_adx_full(high, low, close, 14)

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── ADX cross above threshold ──
        adx_cross_up = np.zeros(n, dtype=np.bool_)
        adx_rising = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if adx[i] > adx_lvl and adx[i - 1] <= adx_lvl:
                adx_cross_up[i] = True
            if adx[i] > adx[i - 1]:
                adx_rising[i] = True

        # ── False breakout filter: skip if ADX crossed 25 and fell back 2+ times
        #    in the last 60 bars ──
        adx_fell_back = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if adx[i] < adx_lvl and adx[i - 1] >= adx_lvl:
                adx_fell_back[i] = 1
        false_breakout_count = np.zeros(n, dtype=np.int32)
        for i in range(n):
            start = max(0, i - 59)
            false_breakout_count[i] = np.sum(adx_fell_back[start:i + 1])
        fb_ok = false_breakout_count < 2

        # ── Filters ──
        time_ok = (time_mins >= 585) & (time_mins <= 885)

        # ── Entries ──
        long_entry = (
            adx_cross_up
            & adx_rising
            & (plus_di > minus_di)
            & ((plus_di - minus_di) > di_spread)
            & (close > vwap)
            & vol_ok & fb_ok & time_ok
        )
        short_entry = (
            adx_cross_up
            & adx_rising
            & (minus_di > plus_di)
            & ((minus_di - plus_di) > di_spread)
            & (close < vwap)
            & vol_ok & fb_ok & time_ok
        )

        # ── Signal exit: ADX declining AND DI lines converging ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            adx_declining = adx[i] < adx[i - 1]
            di_converging_l = (plus_di[i] - minus_di[i]) < (plus_di[i - 1] - minus_di[i - 1])
            di_converging_s = (minus_di[i] - plus_di[i]) < (minus_di[i - 1] - plus_di[i - 1])
            if adx_declining and di_converging_l:
                sig_exit_long[i] = True
            if adx[i] < 20:
                sig_exit_long[i] = True
            if adx_declining and di_converging_s:
                sig_exit_short[i] = True
            if adx[i] < 20:
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
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=60,
        )
