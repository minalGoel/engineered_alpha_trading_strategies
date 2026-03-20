"""ORB ATR-Scaled v1 — cursor_opus46max_096

Thesis: Standard ORB uses fixed multiples of the opening range for targets/stops.
By scaling targets and stops using ATR(14) instead of ORB range, we adapt trade
management to the stock's real-time volatility. Target = 2.0*ATR, stop = 1.0*ATR.
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


def _compute_adx(high, low, close, period):
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    if n < period * 2:
        return adx
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    # Wilder smoothing
    smooth_tr = np.zeros(n, dtype=np.float64)
    smooth_pdm = np.zeros(n, dtype=np.float64)
    smooth_mdm = np.zeros(n, dtype=np.float64)
    smooth_tr[period] = np.sum(tr[1:period + 1])
    smooth_pdm[period] = np.sum(plus_dm[1:period + 1])
    smooth_mdm[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        smooth_tr[i] = smooth_tr[i - 1] - smooth_tr[i - 1] / period + tr[i]
        smooth_pdm[i] = smooth_pdm[i - 1] - smooth_pdm[i - 1] / period + plus_dm[i]
        smooth_mdm[i] = smooth_mdm[i - 1] - smooth_mdm[i - 1] / period + minus_dm[i]

    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        if smooth_tr[i] > 0:
            pdi = 100.0 * smooth_pdm[i] / smooth_tr[i]
            mdi = 100.0 * smooth_mdm[i] / smooth_tr[i]
            if pdi + mdi > 0:
                dx[i] = 100.0 * abs(pdi - mdi) / (pdi + mdi)

    # ADX = Wilder smoothed DX
    start = 2 * period
    if start < n:
        adx[start] = np.mean(dx[period:start + 1])
        for i in range(start + 1, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


class Strategy(BaseStrategy):
    name = "cursor_opus46max_096"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("adx_min", default=18.0, low=14.0, high=25.0),
            TunableParam("vol_mult", default=1.2, low=1.0, high=2.0),
            TunableParam("bar_range_pct", default=0.30, low=0.20, high=0.40),
            TunableParam("stop_atr_mult", default=1.0, low=0.5, high=1.5),
            TunableParam("target_atr_mult", default=2.0, low=1.5, high=3.0),
            TunableParam("vix_low", default=12.0, low=8.0, high=16.0),
            TunableParam("vix_high", default=25.0, low=20.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        adx_min = params.get("adx_min", 18.0)
        vol_mult = params.get("vol_mult", 1.2)
        bar_range_pct = params.get("bar_range_pct", 0.30)
        stop_atr_mult = params.get("stop_atr_mult", 1.0)
        target_atr_mult = params.get("target_atr_mult", 2.0)
        vix_lo = params.get("vix_low", 12.0)
        vix_hi = params.get("vix_high", 25.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_min = df["time_minutes"].to_numpy()
        day_id = df["day_id"].to_numpy()

        # -- VWAP --
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # -- ORB 15-min --
        orb_high = np.zeros(n, dtype=np.float64)
        orb_low = np.zeros(n, dtype=np.float64)
        orb_computed = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cur_h = high[i]
                cur_l = low[i]
            if time_min[i] < 570:
                cur_h = max(cur_h, high[i])
                cur_l = min(cur_l, low[i])
            orb_high[i] = cur_h
            orb_low[i] = cur_l
            if time_min[i] >= 570:
                orb_computed[i] = True

        # -- ADX --
        adx = _compute_adx(high, low, close, 14)

        # -- Bar range position (close in upper/lower pct of bar) --
        bar_range = high - low
        bar_range_safe = np.where(bar_range > 0, bar_range, 1.0)
        bar_position = (close - low) / bar_range_safe  # 0=low, 1=high

        # -- Volume filter --
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        vix_ok = (vix >= vix_lo) & (vix <= vix_hi)
        time_ok = (time_min >= 570) & (time_min <= 630)

        # -- ATR min filter: skip if ATR < 0.1% of price --
        atr = _compute_atr(high, low, close, 14)
        close_safe = np.where(close > 0, close, 1.0)
        atr_pct = atr / close_safe
        atr_ok = atr_pct > 0.001

        long_entry = (
            orb_computed & (close > orb_high) & (adx > adx_min) &
            (bar_position > (1.0 - bar_range_pct)) &
            (close > vwap) & vol_ok & vix_ok & time_ok & atr_ok
        )
        short_entry = (
            orb_computed & (close < orb_low) & (adx > adx_min) &
            (bar_position < bar_range_pct) &
            (close < vwap) & vol_ok & vix_ok & time_ok & atr_ok
        )

        # -- Signal exit: ADX drops below 15 (trend fading) --
        signal_exit_long = adx < 15.0
        signal_exit_short = adx < 15.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr_mult,
            target_atr_mult=target_atr_mult,
            trailing_stop_pct=0.003,
            trailing_activate_pct=0.003,
            time_stop_bars=180,
        )
