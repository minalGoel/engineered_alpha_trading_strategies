"""ORB Multi Stock Portfolio v1 — cursor_opus46max_097

Thesis: Portfolio ORB approach — signal scoring (volume ratio + gap alignment +
VWAP position) gives higher conviction entries. Adapted for single-stock backtest
by requiring high signal score (2+ of 3 confirmations) plus NIFTY directional
alignment as a proxy for "multiple stocks breaking ORB in same direction."
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
    name = "cursor_opus46max_097"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("min_score", default=2.0, low=1.0, high=3.0),
            TunableParam("vol_mult", default=1.2, low=1.0, high=2.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("target_atr_mult", default=2.0, low=1.0, high=3.0),
            TunableParam("vix_low", default=12.0, low=8.0, high=16.0),
            TunableParam("vix_high", default=22.0, low=18.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        min_score = params.get("min_score", 2.0)
        vol_mult = params.get("vol_mult", 1.2)
        stop_pct = params.get("stop_loss_pct", 0.004)
        target_atr_mult = params.get("target_atr_mult", 2.0)
        vix_lo = params.get("vix_low", 12.0)
        vix_hi = params.get("vix_high", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
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

        # -- Two-bar confirmation --
        above_orb = (close > orb_high) & orb_computed
        below_orb = (close < orb_low) & orb_computed
        two_above = np.zeros(n, dtype=np.bool_)
        two_below = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if above_orb[i] and above_orb[i - 1] and day_id[i] == day_id[i - 1]:
                two_above[i] = True
            if below_orb[i] and below_orb[i - 1] and day_id[i] == day_id[i - 1]:
                two_below[i] = True

        # -- Volume filter --
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # -- Gap computation --
        gap_pct = np.zeros(n, dtype=np.float64)
        last_close = close[0]
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                if last_close > 0:
                    gap_pct[i] = (opn[i] - last_close) / last_close * 100.0
            else:
                gap_pct[i] = gap_pct[i - 1]
            if i + 1 < n and day_id[i] != day_id[i + 1]:
                last_close = close[i]
            elif i == n - 1:
                last_close = close[i]

        # -- Index direction (simple: index_close > index at day start) --
        idx_day_start = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                idx_day_start[i] = index_close[i]
            else:
                idx_day_start[i] = idx_day_start[i - 1]

        # -- Signal score: 0-3 (volume ratio > threshold, gap alignment, VWAP position) --
        score_long = np.zeros(n, dtype=np.float64)
        score_short = np.zeros(n, dtype=np.float64)
        for i in range(n):
            # Volume component
            if vol_ratio[i] > vol_mult:
                score_long[i] += 1.0
                score_short[i] += 1.0
            # Gap alignment
            if gap_pct[i] > 0:
                score_long[i] += 1.0
            if gap_pct[i] < 0:
                score_short[i] += 1.0
            # VWAP position
            if close[i] > vwap[i]:
                score_long[i] += 1.0
            if close[i] < vwap[i]:
                score_short[i] += 1.0

        # -- NIFTY alignment (proxy for multi-stock confirmation) --
        nifty_up = index_close > idx_day_start
        nifty_down = index_close < idx_day_start

        vix_ok = (vix >= vix_lo) & (vix <= vix_hi)
        time_ok = (time_min >= 570) & (time_min <= 630)

        long_entry = two_above & (score_long >= min_score) & nifty_up & vix_ok & time_ok
        short_entry = two_below & (score_short >= min_score) & nifty_down & vix_ok & time_ok

        # -- Signal exit: NIFTY reverses direction --
        signal_exit_long = nifty_down.copy()
        signal_exit_short = nifty_up.copy()

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_atr_mult=target_atr_mult,
            trailing_stop_pct=0.003,
            trailing_activate_pct=0.003,
            time_stop_bars=300,
        )
