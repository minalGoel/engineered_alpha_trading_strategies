"""Relative Momentum v1 — cursor_opus46max_027

Thesis: Stocks showing intraday return > 2x NIFTY 50 return exhibit
persistent relative momentum. Institutional allocation that causes
outperformance unfolds over hours. Entry after 10:30 when relative
strength is confirmed and still building.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "cursor_opus46max_027"
    is_long_only = False
    session_start = 630   # 10:30
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rel_return_thresh", default=0.5, low=0.2, high=1.0),
            TunableParam("rel_return_30_thresh", default=0.3, low=0.1, high=0.6),
            TunableParam("nifty_floor", default=-0.5, low=-1.0, high=0.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rel_thresh = params.get("rel_return_thresh", 0.5)
        rel30_thresh = params.get("rel_return_30_thresh", 0.3)
        nifty_floor = params.get("nifty_floor", -0.5)
        stop_pct = params.get("stop_loss_pct", 0.003)
        tgt_pct = params.get("target_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Per-day open tracking for stock and index ──
        stock_return = np.zeros(n, dtype=np.float64)
        nifty_return = np.zeros(n, dtype=np.float64)
        day_open_stock = np.zeros(n, dtype=np.float64)
        day_open_index = np.zeros(n, dtype=np.float64)
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                day_open_stock[i] = opn[i]
                day_open_index[i] = index_close[i]
                prev_day = day_id[i]
            else:
                day_open_stock[i] = day_open_stock[i-1]
                day_open_index[i] = day_open_index[i-1]
            if day_open_stock[i] > 0:
                stock_return[i] = (close[i] - day_open_stock[i]) / day_open_stock[i] * 100.0
            if day_open_index[i] > 0:
                nifty_return[i] = (index_close[i] - day_open_index[i]) / day_open_index[i] * 100.0

        relative_return = stock_return - nifty_return

        # ── 30-bar relative return ──
        rel_return_30 = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            sr30 = (close[i] - close[i-30]) / close[i-30] * 100.0 if close[i-30] > 0 else 0.0
            nr30 = (index_close[i] - index_close[i-30]) / index_close[i-30] * 100.0 if index_close[i-30] > 0 else 0.0
            rel_return_30[i] = sr30 - nr30

        # ── RS still building: relative_return increasing over 10 bars ──
        rs_building = np.zeros(n, dtype=np.bool_)
        rs_weakening = np.zeros(n, dtype=np.bool_)
        for i in range(10, n):
            rs_building[i] = relative_return[i] > relative_return[i-10]
            rs_weakening[i] = relative_return[i] < relative_return[i-10]

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.8 * avg_vol

        # ── Time filter ──
        time_ok = (time_mins >= 630) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            (relative_return > rel_thresh) &
            (rel_return_30 > rel30_thresh) &
            (nifty_return > nifty_floor) &
            (close > vwap) &
            rs_building & vol_ok & time_ok
        )
        short_entry = (
            (relative_return < -rel_thresh) &
            (rel_return_30 < -rel30_thresh) &
            (nifty_return < -nifty_floor) &
            (close < vwap) &
            rs_weakening & vol_ok & time_ok
        )

        # ── Signal exits: relative_return_30 flips sign ──
        signal_exit_long = rel_return_30 < 0
        signal_exit_short = rel_return_30 > 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=tgt_pct,
            trailing_stop_pct=0.002,
            trailing_activate_pct=0.003,
            time_stop_bars=90,
        )
