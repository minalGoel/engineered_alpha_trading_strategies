"""VWAP Gap Fill v1 — cursor_opus46max_009

Thesis: Stocks that gap >0.5% from previous day's VWAP tend to fill
that gap in the first 90 minutes. Enter after 5 bars when gap stops
extending, target is prev day VWAP.
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


def _compute_ema(close, period):
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = close[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_009"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 660     # 11:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("min_gap_pct", default=0.5, low=0.3, high=1.0),
            TunableParam("max_gap_pct", default=3.0, low=1.5, high=4.0),
            TunableParam("vix_max", default=20.0, low=14.0, high=26.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
            TunableParam("trailing_pct", default=0.0015, low=0.0008, high=0.003),
            TunableParam("trailing_activate", default=0.003, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        min_gap = params.get("min_gap_pct", 0.5)
        max_gap = params.get("max_gap_pct", 3.0)
        vix_max = params.get("vix_max", 20.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        trail_pct = params.get("trailing_pct", 0.0015)
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
        opn = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # ── Previous day VWAP ──
        prev_day_vwap = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_id)
        pdv = 0.0
        for d in unique_days:
            mask = day_id == d
            idx = np.where(mask)[0]
            prev_day_vwap[idx] = pdv if pdv > 0 else close[idx[0]]
            pdv = vwap[idx[-1]]

        # ── Gap pct from prev day VWAP ──
        day_open = np.zeros(n, dtype=np.float64)
        bar_from_open = np.zeros(n, dtype=np.int32)
        for d in unique_days:
            mask = day_id == d
            idx = np.where(mask)[0]
            day_open[idx] = opn[idx[0]]
            for j, gi in enumerate(idx):
                bar_from_open[gi] = j

        safe_pdv = np.where(prev_day_vwap > 0, prev_day_vwap, 1.0)
        gap_pct = (day_open - prev_day_vwap) / safe_pdv * 100.0

        # ── Gap fill progress ──
        gap_fill_pct = np.zeros(n, dtype=np.float64)
        gap_denom = prev_day_vwap - day_open
        for i in range(n):
            if abs(gap_denom[i]) > 1e-10:
                gap_fill_pct[i] = (close[i] - day_open[i]) / gap_denom[i] * 100.0

        # ── EMA(5) ──
        ema5 = _compute_ema(close, 5)

        # ── First 5 bars low/high ──
        first5_low = np.full(n, np.inf, dtype=np.float64)
        first5_high = np.full(n, -np.inf, dtype=np.float64)
        for d in unique_days:
            mask = day_id == d
            idx = np.where(mask)[0]
            if len(idx) >= 5:
                fl = np.min(low[idx[:5]])
                fh = np.max(high[idx[:5]])
            else:
                fl = np.min(low[idx])
                fh = np.max(high[idx])
            first5_low[idx] = fl
            first5_high[idx] = fh

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Filters ──
        time_ok = (time_mins >= 560) & (time_mins <= 660)
        vix_ok = vix < vix_max

        # ── Entries ──
        long_entry = (
            (gap_pct < -min_gap) & (gap_pct > -max_gap)
            & (gap_fill_pct < 20)
            & (bar_from_open >= 5)
            & (close > first5_low)
            & (close > opn) & (close > ema5)
            & vix_ok & time_ok
        )
        short_entry = (
            (gap_pct > min_gap) & (gap_pct < max_gap)
            & (gap_fill_pct < 20)
            & (bar_from_open >= 5)
            & (close < first5_high)
            & (close < opn) & (close < ema5)
            & vix_ok & time_ok
        )

        # ── Signal exit: gap extends beyond 2x ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if gap_pct[i] < 0 and close[i] < day_open[i] + 2.0 * (day_open[i] - prev_day_vwap[i]):
                sig_exit_long[i] = True
            if gap_pct[i] > 0 and close[i] > day_open[i] + 2.0 * (day_open[i] - prev_day_vwap[i]):
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=prev_day_vwap.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=60,
        )
