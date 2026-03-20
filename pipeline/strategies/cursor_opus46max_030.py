"""Institutional Flow Momentum v1 — cursor_opus46max_030

Thesis: Large bar volume spikes (>5x SMA(volume,20)) that move price >0.2%
represent institutional block execution. Follow-through buying/selling on
the next bar confirms momentum. Entry on follow-through bar close.
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
    name = "cursor_opus46max_030"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vol_spike_mult", default=5.0, low=3.0, high=8.0),
            TunableParam("bar_return_thresh", default=0.2, low=0.1, high=0.4),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        spike_mult = params.get("vol_spike_mult", 5.0)
        ret_thresh = params.get("bar_return_thresh", 0.2)
        stop_pct = params.get("stop_loss_pct", 0.002)
        tgt_pct = params.get("target_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

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

        # ── Volume spike ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_spike = volume / avg_vol

        # ── Bar return ──
        bar_return = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if opn[i] > 0:
                bar_return[i] = (close[i] - opn[i]) / opn[i] * 100.0

        # ── Detect spike bars and follow-through ──
        # Spike bar: vol > 5x, return > 0.2%, green/red confirmation
        # Entry on NEXT bar if it also closes in same direction
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        for i in range(2, n):
            prev = i - 1
            # Skip first 5 bars and last 10 bars of session
            if time_mins[prev] < 575 or time_mins[prev] > 860:
                continue
            if time_mins[i] < 575 or time_mins[i] > 870:
                continue

            # Bullish spike + follow-through
            if (vol_spike[prev] > spike_mult and
                bar_return[prev] > ret_thresh and
                close[prev] > opn[prev] and
                close[prev] > vwap[prev] and
                close[i] > opn[i]):  # follow-through green bar
                long_entry[i] = True

            # Bearish spike + follow-through
            if (vol_spike[prev] > spike_mult and
                bar_return[prev] < -ret_thresh and
                close[prev] < opn[prev] and
                close[prev] < vwap[prev] and
                close[i] < opn[i]):  # follow-through red bar
                short_entry[i] = True

        # ── Signal exits: reverse vol spike ──
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if vol_spike[i] > spike_mult and bar_return[i] < -ret_thresh:
                signal_exit_long[i] = True
            if vol_spike[i] > spike_mult and bar_return[i] > ret_thresh:
                signal_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=tgt_pct,
            trailing_stop_pct=0.0015,
            trailing_activate_pct=0.0025,
            time_stop_bars=30,
        )
