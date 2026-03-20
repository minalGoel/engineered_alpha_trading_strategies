"""Range Expansion Breakout v1 — cursor_opus46max_041

Thesis: Tight intraday base (range < 0.3% over 30+ bars) followed by
expansion bar (range > 3x average) signals equilibrium break. Direction
of expansion predicts continuation due to exhausted limit orders and
stop triggering.
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
    name = "cursor_opus46max_041"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("base_range_max", default=0.3, low=0.1, high=0.5),
            TunableParam("expansion_ratio", default=3.0, low=2.0, high=5.0),
            TunableParam("vol_mult", default=2.0, low=1.2, high=3.0),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        base_max = params.get("base_range_max", 0.3)
        exp_ratio = params.get("expansion_ratio", 3.0)
        vol_mult = params.get("vol_mult", 2.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
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

        # ── Rolling 30-bar base range ──
        base_range = np.full(n, 999.0, dtype=np.float64)
        base_high = np.zeros(n, dtype=np.float64)
        base_low = np.zeros(n, dtype=np.float64)
        for i in range(29, n):
            h30 = np.max(high[i-29:i+1])
            l30 = np.min(low[i-29:i+1])
            base_high[i] = h30
            base_low[i] = l30
            if l30 > 0:
                base_range[i] = (h30 - l30) / l30 * 100.0

        # ── Average bar range over 30 bars ──
        bar_range = high - low
        avg_bar_range = np.zeros(n, dtype=np.float64)
        for i in range(29, n):
            avg_bar_range[i] = np.mean(bar_range[i-29:i+1])
        avg_bar_range = np.where(avg_bar_range > 0, avg_bar_range, 1e-10)

        # ── Expansion ratio ──
        expansion = bar_range / avg_bar_range

        # ── Close position in bar ──
        bar_range_safe = np.where(bar_range > 0, bar_range, 1e-10)
        close_pos = (close - low) / bar_range_safe

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Breakout above/below base ──
        above_base = np.zeros(n, dtype=np.bool_)
        below_base = np.zeros(n, dtype=np.bool_)
        for i in range(30, n):
            prev_base_high = np.max(high[i-30:i])
            prev_base_low = np.min(low[i-30:i])
            above_base[i] = close[i] > prev_base_high
            below_base[i] = close[i] < prev_base_low

        # ── Filters ──
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 600) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            (base_range < base_max) & (expansion > exp_ratio) &
            above_base & vol_ok & (close > vwap) &
            (close_pos > 0.75) & vix_ok & time_ok
        )
        short_entry = (
            (base_range < base_max) & (expansion > exp_ratio) &
            below_base & vol_ok & (close < vwap) &
            (close_pos < 0.25) & vix_ok & time_ok
        )

        # ── Signal exit: price closes back within base for 2 bars ──
        inside_base = np.zeros(n, dtype=np.int32)
        for i in range(30, n):
            if close[i] >= base_low[i] and close[i] <= base_high[i]:
                inside_base[i] = (inside_base[i-1] + 1) if i > 0 else 1
            else:
                inside_base[i] = 0

        signal_exit_long = inside_base >= 2
        signal_exit_short = inside_base >= 2

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=stop_pct * 2.0,
            time_stop_bars=60,
        )
