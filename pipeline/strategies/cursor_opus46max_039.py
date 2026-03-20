"""Volatility Contraction Breakout v1 — cursor_opus46max_039

Thesis: When ATR(14) contracts to below 20th percentile of session
distribution, it signals a volatility squeeze. Breakout from squeeze
with volume > 2x on expansion bar signals directional move. Stop at
opposite side of squeeze range.
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
    name = "cursor_opus46max_039"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("atr_pctile_thresh", default=20.0, low=10.0, high=30.0),
            TunableParam("squeeze_min_bars", default=10.0, low=5.0, high=20.0),
            TunableParam("vol_mult", default=2.0, low=1.2, high=3.0),
            TunableParam("expansion_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("target_atr_mult", default=2.0, low=1.0, high=3.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        atr_pctile_th = params.get("atr_pctile_thresh", 20.0)
        squeeze_min = int(params.get("squeeze_min_bars", 10.0))
        vol_mult = params.get("vol_mult", 2.0)
        expansion_mult = params.get("expansion_mult", 1.5)
        vix_max = params.get("vix_max", 22.0)
        tgt_atr = params.get("target_atr_mult", 2.0)

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
        day_id = df["day_id"].to_numpy()

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── Session ATR percentile ──
        atr_pctile = np.full(n, 50.0, dtype=np.float64)
        prev_day = -1
        day_start = 0
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                day_start = i
            bars_in_day = i - day_start + 1
            if bars_in_day > 1 and atr[i] > 0:
                session_atrs = atr[day_start:i+1]
                atr_pctile[i] = (np.sum(session_atrs < atr[i]) / len(session_atrs)) * 100.0

        # ── Squeeze bar count ──
        squeeze_bars = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if atr_pctile[i] < atr_pctile_th:
                squeeze_bars[i] = squeeze_bars[i-1] + 1
            else:
                squeeze_bars[i] = 0

        # ── Recently squeezed (within last 5 bars) ──
        recently_squeezed = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            for j in range(max(0, i-4), i+1):
                if squeeze_bars[j] >= squeeze_min:
                    recently_squeezed[i] = True
                    break

        # ── Expansion bar: range > expansion_mult * ATR ──
        bar_range = high - low
        expansion_bar = bar_range > expansion_mult * atr

        # ── Session high/low tracking ──
        session_high = np.zeros(n, dtype=np.float64)
        session_low = np.zeros(n, dtype=np.float64)
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                prev_day = day_id[i]
                session_high[i] = high[i]
                session_low[i] = low[i]
            else:
                session_high[i] = max(session_high[i-1], high[i])
                session_low[i] = min(session_low[i-1], low[i])

        new_session_high = np.zeros(n, dtype=np.bool_)
        new_session_low = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if day_id[i] == day_id[i-1]:
                new_session_high[i] = high[i] > session_high[i-1]
                new_session_low[i] = low[i] < session_low[i-1]

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        # ── Close position in bar ──
        bar_range_safe = np.where(bar_range > 0, bar_range, 1e-10)
        close_pos = (close - low) / bar_range_safe

        # ── Filters ──
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 600) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            recently_squeezed & expansion_bar &
            new_session_high & (close > opn) & (close_pos > 0.60) &
            vol_ok & (close > vwap) & vix_ok & time_ok
        )
        short_entry = (
            recently_squeezed & expansion_bar &
            new_session_low & (close < opn) & (close_pos < 0.40) &
            vol_ok & (close < vwap) & vix_ok & time_ok
        )

        # ── Signal exit: ATR drops back to squeeze ──
        signal_exit_long = (atr_pctile < atr_pctile_th) & (squeeze_bars > 0)
        signal_exit_short = (atr_pctile < atr_pctile_th) & (squeeze_bars > 0)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_atr_mult=tgt_atr,
            stop_loss_pct=0.005,
            time_stop_bars=45,
        )
