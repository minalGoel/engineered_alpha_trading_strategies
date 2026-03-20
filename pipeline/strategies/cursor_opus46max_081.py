"""Straddle Proxy v1 — cursor_opus46max_081

Thesis: Synthetic straddle in equity by entering at a pivot level
(prev day close) during ATR compression. Go long on upper trigger break,
short on lower trigger break. Uses symmetric stop-and-reverse logic
approximated via dual entry signals.
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
    name = "cursor_opus46max_081"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 910     # 15:10
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("atr_compression_mult", default=0.7, low=0.5, high=0.9),
            TunableParam("bb_width_pctile", default=20.0, low=10.0, high=30.0),
            TunableParam("vol_mult", default=1.2, low=1.0, high=1.5),
            TunableParam("target_atr_mult", default=2.5, low=1.5, high=4.0),
            TunableParam("stop_atr_mult", default=1.0, low=0.5, high=1.5),
            TunableParam("vix_low", default=12.0, low=8.0, high=15.0),
            TunableParam("vix_high", default=22.0, low=18.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        comp_mult = params.get("atr_compression_mult", 0.7)
        bb_pctile = params.get("bb_width_pctile", 20.0)
        vol_mult = params.get("vol_mult", 1.2)
        target_atr = params.get("target_atr_mult", 2.5)
        stop_atr = params.get("stop_atr_mult", 1.0)
        vix_lo = params.get("vix_low", 12.0)
        vix_hi = params.get("vix_high", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_id = df["day_id"].to_numpy()

        # ── Pivot = first bar close per day (proxy for prev day close) ──
        pivot = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                pivot[i] = close[i]
            else:
                pivot[i] = pivot[i - 1]

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── ATR SMA(60) for compression detection ──
        atr_sma60 = np.zeros(n, dtype=np.float64)
        for i in range(60, n):
            atr_sma60[i] = np.mean(atr[i - 60:i])
        atr_compressed = (atr < comp_mult * atr_sma60) & (atr_sma60 > 0)

        # ── Recent compression within last 10 bars ──
        recent_compress = np.zeros(n, dtype=np.bool_)
        for i in range(10, n):
            if np.any(atr_compressed[max(0, i - 10):i]):
                recent_compress[i] = True

        # ── BB width percentile ──
        sma20 = np.zeros(n, dtype=np.float64)
        std20 = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            sma20[i] = np.mean(close[i - 20:i])
            std20[i] = np.std(close[i - 20:i])
        bb_width = np.where(sma20 > 0, (4.0 * std20) / sma20, 1.0)
        bb_pctile_arr = np.full(n, 50.0, dtype=np.float64)
        for i in range(120, n):
            window = bb_width[i - 120:i + 1]
            bb_pctile_arr[i] = (np.sum(window < bb_width[i]) / len(window)) * 100.0

        # ── Triggers ──
        stop_dist = 0.5 * atr
        upper_trigger = pivot + stop_dist
        lower_trigger = pivot - stop_dist

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_mult * avg_vol

        vix_ok = (vix >= vix_lo) & (vix <= vix_hi)

        # ── Price was near pivot recently (within 5 bps in last 15 bars) ──
        near_pivot_recent = np.zeros(n, dtype=np.bool_)
        for i in range(15, n):
            for j in range(max(0, i - 15), i):
                if pivot[j] > 0 and abs(close[j] - pivot[j]) / pivot[j] < 0.0005:
                    near_pivot_recent[i] = True
                    break

        squeeze = (bb_pctile_arr < bb_pctile) & recent_compress & near_pivot_recent

        long_entry = squeeze & (close > upper_trigger) & vol_ok & vix_ok
        short_entry = squeeze & (close < lower_trigger) & vol_ok & vix_ok

        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_atr_mult=target_atr,
            trailing_stop_pct=0.003,
            trailing_activate_pct=0.003,
            time_stop_bars=60,
        )
