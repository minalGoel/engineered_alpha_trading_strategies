# AUDIT FIX: Added session window filter to prevent signals outside session_start/session_end.
"""VWAP Pivot Rejection — Grok_4_of_10

Thesis: Rejection at VWAP + previous-day pivot confluence zone signals reversal.
Uses doji/pinbar detection as confirmation.

Assumptions:
- "pivot_s4" and "pivot_r1" interpreted as classic pivot points from previous day
- "rejection" = pinbar pattern (long wick relative to body)
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
    name = "vwap_pivot_rejection_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 10
    assumptions = ["interpreted pivot levels as classic floor pivots from prev day OHLC",
                   "rejection = pinbar (wick > 2x body)"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", default=20.0, low=12.0, high=28.0),
            TunableParam("zone_pct", default=0.002, low=0.001, high=0.005),
            TunableParam("stop_atr_mult", default=0.6, low=0.3, high=1.5),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_max = params.get("vix_max", 20.0)
        zone_pct = params.get("zone_pct", 0.002)
        stop_atr = params.get("stop_atr_mult", 0.6)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_ids = df["day_id"].to_numpy()

        atr20 = _compute_atr(high, low, close, 20)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Previous-day pivot (classic floor pivot: PP = (H+L+C)/3) ──
        pivot = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        prev_pp = np.nan
        for d in unique_days:
            day_idx = np.where(day_ids == d)[0]
            if not np.isnan(prev_pp):
                pivot[day_idx] = prev_pp
            prev_pp = (np.max(high[day_idx]) + np.min(low[day_idx]) + close[day_idx[-1]]) / 3.0

        pivot = np.nan_to_num(pivot, nan=0.0)

        # ── Zone confluence: VWAP near pivot ──
        zone_ok = np.abs(vwap - pivot) < zone_pct * close

        # ── Pinbar detection (wick > 2x body) ──
        body = np.abs(close - open_)
        upper_wick = high - np.maximum(close, open_)
        lower_wick = np.minimum(close, open_) - low
        bullish_pin = (lower_wick > 2.0 * np.clip(body, 0.01, None))
        bearish_pin = (upper_wick > 2.0 * np.clip(body, 0.01, None))

        # ── Filters ──
        vix_ok = vix < vix_max
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entry ──
        long_entry = zone_ok & bullish_pin & (close > vwap) & vix_ok & in_session
        short_entry = zone_ok & bearish_pin & (close < vwap) & vix_ok & in_session

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr20,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_pct=0.005,
            breakeven_pct=0.004,
            time_stop_bars=40,
        )
