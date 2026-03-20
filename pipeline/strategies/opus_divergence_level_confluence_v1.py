"""RSI Divergence at Key Levels — Opus_48

Thesis: RSI(9) divergence (price makes new low but RSI doesn't) at
key support/resistance levels (VWAP, PDH, PDL) signals high-probability
reversals. Enter on confirmed divergence + green/red candle.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[1:period + 1])
    avg_loss = np.mean(loss[1:period + 1])
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


class Strategy(BaseStrategy):
    name = "opus_divergence_level_confluence_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("level_proximity_pct", default=0.001, low=0.0005, high=0.002),
            TunableParam("rsi_long_thresh", default=30.0, low=20.0, high=40.0),
            TunableParam("rsi_short_thresh", default=70.0, low=60.0, high=80.0),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("breakeven_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        level_prox = params.get("level_proximity_pct", 0.001)
        rsi_long = params.get("rsi_long_thresh", 30.0)
        rsi_short = params.get("rsi_short_thresh", 70.0)
        tgt_pct = params.get("target_pct", 0.004)
        stp_pct = params.get("stop_loss_pct", 0.0025)
        be_pct = params.get("breakeven_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        rsi = _compute_rsi(close, 9)
        atr = _compute_atr(high, low, close, 20)

        # ── Previous day high/low (PDH/PDL) ──
        pdh = np.zeros(n, dtype=np.float64)
        pdl = np.zeros(n, dtype=np.float64)

        # Collect per-day high/low
        day_highs = {}
        day_lows = {}
        for i in range(n):
            d = day_id[i]
            if d not in day_highs:
                day_highs[d] = high[i]
                day_lows[d] = low[i]
            else:
                day_highs[d] = max(day_highs[d], high[i])
                day_lows[d] = min(day_lows[d], low[i])

        unique_days = sorted(day_highs.keys())
        pdh_map = {}
        pdl_map = {}
        for idx, d in enumerate(unique_days):
            if idx == 0:
                pdh_map[d] = 0.0
                pdl_map[d] = 0.0
            else:
                prev_d = unique_days[idx - 1]
                pdh_map[d] = day_highs[prev_d]
                pdl_map[d] = day_lows[prev_d]

        for i in range(n):
            pdh[i] = pdh_map.get(day_id[i], 0.0)
            pdl[i] = pdl_map.get(day_id[i], 0.0)

        # ── At key level: close within level_prox% of VWAP, PDH, or PDL ──
        safe_close = np.where(close > 0, close, 1.0)
        near_vwap = np.abs(close - vwap) / safe_close < level_prox
        near_pdh = (pdh > 0) & (np.abs(close - pdh) / safe_close < level_prox)
        near_pdl = (pdl > 0) & (np.abs(close - pdl) / safe_close < level_prox)
        at_level = near_vwap | near_pdh | near_pdl

        # ── RSI divergence detection (20-bar lookback) ──
        lookback = 20
        bullish_div = np.zeros(n, dtype=np.bool_)
        bearish_div = np.zeros(n, dtype=np.bool_)

        for i in range(lookback, n):
            window_start = i - lookback
            # Bullish: price lower low but RSI higher low
            price_min_idx = window_start + np.argmin(low[window_start:i])
            if low[i] < low[price_min_idx] and rsi[i] > rsi[price_min_idx]:
                bullish_div[i] = True
            # Bearish: price higher high but RSI lower high
            price_max_idx = window_start + np.argmax(high[window_start:i])
            if high[i] > high[price_max_idx] and rsi[i] < rsi[price_max_idx]:
                bearish_div[i] = True

        green_bar = close > open_
        red_bar = close < open_
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Long: bullish div + at level + RSI < 30 + green bar
        long_entry = (bullish_div & at_level & (rsi < rsi_long) & green_bar & time_ok)

        # Short: bearish div + at level + RSI > 70 + red bar
        short_entry = (bearish_div & at_level & (rsi > rsi_short) & red_bar & time_ok)

        # Signal exit: RSI crosses 50
        exit_long = np.zeros(n, dtype=np.bool_)
        exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if rsi[i - 1] < 50.0 and rsi[i] >= 50.0:
                exit_long[i] = True
            if rsi[i - 1] > 50.0 and rsi[i] <= 50.0:
                exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=exit_long,
            signal_exit_short=exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stp_pct,
            breakeven_pct=be_pct,
            time_stop_bars=60,
        )
