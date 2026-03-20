"""Keltner Channel Breakout — Opus_27

Thesis: Price breaking above/below the Keltner Channel (EMA ± 2*ATR)
with high relative volume signals a strong directional move. VIX and
VWAP filters reduce false breakouts.
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


def _compute_ema(data, period):
    """EMA."""
    n = len(data)
    ema = np.zeros(n, dtype=np.float64)
    if n < period:
        return ema
    ema[period - 1] = np.mean(data[:period])
    k = 2.0 / (period + 1)
    for i in range(period, n):
        ema[i] = data[i] * k + ema[i - 1] * (1 - k)
    return ema


class Strategy(BaseStrategy):
    name = "opus_atr_channel_breakout_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("kc_mult", default=2.0, low=1.5, high=3.0),
            TunableParam("rel_vol_thresh", default=2.0, low=1.2, high=3.5),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("target_atr_mult", default=1.5, low=1.0, high=3.0),
            TunableParam("stop_atr_mult", default=2.0, low=1.0, high=3.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        kc_mult = params.get("kc_mult", 2.0)
        rel_vol_thresh = params.get("rel_vol_thresh", 2.0)
        vix_max = params.get("vix_max", 22.0)
        target_atr = params.get("target_atr_mult", 1.5)
        stop_atr = params.get("stop_atr_mult", 2.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── EMA(20) and ATR(20) ──
        ema20 = _compute_ema(close, 20)
        atr = _compute_atr(high, low, close, 20)

        # ── Keltner Channels ──
        kc_upper = ema20 + kc_mult * atr
        kc_lower = ema20 - kc_mult * atr

        # ── Relative volume ──
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        rel_vol = volume / avg_vol

        # ── Filters ──
        vix_ok = vix < vix_max
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entries ──
        long_entry = (
            (close > kc_upper)
            & (rel_vol > rel_vol_thresh)
            & vix_ok
            & (close > vwap)
            & time_ok
        )
        short_entry = (
            (close < kc_lower)
            & (rel_vol > rel_vol_thresh)
            & time_ok
        )

        # ── Signal exit: close crosses back inside channel ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if close[i] < kc_upper[i] and close[i - 1] >= kc_upper[i - 1]:
                sig_exit_long[i] = True
            if close[i] > kc_lower[i] and close[i - 1] <= kc_lower[i - 1]:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_atr_mult=target_atr,
            stop_loss_atr_mult=stop_atr,
            time_stop_bars=60,
        )
