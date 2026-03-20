"""Closing Auction Momentum — Opus_34

Thesis: In the last 30 minutes of trading, momentum into the close tends
to persist. Enter on strong directional moves with volume acceleration
above VWAP.
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


class Strategy(BaseStrategy):
    name = "opus_closing_auction_v1"
    is_long_only = False
    session_start = 900   # 15:00
    session_end = 929     # 15:29
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ret_thresh", default=0.0015, low=0.0008, high=0.003),
            TunableParam("vol_accel_thresh", default=1.5, low=1.2, high=2.5),
            TunableParam("target_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ret_thresh = params.get("ret_thresh", 0.0015)
        vol_accel_thresh = params.get("vol_accel_thresh", 1.5)
        target_pct = params.get("target_pct", 0.002)
        stop_pct = params.get("stop_loss_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── 30-bar return ──
        ret_30 = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            if close[i - 30] > 1e-10:
                ret_30[i] = (close[i] - close[i - 30]) / close[i - 30]

        # ── Volume acceleration: current volume / avg of last 30 bars ──
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_accel = volume / avg_vol

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Filters ──
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entries ──
        long_entry = (
            (ret_30 > ret_thresh)
            & (close > vwap)
            & (vol_accel > vol_accel_thresh)
            & time_ok
        )
        short_entry = (
            (ret_30 < -ret_thresh)
            & (close < vwap)
            & (vol_accel > vol_accel_thresh)
            & time_ok
        )

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=29,
        )
