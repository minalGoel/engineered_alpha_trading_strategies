"""RSI Extreme Reversion — Opus_3

Thesis: RSI(7) hitting extreme levels (<15 or >85) with VWAP distance and
volume confirmation signals high-probability short-term reversals.
Exit when RSI crosses 50.
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
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    avg_gain[period] = np.mean(gain[1:period + 1])
    avg_loss[period] = np.mean(loss[1:period + 1])
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain[i] / avg_loss[i])
    return rsi


class Strategy(BaseStrategy):
    name = "opus_rsi_extreme_reversion_v1"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 900     # 15:00
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_long_thresh", default=15.0, low=8.0, high=25.0),
            TunableParam("rsi_short_thresh", default=85.0, low=75.0, high=92.0),
            TunableParam("vwap_dist_bps", default=30.0, low=15.0, high=60.0),
            TunableParam("rel_vol_thresh", default=1.5, low=1.0, high=3.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rsi_long = params.get("rsi_long_thresh", 15.0)
        rsi_short = params.get("rsi_short_thresh", 85.0)
        vwap_dist_bps = params.get("vwap_dist_bps", 30.0)
        rel_vol_thresh = params.get("rel_vol_thresh", 1.5)
        vix_max = params.get("vix_max", 22.0)

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
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── RSI(7) ──
        rsi = _compute_rsi(close, 7)

        # ── VWAP distance in bps ──
        safe_vwap = np.clip(np.abs(vwap), 1e-10, None)
        vwap_dist = (close - vwap) / safe_vwap * 10000.0  # bps

        # ── Relative volume ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > rel_vol_thresh * avg_vol

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── Filters ──
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 565) & (time_mins <= 900)

        # ── Entries ──
        long_entry = ((rsi < rsi_long) & (vwap_dist < -vwap_dist_bps)
                      & vol_ok & vix_ok & time_ok)
        short_entry = ((rsi > rsi_short) & (vwap_dist > vwap_dist_bps)
                       & vol_ok & vix_ok & time_ok)

        # ── Signal exit: RSI crosses 50 ──
        rsi_prev = np.roll(rsi, 1)
        rsi_prev[0] = 50.0
        signal_exit_long = (rsi_prev < 50.0) & (rsi >= 50.0)
        signal_exit_short = (rsi_prev > 50.0) & (rsi <= 50.0)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=1.0,
            time_stop_bars=30,
        )
