"""VWAP-RSI-Volume Triple Confluence — Opus_41

Thesis: When price is extended from VWAP (zscore > 1.5), RSI is
extreme (<20 or >80), and volume spikes (>3x average), a mean
reversion move is likely. Signal exit on RSI crossing 50 or zscore 0.
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
    name = "opus_vwap_rsi_volume_confluence_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=1.5, low=1.0, high=2.5),
            TunableParam("rsi_long_thresh", default=20.0, low=10.0, high=30.0),
            TunableParam("rsi_short_thresh", default=80.0, low=70.0, high=90.0),
            TunableParam("vol_ratio_thresh", default=3.0, low=2.0, high=5.0),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("breakeven_pct", default=0.0025, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 1.5)
        rsi_long = params.get("rsi_long_thresh", 20.0)
        rsi_short = params.get("rsi_short_thresh", 80.0)
        vol_ratio = params.get("vol_ratio_thresh", 3.0)
        vix_max = params.get("vix_max", 22.0)
        be_pct = params.get("breakeven_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
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

        # ── Z-score ──
        dev = close - vwap
        std_30 = df["close"].rolling_std(30).to_numpy().astype(np.float64)
        std_30 = np.nan_to_num(std_30, nan=1e10)
        std_30 = np.clip(std_30, 1e-10, None)
        zscore = dev / std_30
        zscore = np.nan_to_num(zscore, nan=0.0)

        rsi = _compute_rsi(close, 7)
        atr = _compute_atr(high, low, close, 20)

        # ── Volume ratio ──
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_ratio * avg_vol

        vix_ok = vix < vix_max
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entries ──
        long_entry = ((zscore < -zs_thresh) & (rsi < rsi_long) &
                      vol_ok & vix_ok & time_ok)
        short_entry = ((zscore > zs_thresh) & (rsi > rsi_short) &
                       vol_ok & vix_ok & time_ok)

        # ── Signal exits: RSI crosses 50 OR zscore crosses 0 ──
        rsi_cross_50_up = np.zeros(n, dtype=np.bool_)
        rsi_cross_50_down = np.zeros(n, dtype=np.bool_)
        zs_cross_0_up = np.zeros(n, dtype=np.bool_)
        zs_cross_0_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if rsi[i - 1] < 50.0 and rsi[i] >= 50.0:
                rsi_cross_50_up[i] = True
            if rsi[i - 1] > 50.0 and rsi[i] <= 50.0:
                rsi_cross_50_down[i] = True
            if zscore[i - 1] < 0 and zscore[i] >= 0:
                zs_cross_0_up[i] = True
            if zscore[i - 1] > 0 and zscore[i] <= 0:
                zs_cross_0_down[i] = True

        exit_long = rsi_cross_50_up | zs_cross_0_up
        exit_short = rsi_cross_50_down | zs_cross_0_down

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=exit_long,
            signal_exit_short=exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=1.2,
            breakeven_pct=be_pct,
            time_stop_bars=45,
        )
