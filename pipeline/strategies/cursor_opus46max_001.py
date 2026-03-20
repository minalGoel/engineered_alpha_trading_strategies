"""VWAP Mean Reversion Basic v1 — cursor_opus46max_001

Thesis: Stocks deviating significantly from VWAP during the first 90 minutes
revert because VWAP represents institutional fair price. Enter on deviation
with volume + RSI confirmation after two consecutive confirming bars.
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
    name = "cursor_opus46max_001"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_thresh", default=0.6, low=0.3, high=1.2),
            TunableParam("vol_ratio_thresh", default=1.5, low=1.0, high=3.0),
            TunableParam("rsi_long_thresh", default=35.0, low=20.0, high=45.0),
            TunableParam("rsi_short_thresh", default=65.0, low=55.0, high=80.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
            TunableParam("breakeven_pct", default=0.003, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        dev_thresh = params.get("vwap_dev_thresh", 0.6)
        vol_thresh = params.get("vol_ratio_thresh", 1.5)
        rsi_long = params.get("rsi_long_thresh", 35.0)
        rsi_short = params.get("rsi_short_thresh", 65.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        be_pct = params.get("breakeven_pct", 0.003)

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
        opn = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP deviation % ──
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        vwap_dev_pct = (close - vwap) / safe_vwap * 100.0

        # ── Volume ratio ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        # ── ATR(14) ──
        atr = _compute_atr(high, low, close, 14)

        # ── Two consecutive green/red bar confirmation ──
        green2 = np.zeros(n, dtype=np.bool_)
        red2 = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            green2[i] = (close[i] > close[i - 1]) and (close[i - 1] > close[i - 2])
            red2[i] = (close[i] < close[i - 1]) and (close[i - 1] < close[i - 2])

        # ── Time / VIX filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)
        vix_ok = vix < vix_max

        # ── Entries ──
        long_cond = (
            (close < vwap * (1.0 - dev_thresh / 100.0))
            & (vwap_dev_pct < -dev_thresh)
            & (vol_ratio > vol_thresh)
            & (rsi < rsi_long)
            & vix_ok & time_ok & green2
        )
        short_cond = (
            (close > vwap * (1.0 + dev_thresh / 100.0))
            & (vwap_dev_pct > dev_thresh)
            & (vol_ratio > vol_thresh)
            & (rsi > rsi_short)
            & vix_ok & time_ok & red2
        )

        # ── Signal exit: VWAP deviation crosses zero ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if vwap_dev_pct[i - 1] < 0 and vwap_dev_pct[i] >= 0:
                sig_exit_long[i] = True
            if vwap_dev_pct[i - 1] > 0 and vwap_dev_pct[i] <= 0:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_cond,
            short_entry=short_cond,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            stop_loss_atr_mult=1.5,
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=60,
        )
