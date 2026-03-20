"""VWAP RSI Confluence v1 — cursor_opus46max_008

Thesis: VWAP deviation >0.5% AND RSI(14)<30 together provide double
confirmation for mean reversion. Enter when RSI turns up from extreme
with bullish bar. Target is VWAP touch or RSI crossing 50.
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
    name = "cursor_opus46max_008"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 900     # 15:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vwap_dev_thresh", default=0.5, low=0.3, high=1.0),
            TunableParam("rsi_long_thresh", default=30.0, low=15.0, high=40.0),
            TunableParam("rsi_short_thresh", default=70.0, low=60.0, high=85.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.007),
            TunableParam("trailing_pct", default=0.0015, low=0.0008, high=0.003),
            TunableParam("trailing_activate", default=0.0025, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        dev_thresh = params.get("vwap_dev_thresh", 0.5)
        rsi_long = params.get("rsi_long_thresh", 30.0)
        rsi_short = params.get("rsi_short_thresh", 70.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.004)
        trail_pct = params.get("trailing_pct", 0.0015)
        trail_act = params.get("trailing_activate", 0.0025)

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

        # ── VWAP deviation pct ──
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        vwap_dev_pct = (close - vwap) / safe_vwap * 100.0

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        # ── RSI slope (3-bar) ──
        rsi_slope = np.zeros(n, dtype=np.float64)
        for i in range(3, n):
            rsi_slope[i] = rsi[i] - rsi[i - 3]

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── Bar direction (bullish/bearish) ──
        bullish = close > opn
        bearish = close < opn

        # ── Filters ──
        time_ok = (time_mins >= 585) & (time_mins <= 870)
        vix_ok = vix < vix_max

        # ── Entries ──
        long_entry = (
            (vwap_dev_pct < -dev_thresh)
            & (rsi < rsi_long)
            & (rsi_slope > 0)
            & bullish
            & vol_ok & vix_ok & time_ok
        )
        short_entry = (
            (vwap_dev_pct > dev_thresh)
            & (rsi > rsi_short)
            & (rsi_slope < 0)
            & bearish
            & vol_ok & vix_ok & time_ok
        )

        # ── Signal exit: deepening extreme ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # exit if both dev and RSI worsen
            if (vwap_dev_pct[i] < vwap_dev_pct[i - 1]) and (rsi[i] < rsi[i - 1]) and (rsi[i] < 25):
                sig_exit_long[i] = True
            if (vwap_dev_pct[i] > vwap_dev_pct[i - 1]) and (rsi[i] > rsi[i - 1]) and (rsi[i] > 75):
                sig_exit_short[i] = True
            # or exit on reversion target
            if vwap_dev_pct[i] >= 0 and vwap_dev_pct[i - 1] < 0:
                sig_exit_long[i] = True
            if rsi[i] >= 50 and rsi[i - 1] < 50:
                sig_exit_long[i] = True
            if vwap_dev_pct[i] <= 0 and vwap_dev_pct[i - 1] > 0:
                sig_exit_short[i] = True
            if rsi[i] <= 50 and rsi[i - 1] > 50:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=40,
        )
