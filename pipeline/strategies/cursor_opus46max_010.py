"""VWAP Relative Strength v1 — cursor_opus46max_010

Thesis: Comparing stock's VWAP deviation to NIFTY 50's VWAP deviation
reveals relative strength. Stocks showing VWAP-normalised outperformance
tend to continue due to institutional rotation flow.
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


def _compute_ema(close, period):
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = close[0]
    k = 2.0 / (period + 1)
    for i in range(1, n):
        ema[i] = close[i] * k + ema[i - 1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_010"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rvs_thresh", default=0.5, low=0.2, high=1.0),
            TunableParam("confirm_bars", default=3.0, low=2.0, high=5.0),
            TunableParam("target_pct", default=0.006, low=0.003, high=0.01),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.005),
            TunableParam("trailing_pct", default=0.0025, low=0.001, high=0.004),
            TunableParam("trailing_activate", default=0.0035, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rvs_thresh = params.get("rvs_thresh", 0.5)
        cb = int(params.get("confirm_bars", 3))
        tgt_pct = params.get("target_pct", 0.006)
        stop_pct = params.get("stop_loss_pct", 0.003)
        trail_pct = params.get("trailing_pct", 0.0025)
        trail_act = params.get("trailing_activate", 0.0035)

        n = len(df)

        # ── Stock VWAP ──
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
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # ── Approximate index VWAP using index_close with equal volume weight ──
        # Use cumulative mean of index_close per day as proxy
        nifty_vwap = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_id)
        for d in unique_days:
            mask = day_id == d
            idx = np.where(mask)[0]
            cum = np.cumsum(index_close[idx])
            counts = np.arange(1, len(idx) + 1, dtype=np.float64)
            nifty_vwap[idx] = cum / counts

        # ── Stock and Nifty VWAP deviations ──
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        stock_vwap_dev = (close - vwap) / safe_vwap * 100.0

        safe_nvwap = np.where(nifty_vwap > 0, nifty_vwap, 1.0)
        nifty_vwap_dev = (index_close - nifty_vwap) / safe_nvwap * 100.0

        # ── Relative VWAP strength ──
        rvs = stock_vwap_dev - nifty_vwap_dev

        # ── RVS SMA(20) ──
        rvs_sma = np.zeros(n, dtype=np.float64)
        for i in range(19, n):
            rvs_sma[i] = np.mean(rvs[i - 19:i + 1])

        # ── EMA(9) ──
        ema9 = _compute_ema(close, 9)

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── RVS increasing / decreasing for N bars ──
        rvs_inc = np.zeros(n, dtype=np.bool_)
        rvs_dec = np.zeros(n, dtype=np.bool_)
        for i in range(cb, n):
            inc = True
            dec = True
            for j in range(cb):
                if rvs[i - j] <= rvs[i - j - 1]:
                    inc = False
                if rvs[i - j] >= rvs[i - j - 1]:
                    dec = False
            rvs_inc[i] = inc
            rvs_dec[i] = dec

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.8 * avg_vol

        # ── Bar count from open > 45 ──
        bar_from_open = np.zeros(n, dtype=np.int32)
        for d in unique_days:
            mask = day_id == d
            idx = np.where(mask)[0]
            for j, gi in enumerate(idx):
                bar_from_open[gi] = j

        # ── Filters ──
        time_ok = (time_mins >= 600) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            (rvs > rvs_thresh)
            & (rvs > rvs_sma)
            & (stock_vwap_dev > 0)
            & (bar_from_open > 45)
            & rvs_inc
            & (close > ema9)
            & vol_ok & time_ok
        )
        short_entry = (
            (rvs < -rvs_thresh)
            & (rvs < rvs_sma)
            & (stock_vwap_dev < 0)
            & (bar_from_open > 45)
            & rvs_dec
            & (close < ema9)
            & vol_ok & time_ok
        )

        # ── Signal exit: RVS reverts toward zero ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if rvs[i] <= 0 and rvs[i - 1] > 0:
                sig_exit_long[i] = True
            if rvs[i] >= 0 and rvs[i - 1] < 0:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stop_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=90,
        )
