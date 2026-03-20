"""Microstructure Reversion v1 — cursor_opus46max_017

Thesis: Inferred buy/sell volume imbalance from bar OHLCV detects
exhaustion of aggressive order flow. When buy-imbalance is extreme
(>0.70 or <0.30) and price stalls, enter for reversion.
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
    name = "cursor_opus46max_017"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("imbalance_long_thresh", default=0.30, low=0.15, high=0.40),
            TunableParam("imbalance_short_thresh", default=0.70, low=0.60, high=0.85),
            TunableParam("price_change_thresh", default=0.3, low=0.1, high=0.6),
            TunableParam("target_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        imb_long = params.get("imbalance_long_thresh", 0.30)
        imb_short = params.get("imbalance_short_thresh", 0.70)
        pc_thresh = params.get("price_change_thresh", 0.3)
        tgt_pct = params.get("target_pct", 0.002)
        stop_pct = params.get("stop_loss_pct", 0.002)

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
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Buy/Sell volume approximation ──
        bar_range = high - low
        bar_range = np.where(bar_range > 0, bar_range, 1e-10)
        buy_vol = volume * (close - low) / bar_range
        sell_vol = volume * (high - close) / bar_range

        # ── Rolling 10-bar buy imbalance ──
        buy_imb_10 = np.zeros(n, dtype=np.float64)
        for i in range(9, n):
            total = np.sum(volume[i - 9:i + 1])
            if total > 0:
                buy_imb_10[i] = np.sum(buy_vol[i - 9:i + 1]) / total
            else:
                buy_imb_10[i] = 0.5

        # ── Price change over 10 bars ──
        price_change_10 = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            if close[i - 10] > 0:
                price_change_10[i] = (close[i] - close[i - 10]) / close[i - 10] * 100.0

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 0.8 * avg_vol

        # ── ATR ──
        atr = _compute_atr(high, low, close, 10)

        # ── Confirmation: imbalance easing ──
        imb_easing_buy = np.zeros(n, dtype=np.bool_)
        imb_easing_sell = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            imb_easing_buy[i] = buy_imb_10[i] > buy_imb_10[i - 1]
            imb_easing_sell[i] = buy_imb_10[i] < buy_imb_10[i - 1]

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 885)

        # ── Entries ──
        long_entry = (
            (buy_imb_10 < imb_long)
            & (price_change_10 < -pc_thresh)
            & (close < vwap)
            & imb_easing_buy
            & vol_ok & time_ok
        )
        short_entry = (
            (buy_imb_10 > imb_short)
            & (price_change_10 > pc_thresh)
            & (close > vwap)
            & imb_easing_sell
            & vol_ok & time_ok
        )

        # ── Signal exit: imbalance deepens ──
        sig_exit_long = buy_imb_10 < 0.20
        sig_exit_short = buy_imb_10 > 0.80

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=20,
        )
