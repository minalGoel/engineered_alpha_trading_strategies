"""Multi-Asset Signal v1 — cursor_opus46max_163

Thesis: Cross-asset signals from NIFTY-VIX divergence. When NIFTY falls
but VIX also falls (complacent dip), go long oversold stocks. When NIFTY
rises but VIX rises (hedging divergence), short overbought stocks.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


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
    name = "cursor_opus46max_163"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("nifty_roc_bps", default=15.0, low=8.0, high=30.0),
            TunableParam("rsi_long_thresh", default=40.0, low=30.0, high=45.0),
            TunableParam("rsi_short_thresh", default=60.0, low=55.0, high=70.0),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("trailing_stop_pct", default=0.0015, low=0.001, high=0.003),
            TunableParam("trailing_activate_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        nifty_roc_bps = params.get("nifty_roc_bps", 15.0)
        rsi_long = params.get("rsi_long_thresh", 40.0)
        rsi_short = params.get("rsi_short_thresh", 60.0)
        target_pct = params.get("target_pct", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        trail_pct = params.get("trailing_stop_pct", 0.0015)
        trail_act = params.get("trailing_activate_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        idx_close = df["index_close"].to_numpy().astype(np.float64)
        idx_close = np.nan_to_num(idx_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        rsi = _compute_rsi(close, 14)

        # NIFTY 15-bar ROC (bps)
        nifty_roc15 = np.zeros(n, dtype=np.float64)
        for i in range(15, n):
            if idx_close[i - 15] > 0:
                nifty_roc15[i] = (idx_close[i] / idx_close[i - 15] - 1.0) * 10000.0

        # VIX 15-bar ROC
        vix_roc15 = np.zeros(n, dtype=np.float64)
        for i in range(15, n):
            if vix[i - 15] > 0.1:
                vix_roc15[i] = (vix[i] / vix[i - 15] - 1.0) * 100.0

        # Divergence: NIFTY and VIX moving in same direction (unusual)
        divergence = np.sign(nifty_roc15) * np.sign(vix_roc15)

        # Divergence persists for 5+ bars
        div_persist = np.zeros(n, dtype=np.bool_)
        for i in range(4, n):
            if all(divergence[i - j] > 0 for j in range(5)):
                div_persist[i] = True

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        time_ok = (time_mins >= 560) & (time_mins <= 915)
        vix_range_ok = (vix >= 12.0) & (vix <= 28.0)

        # Long: NIFTY down + VIX down (complacent dip), stock oversold below VWAP
        long_entry = (nifty_roc15 < -nifty_roc_bps) & (vix_roc15 < 0) & \
                     div_persist & (close < vwap) & (rsi < rsi_long) & \
                     vol_ok & time_ok & vix_range_ok

        # Short: NIFTY up + VIX up (hedging divergence), stock overbought above VWAP
        short_entry = (nifty_roc15 > nifty_roc_bps) & (vix_roc15 > 0) & \
                      div_persist & (close > vwap) & (rsi > rsi_short) & \
                      vol_ok & time_ok & vix_range_ok

        # Signal exit: divergence resolves (NIFTY and VIX re-align)
        sig_exit_long = divergence <= 0
        sig_exit_short = divergence <= 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=60,
        )
