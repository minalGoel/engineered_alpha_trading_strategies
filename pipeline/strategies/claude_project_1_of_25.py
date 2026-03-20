"""VWAP Mean Reversion v1 — claude_project_1_of_25

Thesis: Price overshoots VWAP and reverts. Enter on extreme z-score
deviations with volume and RSI confirmation; target is VWAP touch.
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
    name = "vwap_mean_reversion_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=2.0, low=1.2, high=3.0),
            TunableParam("rel_vol_thresh", default=1.2, low=0.8, high=2.5),
            TunableParam("rsi_long_thresh", default=35.0, low=20.0, high=45.0),
            TunableParam("rsi_short_thresh", default=65.0, low=55.0, high=80.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
            TunableParam("breakeven_pct", default=0.0025, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 2.0)
        rel_vol = params.get("rel_vol_thresh", 1.2)
        rsi_long = params.get("rsi_long_thresh", 35.0)
        rsi_short = params.get("rsi_short_thresh", 65.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.005)
        be_pct = params.get("breakeven_pct", 0.0025)

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
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Z-score ──
        dev = close - vwap
        std_30 = df["close"].rolling_std(30).to_numpy().astype(np.float64)
        std_30 = np.nan_to_num(std_30, nan=1e10)
        std_30 = np.clip(std_30, 1e-10, None)
        zscore = dev / std_30
        zscore = np.nan_to_num(zscore, nan=0.0)

        # ── RSI(14) ──
        rsi = _compute_rsi(close, 14)

        # ── Relative volume ──
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > rel_vol * avg_vol

        # ── Time filter: skip first/last 15 min ──
        time_ok = (time_mins >= 585) & (time_mins <= 900)

        # ── VIX filter ──
        vix_ok = vix < vix_max

        # ── Entries ──
        long_entry = (zscore < -zs_thresh) & vol_ok & (rsi < rsi_long) & vix_ok & time_ok
        short_entry = (zscore > zs_thresh) & vol_ok & (rsi > rsi_short) & vix_ok & time_ok

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=vwap.copy(),
            stop_loss_pct=stop_pct,
            use_target_indicator=True,
            breakeven_pct=be_pct,
            time_stop_bars=90,
        )
