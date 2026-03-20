"""ROC Momentum v1 — cursor_opus46max_024

Thesis: ROC(20) exceeding its 95th percentile (approximated within session)
signals an abnormally strong momentum burst. Enter with volume surge > 2x
and VWAP alignment after 2-bar sustained extreme confirmation.
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
    name = "cursor_opus46max_024"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("roc_abs_thresh", default=0.3, low=0.15, high=0.6),
            TunableParam("vol_surge_mult", default=2.0, low=1.3, high=3.0),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("trailing_pct", default=0.0015, low=0.0008, high=0.003),
            TunableParam("trailing_activate", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        roc_abs = params.get("roc_abs_thresh", 0.3)
        vol_surge = params.get("vol_surge_mult", 2.0)
        tgt_pct = params.get("target_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.002)
        trail_pct = params.get("trailing_pct", 0.0015)
        trail_act = params.get("trailing_activate", 0.003)

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
        day_id = df["day_id"].to_numpy()

        # ── ROC(20) ──
        roc20 = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            if close[i - 20] > 1e-10:
                roc20[i] = (close[i] - close[i - 20]) / close[i - 20] * 100.0

        # ── ROC percentile (rolling 200-bar window as proxy for ~5 session
        #    distribution; fall back to available data if fewer bars) ──
        roc_pctile = np.full(n, 50.0, dtype=np.float64)
        for i in range(40, n):
            lookback = min(i + 1, 200)
            window = roc20[i - lookback + 1:i + 1]
            roc_pctile[i] = np.sum(window <= roc20[i]) / len(window) * 100.0

        # ── Volume filter ──
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > vol_surge * avg_vol

        # ── ATR ──
        atr = _compute_atr(high, low, close, 14)

        # ── 2-bar sustained extreme confirmation ──
        long_extreme = (roc_pctile > 90) & (roc20 > roc_abs)
        short_extreme = (roc_pctile < 10) & (roc20 < -roc_abs)

        sustained_long = np.zeros(n, dtype=np.bool_)
        sustained_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if long_extreme[i] and long_extreme[i - 1]:
                sustained_long[i] = True
            if short_extreme[i] and short_extreme[i - 1]:
                sustained_short[i] = True

        # ── Filters ──
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # ── Entries ──
        long_entry = (
            (roc_pctile > 95)
            & (roc20 > roc_abs)
            & sustained_long
            & (close > vwap)
            & vol_ok & time_ok
        )
        short_entry = (
            (roc_pctile < 5)
            & (roc20 < -roc_abs)
            & sustained_short
            & (close < vwap)
            & vol_ok & time_ok
        )

        # ── Signal exit: ROC percentile normalizes (drops below 50 for longs,
        #    rises above 50 for shorts) ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if roc_pctile[i] < 50 and roc_pctile[i - 1] >= 50:
                sig_exit_long[i] = True
            if roc20[i] < 0:
                sig_exit_long[i] = True
            if roc_pctile[i] > 50 and roc_pctile[i - 1] <= 50:
                sig_exit_short[i] = True
            if roc20[i] > 0:
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
            time_stop_bars=30,
        )
