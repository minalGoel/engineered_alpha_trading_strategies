"""VWAP Crossover Flow — Opus_33

Thesis: After price spends 30+ bars on one side of VWAP then crosses
with a volume surge (3x), it signals a regime change and continuation
in the cross direction.
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
    name = "opus_vwap_crossover_flow_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bars_side_thresh", default=30.0, low=15.0, high=50.0),
            TunableParam("vol_surge_thresh", default=3.0, low=2.0, high=5.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=30.0),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("breakeven_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        bars_side_thresh = int(params.get("bars_side_thresh", 30.0))
        vol_surge_thresh = params.get("vol_surge_thresh", 3.0)
        vix_max = params.get("vix_max", 22.0)
        target_pct = params.get("target_pct", 0.004)
        stop_pct = params.get("stop_loss_pct", 0.003)
        be_pct = params.get("breakeven_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy().astype(np.int64)

        # ── VWAP (daily reset) ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # ── Track consecutive bars above/below VWAP (daily reset) ──
        bars_above = np.zeros(n, dtype=np.int32)
        bars_below = np.zeros(n, dtype=np.int32)
        cur_above = 0
        cur_below = 0
        prev_day = -1
        for i in range(n):
            if day_id[i] != prev_day:
                cur_above = 0
                cur_below = 0
                prev_day = day_id[i]
            if close[i] > vwap[i]:
                cur_above += 1
                cur_below = 0
            elif close[i] < vwap[i]:
                cur_below += 1
                cur_above = 0
            else:
                pass  # keep counts
            bars_above[i] = cur_above
            bars_below[i] = cur_below

        # ── VWAP cross detection ──
        vwap_cross_up = np.zeros(n, dtype=np.bool_)
        vwap_cross_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if day_id[i] == day_id[i - 1]:
                if close[i] > vwap[i] and close[i - 1] <= vwap[i - 1]:
                    vwap_cross_up[i] = True
                if close[i] < vwap[i] and close[i - 1] >= vwap[i - 1]:
                    vwap_cross_down[i] = True

        # ── Volume surge ──
        avg_vol = df["volume"].rolling_mean(30).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_surge = volume / avg_vol

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Filters ──
        vix_ok = vix < vix_max
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── For cross-up, we need bars_below at bar i-1 ──
        prev_bars_below = np.zeros(n, dtype=np.int32)
        prev_bars_above = np.zeros(n, dtype=np.int32)
        prev_bars_below[1:] = bars_below[:-1]
        prev_bars_above[1:] = bars_above[:-1]

        # ── Entries ──
        long_entry = (
            vwap_cross_up
            & (prev_bars_below >= bars_side_thresh)
            & (vol_surge > vol_surge_thresh)
            & vix_ok
            & time_ok
        )
        short_entry = (
            vwap_cross_down
            & (prev_bars_above >= bars_side_thresh)
            & (vol_surge > vol_surge_thresh)
            & time_ok
        )

        # ── Signal exit: close crosses back ──
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if day_id[i] == day_id[i - 1]:
                if close[i] < vwap[i] and close[i - 1] >= vwap[i - 1]:
                    sig_exit_long[i] = True
                if close[i] > vwap[i] and close[i - 1] <= vwap[i - 1]:
                    sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            breakeven_pct=be_pct,
            time_stop_bars=90,
        )
