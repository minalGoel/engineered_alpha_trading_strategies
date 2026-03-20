"""Gap-VWAP-Momentum Trifecta — Opus_43

Thesis: After a gap > 0.3%, price pulls back to VWAP and bounces.
Enter on confirmed VWAP touch (within 0.1%) when close is back above
VWAP. Exit on VWAP cross against trade.
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
    name = "opus_gap_vwap_momentum_trifecta_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 660     # 11:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh", default=0.003, low=0.002, high=0.006),
            TunableParam("vwap_touch_pct", default=0.001, low=0.0005, high=0.002),
            TunableParam("target_pct", default=0.004, low=0.002, high=0.008),
            TunableParam("stop_loss_pct", default=0.0025, low=0.0015, high=0.004),
            TunableParam("trailing_stop_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("trailing_activate_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_thresh = params.get("gap_thresh", 0.003)
        vwap_touch = params.get("vwap_touch_pct", 0.001)
        tgt_pct = params.get("target_pct", 0.004)
        stp_pct = params.get("stop_loss_pct", 0.0025)
        trail_pct = params.get("trailing_stop_pct", 0.002)
        trail_act = params.get("trailing_activate_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── VWAP ──
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        atr = _compute_atr(high, low, close, 20)

        # ── Gap pct per day (open vs prev close) ──
        gap_pct = np.zeros(n, dtype=np.float64)
        open_of_day = np.zeros(n, dtype=np.float64)
        prev_day = -1
        prev_close = 0.0
        cur_open = 0.0

        for i in range(n):
            if day_id[i] != prev_day:
                if prev_day != -1:
                    prev_close = close[i - 1]
                cur_open = open_[i]
                prev_day = day_id[i]
            open_of_day[i] = cur_open
            if prev_close > 0:
                gap_pct[i] = (cur_open - prev_close) / prev_close
            else:
                gap_pct[i] = 0.0

        # ── VWAP touch: low within vwap_touch% of vwap (long), high within (short) ──
        safe_vwap = np.where(vwap > 0, vwap, 1.0)
        low_touches_vwap = np.abs(low - vwap) / safe_vwap < vwap_touch
        high_touches_vwap = np.abs(high - vwap) / safe_vwap < vwap_touch

        # Return from open
        safe_open = np.where(open_of_day > 0, open_of_day, 1.0)
        ret_from_open = (close - open_of_day) / safe_open

        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Long: gap > thresh, low touches VWAP, close > VWAP, ret_from_open > 0
        long_entry = ((gap_pct > gap_thresh) & low_touches_vwap &
                      (close > vwap) & (ret_from_open > 0) & time_ok)

        # Short: gap < -thresh, high touches VWAP, close < VWAP
        short_entry = ((gap_pct < -gap_thresh) & high_touches_vwap &
                       (close < vwap) & (ret_from_open < 0) & time_ok)

        # Signal exit: close crosses VWAP against trade
        exit_long = close < vwap
        exit_short = close > vwap

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=exit_long,
            signal_exit_short=exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=tgt_pct,
            stop_loss_pct=stp_pct,
            trailing_stop_pct=trail_pct,
            trailing_activate_pct=trail_act,
            time_stop_bars=90,
        )
