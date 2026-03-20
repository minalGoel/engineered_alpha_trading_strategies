# AUDIT FIX: Add session window filtering to prevent entries outside session_start/session_end
"""Defensive-Cyclical Switch v1 — cursor_opus46max_118

Thesis: Intraday rotation between defensive and cyclical sectors is predicted
by VIX changes within the session.  When VIX rises >0.5, money flows from
cyclicals to defensives.  Trade using VIX direction and stock-vs-index spread.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "cursor_opus46max_118"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 4
    assumptions = ["Defensive/cyclical rotation proxied via VIX change and VWAP position"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_change_thresh", default=0.5, low=0.3, high=1.0),
            TunableParam("stop_loss_pct", default=0.0015, low=0.0008, high=0.003),
            TunableParam("target_pct", default=0.002, low=0.001, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_chg_thresh = params.get("vix_change_thresh", 0.5)
        stop_pct = params.get("stop_loss_pct", 0.0015)
        target_pct = params.get("target_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # VIX change over 30 bars
        vix_change = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            vix_change[i] = vix[i] - vix[i-30]

        # Risk-off: VIX rising (buy defensives = stocks above VWAP in low-beta)
        # Risk-on: VIX falling (buy cyclicals = stocks that dropped below VWAP)
        # Since we trade a single stock, we use its VWAP position as the signal

        # VIX absolute level filter
        vix_active = vix > 12

        # Confirmation: VIX change direction and stock starting to move
        vix_rising = vix_change > vix_chg_thresh
        vix_falling = vix_change < -vix_chg_thresh

        # When VIX rises and stock is below VWAP (cyclical being sold), buy for bounce
        # When VIX falls and stock is above VWAP (defensive being bought), short for fade
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            # Risk-on: VIX falling, stock below VWAP = buy cyclical bounce
            if (vix_falling[i] and close[i] < vwap[i] and
                    close[i] > close[i-1] and vix_active[i] and in_session[i]):
                long_entry[i] = True
            # Risk-off: VIX rising, stock above VWAP = short cyclical
            if (vix_rising[i] and close[i] > vwap[i] and
                    close[i] < close[i-1] and vix_active[i] and in_session[i]):
                short_entry[i] = True

        # Signal exit: VIX regime changes
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if vix_rising[i] and not vix_rising[i-1]:
                sig_exit_long[i] = True
            if vix_falling[i] and not vix_falling[i-1]:
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_stop_pct=0.0008,
            trailing_activate_pct=0.0012,
            time_stop_bars=45,
        )
