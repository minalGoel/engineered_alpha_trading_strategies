"""RBI Policy Reaction v1 — cursor_opus46max_122

Thesis: RBI monetary policy announcements create a predictable pattern:
knee-jerk reaction, partial reversal, then trend resumption.  Trade
the reversal phase and trend phase using VIX-confirmed price action.
Proxy RBI day using VIX spikes around 10:00 IST.
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
    name = "cursor_opus46max_122"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 780     # 13:00 (post-RBI volatility settles by afternoon)
    max_trades_per_day = 3
    assumptions = ["RBI policy days detected via VIX spike pattern around 10:00"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_spike_thresh", default=0.5, low=0.3, high=1.0),
            TunableParam("reaction_bps", default=50.0, low=30.0, high=100.0),
            TunableParam("retracement_pct", default=0.3, low=0.2, high=0.5),
            TunableParam("stop_loss_pct", default=0.002, low=0.001, high=0.004),
            TunableParam("target_pct", default=0.003, low=0.0015, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_spike = params.get("vix_spike_thresh", 0.5)
        reaction_bps = params.get("reaction_bps", 50.0)
        retrace_pct = params.get("retracement_pct", 0.3)
        stop_pct = params.get("stop_loss_pct", 0.002)
        target_pct = params.get("target_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Detect VIX spike (proxy for event)
        vix_at_945 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if time_mins[i] <= 585:
                vix_at_945[i] = vix[i]
            elif i > 0 and day_id[i] == day_id[i-1]:
                vix_at_945[i] = vix_at_945[i-1]
            else:
                vix_at_945[i] = vix[i]
        vix_change = vix - vix_at_945

        # Price reaction from 10:00 level
        close_at_1000 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if time_mins[i] <= 600:
                close_at_1000[i] = close[i]
            elif i > 0 and day_id[i] == day_id[i-1]:
                close_at_1000[i] = close_at_1000[i-1]
            else:
                close_at_1000[i] = close[i]

        safe_base = np.where(close_at_1000 > 1e-10, close_at_1000, 1e-10)
        reaction = (close - close_at_1000) / safe_base * 10000.0

        # Phase 3 (10:15-10:45): fade the overshoot
        phase3 = (time_mins >= 615) & (time_mins <= 645)
        # Phase 4 (10:45-12:00): follow the confirmed trend
        phase4 = (time_mins >= 645) & (time_mins <= 720)

        # Event detection: |VIX change| > threshold
        event_detected = np.abs(vix_change) > vix_spike

        # Phase 3: fade (buy after sell-off, sell after rally)
        phase3_long = np.zeros(n, dtype=np.bool_)
        phase3_short = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            if (phase3[i] and event_detected[i] and reaction[i] < -reaction_bps and
                    close[i] > close[i-1] and close[i-1] > close[i-2]):
                phase3_long[i] = True
            if (phase3[i] and event_detected[i] and reaction[i] > reaction_bps and
                    close[i] < close[i-1] and close[i-1] < close[i-2]):
                phase3_short[i] = True

        # Phase 4: trend follow
        phase4_long = np.zeros(n, dtype=np.bool_)
        phase4_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (phase4[i] and event_detected[i] and reaction[i] > 0 and
                    close[i] > vwap[i]):
                phase4_long[i] = True
            if (phase4[i] and event_detected[i] and reaction[i] < 0 and
                    close[i] < vwap[i]):
                phase4_short[i] = True

        long_entry = phase3_long | phase4_long
        short_entry = phase3_short | phase4_short

        # Signal exit: VIX spike reverses
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if vix_change[i] * vix_change[i-1] < 0:  # sign change
                sig_exit_long[i] = True
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
            trailing_stop_pct=0.001,
            trailing_activate_pct=0.002,
            time_stop_bars=60,
        )
