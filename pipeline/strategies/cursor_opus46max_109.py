"""Trade Arrival Rate v1 — cursor_opus46max_109

Thesis: When trade arrival rate deviates significantly from expected intraday
pattern with consistent price direction, momentum persists for 3-8 min.
Proxy trade count with volume spikes relative to time-of-day average.
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
    name = "cursor_opus46max_109"
    is_long_only = False
    session_start = 580   # 09:40
    session_end = 920     # 15:20
    max_trades_per_day = 6
    assumptions = ["Trade count proxied via volume relative to rolling average"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("arrival_ratio_thresh", default=2.0, low=1.5, high=3.0),
            TunableParam("confirm_ratio", default=1.5, low=1.2, high=2.0),
            TunableParam("stop_loss_pct", default=0.0012, low=0.0006, high=0.002),
            TunableParam("target_pct", default=0.0025, low=0.0015, high=0.004),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        arr_thresh = params.get("arrival_ratio_thresh", 2.0)
        confirm_ratio = params.get("confirm_ratio", 1.5)
        stop_pct = params.get("stop_loss_pct", 0.0012)
        target_pct = params.get("target_pct", 0.0025)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = np.nan_to_num(df_v["_vwap"].to_numpy().astype(np.float64), nan=close[0] if n > 0 else 0.0)

        # Arrival ratio: volume / SMA(volume, 20) as proxy for trade count ratio
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        arrival_ratio = volume / avg_vol

        # Arrival direction: sign(close - open)
        arrival_dir = np.sign(close - opn)

        # Price change per unit volume (bps proxy)
        safe_opn = np.where(opn > 1e-10, opn, 1e-10)
        price_per_vol = (close - opn) / safe_opn * 10000.0

        # Confirmation: arrival_ratio > confirm on next bar
        confirmed = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if arrival_ratio[i-1] > arr_thresh and arrival_ratio[i] > confirm_ratio:
                confirmed[i] = True

        long_entry = (confirmed & (arrival_dir > 0) &
                      (price_per_vol > 0.01) & (close > vwap))
        short_entry = (confirmed & (arrival_dir < 0) &
                       (price_per_vol < -0.01) & (close < vwap))

        # Signal exit: arrival ratio drops below 1.0
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if arrival_ratio[i] < 1.0 and arrival_ratio[i-1] >= 1.0:
                sig_exit_long[i] = True
                sig_exit_short[i] = True

        # Skip if recent anomaly (within last 10 bars)
        for i in range(n):
            if long_entry[i] or short_entry[i]:
                start = max(0, i - 10)
                for j in range(start, i):
                    if arrival_ratio[j] > arr_thresh:
                        long_entry[i] = False
                        short_entry[i] = False
                        break

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
            trailing_activate_pct=0.0015,
            time_stop_bars=10,
        )
