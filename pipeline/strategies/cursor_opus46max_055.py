# AUDIT FIX: time_ok missing lower bound (session_start=560) — entries fired before session start
# AUDIT FIX: signal_exit_short = pd_bps < 0 was always True (stock price always < index) — replaced with z-score crossing
"""ETF NAV Arbitrage v1 — cursor_opus46max_055

Thesis: ETF premium/discount to indicative NAV mean-reverts as APs step in.
Adapted: stock premium/discount vs index as NAV proxy, z-score entry.
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
    name = "cursor_opus46max_055"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 925     # 15:25
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("pd_thresh_bps", default=15.0, low=8.0, high=30.0),
            TunableParam("pd_z_thresh", default=1.5, low=1.0, high=2.5),
            TunableParam("pd_lookback", default=60.0, low=30.0, high=120.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        pd_thresh = params.get("pd_thresh_bps", 15.0)
        pd_z_thresh = params.get("pd_z_thresh", 1.5)
        lb = int(params.get("pd_lookback", 60.0))
        stop_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        volume = df["volume"].to_numpy().astype(np.float64)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Premium/discount in bps
        pd_bps = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if index_close[i] > 1e-8:
                pd_bps[i] = (close[i] - index_close[i]) / index_close[i] * 10000.0

        # Z-score of premium/discount
        pd_z = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            seg = pd_bps[i - lb:i + 1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                pd_z[i] = (pd_bps[i] - mu) / std

        # Volume ratio
        vol_sma = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            vol_sma[i] = np.mean(volume[i-29:i+1])
        vol_sma = np.clip(vol_sma, 1.0, None)
        vol_ratio = volume / vol_sma

        time_ok = (time_mins >= 560) & (time_mins <= 900)  # AUDIT FIX: added lower bound >= 560 (session_start)

        # Entry: discount (buy) or premium (sell) with narrowing confirmation
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (pd_bps[i] < -pd_thresh and pd_z[i] < -pd_z_thresh
                    and vol_ratio[i] > 0.5 and pd_bps[i] > pd_bps[i-1] - 2.0
                    and time_ok[i]):
                long_entry[i] = True
            if (pd_bps[i] > pd_thresh and pd_z[i] > pd_z_thresh
                    and vol_ratio[i] > 0.5 and pd_bps[i] < pd_bps[i-1] + 2.0
                    and time_ok[i]):
                short_entry[i] = True

        # Exit: use z-score crossing zero (avoids always-true condition when stock << index)
        # AUDIT FIX: pd_bps < 0 was always True for stocks vs large-cap index; use z-score reversion instead
        signal_exit_long = pd_z > 0
        signal_exit_short = pd_z < 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=45,
        )
