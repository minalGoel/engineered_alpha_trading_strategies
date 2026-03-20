"""Implied vs Realized Vol v1 — cursor_opus46max_074

Thesis: When VIX (implied) spikes vs realized vol (VRP elevated), NIFTY
stocks tend to mean-revert. Long on VRP spike + VIX declining.
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


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff = close[i] - close[i-1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    if avg_loss > 1e-10:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 1e-10:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_074"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 1

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vrp_z_long", default=2.0, low=1.5, high=3.0),
            TunableParam("vrp_z_short", default=-1.5, low=-2.5, high=-1.0),
            TunableParam("vix_z_long", default=1.5, low=1.0, high=2.5),
            TunableParam("rsi_low", default=40.0, low=25.0, high=50.0),
            TunableParam("rsi_high", default=70.0, low=60.0, high=80.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vrp_z_long = params.get("vrp_z_long", 2.0)
        vrp_z_short = params.get("vrp_z_short", -1.5)
        vix_z_long = params.get("vix_z_long", 1.5)
        rsi_low = params.get("rsi_low", 40.0)
        rsi_high = params.get("rsi_high", 70.0)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=15.0)

        rsi30 = _compute_rsi(close, 30)

        # Realized vol (annualized from 120-bar returns)
        returns = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                returns[i] = np.log(close[i] / close[i-1])
        rv = np.zeros(n, dtype=np.float64)
        rv_lb = 120
        for i in range(rv_lb, n):
            rv[i] = np.std(returns[i - rv_lb + 1:i + 1]) * np.sqrt(252 * 375) * 100.0

        # VRP = VIX - RV
        vrp = vix - rv

        # VRP z-score (rolling 500 bars ~= multi-day)
        vrp_z = np.zeros(n, dtype=np.float64)
        vrp_lb = 500
        for i in range(vrp_lb, n):
            seg = vrp[i - vrp_lb + 1:i + 1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                vrp_z[i] = (vrp[i] - mu) / std

        # VIX z-score
        vix_z = np.zeros(n, dtype=np.float64)
        for i in range(vrp_lb, n):
            seg = vix[i - vrp_lb + 1:i + 1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                vix_z[i] = (vix[i] - mu) / std

        # VIX declining check (5-bar)
        vix_declining = np.zeros(n, dtype=np.bool_)
        for i in range(5, n):
            if vix[i] < vix[i-5]:
                vix_declining[i] = True

        # VIX rising
        vix_rising = np.zeros(n, dtype=np.bool_)
        for i in range(5, n):
            if vix[i] > vix[i-5]:
                vix_rising[i] = True

        # Entry
        long_entry = ((vrp_z > vrp_z_long) & (vix_z > vix_z_long)
                       & vix_declining & (rsi30 < rsi_low))
        short_entry = ((vrp_z < vrp_z_short) & (vix_z < -1.0)
                        & vix_rising & (rsi30 > rsi_high))

        # Signal exit: VIX makes new session high (long) or VRP normalizes
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        # Track session VIX high
        day_ids = df["day_id"].to_numpy()
        unique_days = np.unique(day_ids)
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            running_vix_high = -np.inf
            for i_idx in idx:
                if vix[i_idx] > running_vix_high:
                    running_vix_high = vix[i_idx]
                    signal_exit_long[i_idx] = True  # VIX new high, bad for longs

        # VRP normalizing for shorts
        signal_exit_short = vrp_z > 0.5

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=0.005,
            trailing_stop_pct=0.0015,
            trailing_activate_pct=0.003,
            time_stop_bars=350,
        )
