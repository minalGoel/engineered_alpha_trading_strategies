# AUDIT FIX: Add session window filtering to prevent entries outside session_start/session_end
"""NIFTY-BANKNIFTY Spread v1 — cursor_opus46max_117

Thesis: BANK NIFTY carries ~33% weight in NIFTY 50.  The intraday return
spread between BANK NIFTY and NIFTY is mean-reverting.  Trade spread
compression using stock-vs-index return divergence as proxy.
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
    name = "cursor_opus46max_117"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 920     # 15:20
    max_trades_per_day = 4
    assumptions = ["NIFTY/BANKNIFTY spread proxied using stock-vs-index return z-score"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=1.5, low=1.0, high=2.5),
            TunableParam("velocity_thresh", default=3.0, low=1.0, high=5.0),
            TunableParam("vix_max", default=20.0, low=14.0, high=25.0),
            TunableParam("stop_loss_pct", default=0.0015, low=0.0008, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 1.5)
        vel_thresh = params.get("velocity_thresh", 3.0)
        vix_max = params.get("vix_max", 20.0)
        stop_pct = params.get("stop_loss_pct", 0.0015)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        day_id = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        in_session = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Cumulative returns
        stock_ret = np.zeros(n, dtype=np.float64)
        index_ret = np.zeros(n, dtype=np.float64)
        day_open_s = close[0]
        day_open_i = index_close[0]
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                day_open_s = close[i]
                day_open_i = index_close[i] if index_close[i] > 1e-10 else 1.0
            if day_open_s > 1e-10:
                stock_ret[i] = (close[i] - day_open_s) / day_open_s * 10000.0
            if day_open_i > 1e-10:
                index_ret[i] = (index_close[i] - day_open_i) / day_open_i * 10000.0

        # NB spread proxy
        nb_spread = stock_ret - index_ret

        # Z-score over 45 bars
        lookback = 45
        nb_zs = np.zeros(n, dtype=np.float64)
        for i in range(lookback - 1, n):
            window = nb_spread[i - lookback + 1: i + 1]
            mu = np.mean(window)
            std = np.std(window)
            if std > 1e-10:
                nb_zs[i] = (nb_spread[i] - mu) / std

        # Spread velocity: spread change over 5 bars
        velocity = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            velocity[i] = nb_spread[i] - nb_spread[i-5]

        vix_ok = vix < vix_max

        # Long: z-score < -thresh and velocity slowing (crossing above -1)
        # Short: z-score > thresh and velocity slowing
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (nb_zs[i] < -zs_thresh and velocity[i] > -1 and velocity[i-1] <= -1 and
                    vix_ok[i] and in_session[i]):
                long_entry[i] = True
            if (nb_zs[i] > zs_thresh and velocity[i] < 1 and velocity[i-1] >= 1 and
                    vix_ok[i] and in_session[i]):
                short_entry[i] = True

        # Signal exit: z-score reverts to 0 or VIX spikes
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if nb_zs[i] >= 0 and nb_zs[i-1] < 0:
                sig_exit_long[i] = True
            if nb_zs[i] <= 0 and nb_zs[i-1] > 0:
                sig_exit_short[i] = True
            if vix[i] > 22:
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
            time_stop_bars=60,
        )
