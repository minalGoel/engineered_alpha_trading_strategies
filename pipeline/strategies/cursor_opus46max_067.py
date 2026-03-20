"""Gap Relative Index v1 — cursor_opus46max_067

Thesis: Excess gap (stock gap minus beta-adjusted index gap) mean-reverts.
Enter on 3-bar relative return uptick/downtick.
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


def _compute_rsi(data, period):
    n = len(data)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff = data[i] - data[i-1]
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
    name = "cursor_opus46max_067"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 870     # 14:30
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("excess_gap_min", default=0.005, low=0.003, high=0.01),
            TunableParam("rel_rsi_low", default=30.0, low=15.0, high=40.0),
            TunableParam("rel_rsi_high", default=70.0, low=60.0, high=85.0),
            TunableParam("vix_max", default=22.0, low=16.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.007),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        excess_gap_min = params.get("excess_gap_min", 0.005)
        rsi_low = params.get("rel_rsi_low", 30.0)
        rsi_high = params.get("rel_rsi_high", 70.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Returns
        stock_ret = np.zeros(n, dtype=np.float64)
        idx_ret = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                stock_ret[i] = (close[i] - close[i-1]) / close[i-1]
            if index_close[i-1] > 0:
                idx_ret[i] = (index_close[i] - index_close[i-1]) / index_close[i-1]

        # Relative return cumulative from session open
        rel_ret = stock_ret - idx_ret
        rel_ret_cum = np.cumsum(rel_ret)

        # RSI of relative return
        rel_rsi = _compute_rsi(rel_ret_cum, 14)

        # Excess gap per day
        excess_gap = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        prev_close_s = np.nan
        prev_close_i = np.nan
        for d in unique_days:
            didx = np.where(day_ids == d)[0]
            if len(didx) == 0:
                continue
            if not np.isnan(prev_close_s) and prev_close_s > 0 and prev_close_i > 0:
                s_gap = (open_[didx[0]] - prev_close_s) / prev_close_s
                i_gap = (index_close[didx[0]] - prev_close_i) / prev_close_i if index_close[didx[0]] > 0 else 0
                excess_gap[didx] = s_gap - i_gap
            prev_close_s = close[didx[-1]]
            prev_close_i = index_close[didx[-1]]

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 565) & (time_mins <= 630)

        # 3-bar relative return uptick/downtick
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(3, n):
            if (excess_gap[i] < -excess_gap_min and rel_rsi[i] < rsi_low
                    and vix_ok[i] and time_ok[i]
                    and rel_ret_cum[i] > rel_ret_cum[i-1]
                    and rel_ret_cum[i-1] > rel_ret_cum[i-2]
                    and rel_ret_cum[i-2] > rel_ret_cum[i-3]):
                long_entry[i] = True
            if (excess_gap[i] > excess_gap_min and rel_rsi[i] > rsi_high
                    and vix_ok[i] and time_ok[i]
                    and rel_ret_cum[i] < rel_ret_cum[i-1]
                    and rel_ret_cum[i-1] < rel_ret_cum[i-2]
                    and rel_ret_cum[i-2] < rel_ret_cum[i-3]):
                short_entry[i] = True

        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=0.005,
            time_stop_bars=90,
        )
