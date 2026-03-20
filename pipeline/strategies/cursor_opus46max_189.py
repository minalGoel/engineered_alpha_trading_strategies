"""Copula Dependence v1 — cursor_opus46max_189

Thesis: Tail dependence between stock and index differs from normal correlation.
When one crashes but the other hasn't responded despite historically high tail
dependence, trade the convergence. Adapted: use empirical tail dependence
between stock and index_close returns.
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
    name = "cursor_opus46max_189"
    is_long_only = False
    session_start = 675   # ~11:15
    session_end = 910     # 15:10
    max_trades_per_day = 4

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("copula_window", default=120.0, low=80.0, high=180.0),
            TunableParam("tail_thresh", default=0.10, low=0.05, high=0.20),
            TunableParam("divergence_thresh", default=0.30, low=0.15, high=0.50),
            TunableParam("cond_prob_min", default=0.40, low=0.25, high=0.60),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        cop_win = int(params.get("copula_window", 120.0))
        tail_th = params.get("tail_thresh", 0.10)
        div_th = params.get("divergence_thresh", 0.30)
        cond_prob_min = params.get("cond_prob_min", 0.40)
        stop_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # Returns
        stock_ret = np.zeros(n, dtype=np.float64)
        index_ret = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i-1] > 0:
                stock_ret[i] = (close[i] - close[i-1]) / close[i-1]
            if index_close[i-1] > 0:
                index_ret[i] = (index_close[i] - index_close[i-1]) / index_close[i-1]

        # Empirical CDF (uniform marginals)
        u_stock = np.zeros(n, dtype=np.float64)
        u_index = np.zeros(n, dtype=np.float64)
        # Empirical conditional tail probability
        cond_prob_lower = np.zeros(n, dtype=np.float64)
        cond_prob_upper = np.zeros(n, dtype=np.float64)

        for i in range(cop_win, n):
            sr = stock_ret[i-cop_win+1:i+1]
            ir = index_ret[i-cop_win+1:i+1]

            # Empirical CDF for current values
            u_stock[i] = np.mean(sr <= stock_ret[i])
            u_index[i] = np.mean(ir <= index_ret[i])

            # Conditional tail probability
            # P(stock in bottom tail | index in bottom tail)
            idx_low = ir <= np.percentile(ir, tail_th * 100)
            if np.sum(idx_low) > 0:
                stock_low = sr[idx_low] <= np.percentile(sr, tail_th * 100)
                cond_prob_lower[i] = np.mean(stock_low)

            # P(stock in top tail | index in top tail)
            idx_high = ir >= np.percentile(ir, (1 - tail_th) * 100)
            if np.sum(idx_high) > 0:
                stock_high = sr[idx_high] >= np.percentile(sr, (1 - tail_th) * 100)
                cond_prob_upper[i] = np.mean(stock_high)

        # Divergence: one in tail, other not
        # Index crashed but stock hasn't
        idx_crashed = u_index < tail_th
        stock_ok = u_stock > div_th
        # Index rallied but stock hasn't
        idx_rallied = u_index > (1.0 - tail_th)
        stock_lagging = u_stock < (1.0 - div_th)

        # ROC for confirmation
        roc3 = np.zeros(n, dtype=np.float64)
        for i in range(3, n):
            if close[i-3] > 0:
                roc3[i] = (close[i] - close[i-3]) / close[i-3] * 10000

        time_ok = (time_mins >= 675) & (time_mins <= 910)

        # Long: index crashed, stock hasn't yet, high tail dependence => buy stock dip
        long_entry = (
            idx_crashed & stock_ok
            & (cond_prob_lower > cond_prob_min)
            & (roc3 > 0)  # stock starting to catch up
            & time_ok
        )
        # Short: index rallied, stock hasn't caught up => short stock expecting it to lag
        # Actually: short when stock rallied alone without index
        short_entry = (
            idx_rallied & stock_lagging
            & (cond_prob_upper > cond_prob_min)
            & (roc3 < 0)
            & time_ok
        )

        # Signal exit: divergence resolves
        sig_exit_long = np.abs(u_stock - u_index) < 0.20
        sig_exit_short = np.abs(u_stock - u_index) < 0.20

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
