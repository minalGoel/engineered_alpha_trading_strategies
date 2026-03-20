"""BankNifty Lead-Lag v1 — claude_project_8_of_25

Thesis: The index sometimes leads individual stocks. When the index moves
but the stock lags, the stock tends to catch up. Scalp the convergence.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "banknifty_lead_lag_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 12

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("idx_ret_thresh", default=0.0015, low=0.0008, high=0.003),
            TunableParam("lag_ratio", default=0.5, low=0.2, high=0.8),
            TunableParam("target_pct", default=0.003, low=0.001, high=0.006),
            TunableParam("stop_loss_pct", default=0.0025, low=0.001, high=0.005),
            TunableParam("vix_max", default=24.0, low=16.0, high=32.0),
            TunableParam("ret_lookback", default=5.0, low=3.0, high=10.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        idx_ret_thresh = params.get("idx_ret_thresh", 0.0015)
        lag_ratio = params.get("lag_ratio", 0.5)
        target_pct = params.get("target_pct", 0.003)
        stop_pct = params.get("stop_loss_pct", 0.0025)
        vix_max = params.get("vix_max", 24.0)
        lb = int(params.get("ret_lookback", 5.0))

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── Returns over lookback ──
        stock_ret = np.zeros(n, dtype=np.float64)
        idx_ret = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            if close[i - lb] > 0:
                stock_ret[i] = (close[i] - close[i - lb]) / close[i - lb]
            if index_close[i - lb] > 0:
                idx_ret[i] = (index_close[i] - index_close[i - lb]) / index_close[i - lb]

        # ── VIX filter ──
        vix_ok = vix < vix_max

        # ── Long: index up significantly, stock lagging ──
        long_entry = (
            (idx_ret > idx_ret_thresh)
            & (stock_ret < idx_ret * lag_ratio)
            & vix_ok
        )
        # ── Short: index down significantly, stock lagging on downside ──
        short_entry = (
            (idx_ret < -idx_ret_thresh)
            & (stock_ret > idx_ret * lag_ratio)
            & vix_ok
        )

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=30,
        )
