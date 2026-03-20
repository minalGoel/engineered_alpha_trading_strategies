"""Index Momentum — GPT_10_of_10

Thesis: When the Nifty50 index has strong 5-bar return (>0.2%), individual stocks
tend to follow due to market beta. Captures broad-based momentum from institutional
flows and macro announcements.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "index_momentum_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("return_thresh", default=0.2, low=0.05, high=0.5),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.008),
            TunableParam("target_pct", default=0.003, low=0.001, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ret_thresh = params.get("return_thresh", 0.2)
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_pct = params.get("target_pct", 0.003)

        n = len(df)

        # ── Index 5-bar return (%) ──
        idx_close = df["index_close"].to_numpy().astype(np.float64)
        idx_close = np.nan_to_num(idx_close, nan=0.0)

        idx_ret_5 = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            if idx_close[i - 5] > 0:
                idx_ret_5[i] = (idx_close[i] - idx_close[i - 5]) / idx_close[i - 5] * 100.0

        # ── Entry ──
        long_entry = idx_ret_5 > ret_thresh
        short_entry = idx_ret_5 < -ret_thresh

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            time_stop_bars=10,
        )
