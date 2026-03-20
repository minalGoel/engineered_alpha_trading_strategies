"""Sector ETF Spread Reversion — claude_project_22_of_25

Thesis: When a stock's short-term return deviates significantly from
its benchmark index, the spread tends to mean-revert. A z-scored
residual identifies extreme dislocations for reversion trades.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "sector_etf_spread_reversion_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 6

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry_thresh", default=1.8, low=1.0, high=3.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.01),
            TunableParam("vix_max", default=24.0, low=18.0, high=30.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zscore_thresh = params.get("zscore_entry_thresh", 1.8)
        stop_pct = params.get("stop_loss_pct", 0.005)
        vix_max = params.get("vix_max", 24.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)

        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── 20-bar returns for stock and index ──
        ret_period = 20
        stock_ret = np.zeros(n, dtype=np.float64)
        index_ret = np.zeros(n, dtype=np.float64)
        for i in range(ret_period, n):
            if close[i - ret_period] > 0:
                stock_ret[i] = (close[i] - close[i - ret_period]) / close[i - ret_period]
            if index_close[i - ret_period] > 0:
                index_ret[i] = (index_close[i] - index_close[i - ret_period]) / index_close[i - ret_period]

        # ── Residual = stock return - index return ──
        residual = stock_ret - index_ret

        # ── Z-score of residual over 50 bars ──
        zscore_window = 50
        residual_zscore = np.zeros(n, dtype=np.float64)
        for i in range(zscore_window, n):
            window = residual[i - zscore_window:i + 1]
            mu = np.mean(window)
            sigma = np.std(window)
            if sigma > 1e-10:
                residual_zscore[i] = (residual[i] - mu) / sigma

        # ── 2-bar persistence: current AND previous bar both exceed threshold ──
        persist_long = np.zeros(n, dtype=np.bool_)
        persist_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            persist_long[i] = (residual_zscore[i] < -zscore_thresh) & (residual_zscore[i - 1] < -zscore_thresh)
            persist_short[i] = (residual_zscore[i] > zscore_thresh) & (residual_zscore[i - 1] > zscore_thresh)

        # ── VIX filter ──
        vix_ok = vix < vix_max

        # ── Entry ──
        long_entry = persist_long & vix_ok
        short_entry = persist_short & vix_ok

        # ── Signal exit: zscore crosses 0 ──
        signal_exit_long = residual_zscore > 0.0
        signal_exit_short = residual_zscore < 0.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=120,
        )
