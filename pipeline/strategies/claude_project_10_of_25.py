"""Sector Pair Reversion v1 — claude_project_10_of_25

Thesis: When a stock diverges significantly from its index (sector proxy),
the spread tends to revert. Enter on extreme z-score of the return spread.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "sector_pair_reversion_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=2.0, low=1.2, high=3.5),
            TunableParam("spread_lookback", default=30.0, low=15.0, high=60.0),
            TunableParam("vix_max", default=24.0, low=16.0, high=32.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.002, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 2.0)
        lb = int(params.get("spread_lookback", 30.0))
        vix_max = params.get("vix_max", 24.0)
        stop_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── 1-bar returns ──
        stock_ret = np.zeros(n, dtype=np.float64)
        idx_ret = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close[i - 1] > 0:
                stock_ret[i] = (close[i] - close[i - 1]) / close[i - 1]
            if index_close[i - 1] > 0:
                idx_ret[i] = (index_close[i] - index_close[i - 1]) / index_close[i - 1]

        # ── Spread: stock return - index return ──
        spread = stock_ret - idx_ret

        # ── Cumulative spread and z-score ──
        cum_spread = np.cumsum(spread)
        zscore = np.zeros(n, dtype=np.float64)
        for i in range(lb, n):
            seg = cum_spread[i - lb:i + 1]
            mu = np.mean(seg)
            std = np.std(seg)
            if std > 1e-10:
                zscore[i] = (cum_spread[i] - mu) / std
        zscore = np.nan_to_num(zscore, nan=0.0)

        # ── VIX filter ──
        vix_ok = vix < vix_max

        # ── Entry ──
        long_entry = (zscore < -zs_thresh) & vix_ok
        short_entry = (zscore > zs_thresh) & vix_ok

        # ── Signal exit: zscore crosses zero ──
        signal_exit_long = zscore > 0
        signal_exit_short = zscore < 0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=np.zeros(n, dtype=np.float64),
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            time_stop_bars=180,
        )
