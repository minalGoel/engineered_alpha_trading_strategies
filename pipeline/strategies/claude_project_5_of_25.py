"""Relative Strength Index Divergence v1 — claude_project_5_of_25

Thesis: When a stock's return deviates significantly from the index
(residual z-score), it tends to revert. Enter on extreme residual
z-scores with persistence confirmation.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


class Strategy(BaseStrategy):
    name = "relative_strength_index_divergence_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 900     # 15:00
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_thresh", default=2.0, low=1.2, high=3.5),
            TunableParam("lookback", default=30.0, low=15.0, high=60.0),
            TunableParam("persist_bars", default=3.0, low=1.0, high=6.0),
            TunableParam("vix_max", default=24.0, low=16.0, high=32.0),
            TunableParam("stop_loss_pct", default=0.006, low=0.003, high=0.010),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_thresh = params.get("zscore_thresh", 2.0)
        lookback = int(params.get("lookback", 30.0))
        persist = int(params.get("persist_bars", 3.0))
        vix_max = params.get("vix_max", 24.0)
        stop_pct = params.get("stop_loss_pct", 0.006)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        # ── Returns over lookback period ──
        stock_ret = np.zeros(n, dtype=np.float64)
        idx_ret = np.zeros(n, dtype=np.float64)
        for i in range(lookback, n):
            if close[i - lookback] > 0:
                stock_ret[i] = (close[i] - close[i - lookback]) / close[i - lookback]
            if index_close[i - lookback] > 0:
                idx_ret[i] = (index_close[i] - index_close[i - lookback]) / index_close[i - lookback]

        # ── Residual: stock return minus expected (using index as proxy) ──
        residual = stock_ret - idx_ret

        # ── Z-score of residual using rolling std ──
        res_mean = np.zeros(n, dtype=np.float64)
        res_std = np.zeros(n, dtype=np.float64)
        window = lookback
        for i in range(window, n):
            seg = residual[i - window:i]
            res_mean[i] = np.mean(seg)
            res_std[i] = np.std(seg)
        res_std = np.clip(res_std, 1e-10, None)
        zscore = (residual - res_mean) / res_std
        zscore = np.nan_to_num(zscore, nan=0.0)

        # ── Persistence: z-score must stay extreme for N bars ──
        long_persist = np.zeros(n, dtype=np.bool_)
        short_persist = np.zeros(n, dtype=np.bool_)
        for i in range(persist, n):
            if all(zscore[i - j] < -zs_thresh for j in range(persist)):
                long_persist[i] = True
            if all(zscore[i - j] > zs_thresh for j in range(persist)):
                short_persist[i] = True

        # ── VIX filter ──
        vix_ok = vix < vix_max

        # ── Entry ──
        long_entry = long_persist & vix_ok
        short_entry = short_persist & vix_ok

        # ── Signal exit: z-score crosses zero ──
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
            time_stop_bars=120,
        )
