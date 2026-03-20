"""Pairs Plus VIX Overlay — Opus_44

Thesis: Stock-vs-index spread z-score reverts to mean in low
volatility regimes (VIX < 16). Enter on extreme spread divergence,
exit when zscore crosses zero or VIX rises above 18.
"""
import numpy as np
import polars as pl
from pipeline.strategies.base import BaseStrategy, StrategySignals, TunableParam


def _compute_atr(high, low, close, period):
    """Wilder's ATR."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    if n < period + 1:
        return atr
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


class Strategy(BaseStrategy):
    name = "opus_pairs_plus_vix_overlay_v1"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 885     # 14:45
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("zscore_entry", default=2.0, low=1.5, high=3.0),
            TunableParam("vix_entry_max", default=16.0, low=12.0, high=20.0),
            TunableParam("vix_exit_max", default=18.0, low=15.0, high=22.0),
            TunableParam("stop_loss_pct", default=0.005, low=0.003, high=0.008),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        zs_entry = params.get("zscore_entry", 2.0)
        vix_entry = params.get("vix_entry_max", 16.0)
        vix_exit = params.get("vix_exit_max", 18.0)
        stp_pct = params.get("stop_loss_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr = _compute_atr(high, low, close, 20)

        # ── Spread: log(close) - log(index_close), z-scored over 120 bars ──
        safe_close = np.clip(close, 1e-10, None)
        safe_idx = np.clip(index_close, 1e-10, None)
        spread = np.log(safe_close) - np.log(safe_idx)
        spread = np.nan_to_num(spread, nan=0.0)

        # Rolling mean and std of spread (120 bars)
        spread_mean = np.zeros(n, dtype=np.float64)
        spread_std = np.zeros(n, dtype=np.float64)
        lookback = 120
        for i in range(lookback, n):
            window = spread[i - lookback + 1:i + 1]
            spread_mean[i] = np.mean(window)
            s = np.std(window)
            spread_std[i] = s if s > 1e-10 else 1e-10

        zscore = np.zeros(n, dtype=np.float64)
        for i in range(lookback, n):
            zscore[i] = (spread[i] - spread_mean[i]) / spread_std[i]
        zscore = np.nan_to_num(zscore, nan=0.0)

        vix_ok_entry = vix < vix_entry
        vix_high = vix > vix_exit
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # Entries
        long_entry = (zscore < -zs_entry) & vix_ok_entry & time_ok
        short_entry = (zscore > zs_entry) & vix_ok_entry & time_ok

        # Signal exits: zscore crosses 0 OR VIX > exit threshold
        zs_cross_0_up = np.zeros(n, dtype=np.bool_)
        zs_cross_0_down = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if zscore[i - 1] < 0 and zscore[i] >= 0:
                zs_cross_0_up[i] = True
            if zscore[i - 1] > 0 and zscore[i] <= 0:
                zs_cross_0_down[i] = True

        exit_long = zs_cross_0_up | vix_high
        exit_short = zs_cross_0_down | vix_high

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=exit_long,
            signal_exit_short=exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stp_pct,
            time_stop_bars=120,
        )
