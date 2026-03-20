"""IV-RV Dislocation — Opus_28

Thesis: When implied volatility (VIX proxy) is significantly higher than
realized volatility, options are overpriced and the underlying tends to
rally (vol crush). When IV < RV, fear is underpriced and downside follows.
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
    name = "opus_implied_realized_vol_v1"
    is_long_only = False
    session_start = 600   # 10:00
    session_end = 870     # 14:30
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("iv_rv_long_thresh", default=1.3, low=1.1, high=1.6),
            TunableParam("iv_rv_short_thresh", default=0.8, low=0.6, high=0.95),
            TunableParam("index_ret_thresh", default=0.003, low=0.001, high=0.006),
            TunableParam("target_pct", default=0.005, low=0.003, high=0.008),
            TunableParam("stop_loss_pct", default=0.0035, low=0.002, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        iv_rv_long = params.get("iv_rv_long_thresh", 1.3)
        iv_rv_short = params.get("iv_rv_short_thresh", 0.8)
        index_ret_thresh = params.get("index_ret_thresh", 0.003)
        target_pct = params.get("target_pct", 0.005)
        stop_pct = params.get("stop_loss_pct", 0.0035)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        index_close = df["index_close"].to_numpy().astype(np.float64)
        index_close = np.nan_to_num(index_close, nan=0.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # ── Realized vol: 20-bar rolling std of close-to-close returns, annualized ──
        returns = np.zeros(n, dtype=np.float64)
        returns[1:] = (close[1:] - close[:-1]) / np.clip(close[:-1], 1e-10, None)

        rv_20 = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            window = returns[i - 19:i + 1]
            rv_20[i] = np.std(window) * np.sqrt(252 * 78)  # annualize

        rv_20 = np.clip(rv_20, 1e-10, None)

        # ── IV/RV ratio ──
        iv_rv_ratio = vix / rv_20
        iv_rv_ratio = np.nan_to_num(iv_rv_ratio, nan=1.0)
        iv_rv_ratio = np.clip(iv_rv_ratio, 0.01, 100.0)

        # ── Index 30-bar return ──
        index_ret_30 = np.zeros(n, dtype=np.float64)
        for i in range(30, n):
            if abs(index_close[i - 30]) > 1e-10:
                index_ret_30[i] = (index_close[i] - index_close[i - 30]) / index_close[i - 30]

        # ── VIX declining 3 bars ──
        vix_declining = np.zeros(n, dtype=np.bool_)
        for i in range(3, n):
            if vix[i] < vix[i - 1] and vix[i - 1] < vix[i - 2] and vix[i - 2] < vix[i - 3]:
                vix_declining[i] = True

        # ── ATR(20) ──
        atr = _compute_atr(high, low, close, 20)

        # ── Filters ──
        time_ok = (time_mins >= self.session_start) & (time_mins <= self.session_end)

        # ── Entries ──
        long_entry = (
            (iv_rv_ratio > iv_rv_long)
            & (index_ret_30 > -index_ret_thresh)
            & vix_declining
            & time_ok
        )
        short_entry = (
            (iv_rv_ratio < iv_rv_short)
            & (index_ret_30 < index_ret_thresh)
            & time_ok
        )

        # ── Signal exit: iv_rv_ratio returns to 0.9-1.1 ──
        sig_exit = (iv_rv_ratio >= 0.9) & (iv_rv_ratio <= 1.1)
        sig_exit[:20] = False

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit,
            signal_exit_short=sig_exit,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            target_pct=target_pct,
            stop_loss_pct=stop_pct,
            time_stop_bars=200,
        )
