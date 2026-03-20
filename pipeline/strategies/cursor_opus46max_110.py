"""Spread Dynamics v1 — cursor_opus46max_110

Thesis: Bid-ask spread expansion precedes short-term volatility.  After
the event, fade the move when spreads contract.  Approximate spread using
bar range relative to typical range.
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
    name = "cursor_opus46max_110"
    is_long_only = False
    session_start = 580   # 09:40
    session_end = 920     # 15:20
    max_trades_per_day = 8
    assumptions = ["Bid-ask spread proxied via bar range / average bar range"]

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("expansion_thresh", default=2.0, low=1.5, high=3.0),
            TunableParam("contraction_thresh", default=1.8, low=1.3, high=2.5),
            TunableParam("post_event_bps", default=10.0, low=5.0, high=20.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.0015, low=0.0008, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        exp_thresh = params.get("expansion_thresh", 2.0)
        cont_thresh = params.get("contraction_thresh", 1.8)
        post_bps = params.get("post_event_bps", 10.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.0015)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        volume = np.clip(volume, 1.0, None)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)

        # Spread proxy: bar range as fraction of close
        bar_range = (high - low)
        safe_close = np.where(close > 1e-10, close, 1e-10)
        spread_bps = bar_range / safe_close * 10000.0

        # SMA of spread over 30 bars
        spread_sma = np.ones(n, dtype=np.float64)
        for i in range(29, n):
            spread_sma[i] = np.mean(spread_bps[i-29:i+1])
        spread_sma = np.clip(spread_sma, 0.01, None)
        spread_ratio = spread_bps / spread_sma

        # Detect spread contraction from expansion
        spread_contracting = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if spread_ratio[i] < spread_ratio[i-1] and spread_ratio[i-1] > exp_thresh:
                spread_contracting[i] = True

        # Post event return: track close at spread peak
        close_at_peak = np.zeros(n, dtype=np.float64)
        in_event = False
        peak_close = 0.0
        for i in range(n):
            if spread_ratio[i] > exp_thresh:
                in_event = True
                peak_close = close[i]
            if in_event:
                close_at_peak[i] = peak_close
            if spread_ratio[i] < 1.2:
                in_event = False

        post_event_ret = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if close_at_peak[i] > 1e-10:
                post_event_ret[i] = (close[i] - close_at_peak[i]) / close_at_peak[i] * 10000.0

        vix_ok = vix < vix_max

        # Volume confirmation: volume > 2x normal during event
        avg_vol = np.ones(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > 1.5 * avg_vol

        # Long: spread contracting, price dropped, close > low of event
        long_entry = (spread_contracting & (spread_ratio < cont_thresh) &
                      (post_event_ret < -post_bps) & (close > low) & vix_ok & vol_ok)
        # Short: spread contracting, price rose
        short_entry = (spread_contracting & (spread_ratio < cont_thresh) &
                       (post_event_ret > post_bps) & (close < high) & vix_ok & vol_ok)

        # Signal exit: spread re-expands
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if spread_ratio[i] > exp_thresh:
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
            trailing_stop_pct=0.0005,
            trailing_activate_pct=0.001,
            time_stop_bars=5,
        )
