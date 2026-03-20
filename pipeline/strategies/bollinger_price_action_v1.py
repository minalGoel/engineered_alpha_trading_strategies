"""Bollinger Band + Price Action — Grok_5_of_10

Thesis: Price action rejection (pinbar) at Bollinger Band extremes in
the direction of the broader market trend. More selective than simple BB touch.
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
    name = "bollinger_price_action_v1"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 915     # 15:15
    max_trades_per_day = 12

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("vix_max", default=18.0, low=12.0, high=28.0),
            TunableParam("rel_vol_thresh", default=1.2, low=1.0, high=2.5),
            TunableParam("stop_atr_mult", default=1.0, low=0.5, high=2.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        vix_max = params.get("vix_max", 18.0)
        rv_thresh = params.get("rel_vol_thresh", 1.2)
        stop_atr = params.get("stop_atr_mult", 1.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        idx = np.nan_to_num(df["index_close"].to_numpy().astype(np.float64), nan=0.0)

        atr14 = _compute_atr(high, low, close, 14)

        # BB(20, 2)
        sma20 = df["close"].rolling_mean(20).to_numpy().astype(np.float64)
        std20 = df["close"].rolling_std(20).to_numpy().astype(np.float64)
        sma20 = np.nan_to_num(sma20, nan=0.0)
        std20 = np.nan_to_num(std20, nan=1e10)
        bb_upper = sma20 + 2.0 * std20
        bb_lower = sma20 - 2.0 * std20

        # Relative volume
        avg_vol_20 = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol_20 = np.nan_to_num(avg_vol_20, nan=1.0)
        avg_vol_20 = np.clip(avg_vol_20, 1.0, None)
        rel_vol = volume / avg_vol_20

        # Index 5-bar return
        idx_ret5 = np.zeros(n, dtype=np.float64)
        for i in range(5, n):
            if idx[i-5] > 0:
                idx_ret5[i] = (idx[i] - idx[i-5]) / idx[i-5]

        # Pinbar detection
        body = np.abs(close - open_)
        body_safe = np.clip(body, 0.01, None)
        lower_wick = np.minimum(close, open_) - low
        upper_wick = high - np.maximum(close, open_)
        bullish_pin = lower_wick > 2.0 * body_safe
        bearish_pin = upper_wick > 2.0 * body_safe

        # Filters
        vix_ok = vix < vix_max
        vol_ok = rel_vol > rv_thresh

        # Entry
        long_entry = (close <= bb_lower) & bullish_pin & vix_ok & vol_ok & (idx_ret5 > 0)
        short_entry = (close >= bb_upper) & bearish_pin & vix_ok & vol_ok & (idx_ret5 < 0)

        # Target: middle band
        target_indicator = sma20.copy()

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr14,
            target_indicator=target_indicator,
            stop_loss_atr_mult=stop_atr,
            use_target_indicator=True,
            time_stop_bars=20,
        )
