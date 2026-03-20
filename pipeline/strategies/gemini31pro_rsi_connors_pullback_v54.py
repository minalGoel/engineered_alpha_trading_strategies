"""RSI Connors Pullback — cursor_gemini31pro_strategy_054

Thesis: A short-term extreme oversold reading (RSI-2 < threshold) within a
longer-term uptrend (close > SMA-200) indicates temporary liquidity exhaustion.
Counter-party market makers step in to provide liquidity, bouncing the price.
Exit when close crosses above SMA(5). Volume surge confirmation required.
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
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    if n < period:
        return atr
    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def _compute_rsi(close, period):
    """Wilder's smoothed RSI."""
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss < 1e-10:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss < 1e-10:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


class Strategy(BaseStrategy):
    name = "gemini31pro_rsi_connors_pullback_v54"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("rsi_long_thresh", default=50.0, low=20.0, high=60.0),
            TunableParam("rsi_short_thresh", default=20.0, low=10.0, high=50.0),
            TunableParam("vix_max", default=25.0, low=15.0, high=35.0),
            TunableParam("stop_atr_mult", default=1.5, low=0.8, high=3.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        rsi_long = params.get("rsi_long_thresh", 50.0)
        rsi_short = params.get("rsi_short_thresh", 20.0)
        vix_max = params.get("vix_max", 25.0)
        stop_atr = params.get("stop_atr_mult", 1.5)
        be_pct = params.get("breakeven_pct", 0.005)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(df["volume"].to_numpy().astype(np.float64), nan=1.0)
        vix = np.nan_to_num(df["vix"].to_numpy().astype(np.float64), nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr14 = _compute_atr(high, low, close, 14)
        rsi2 = _compute_rsi(close, 2)

        # SMA(200) for macro trend
        sma200 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - 199)
            sma200[i] = np.mean(close[start:i + 1])

        # SMA(5) for exit target
        sma5 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - 4)
            sma5[i] = np.mean(close[start:i + 1])

        # Volume SMA(20)
        vol_sma20 = np.full(n, 1.0, dtype=np.float64)
        for i in range(19, n):
            vol_sma20[i] = np.mean(volume[i - 19:i + 1])
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        vol_ok = volume > vol_sma20

        # Filters
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 570) & (time_mins <= 870)  # 09:30-14:30

        # Entry conditions
        long_entry = (close > sma200) & (rsi2 < rsi_long) & vol_ok & vix_ok & time_ok
        short_entry = (close < sma200) & (rsi2 > rsi_short) & vol_ok & vix_ok & time_ok

        # Signal exit: close crosses SMA(5) — long exits when close > SMA5, short when close < SMA5
        signal_exit_long = close > sma5
        signal_exit_short = close < sma5

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=sma5,
            use_target_indicator=True,
            stop_loss_atr_mult=stop_atr,
            breakeven_pct=be_pct,
            time_stop_bars=30,
        )
