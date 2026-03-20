"""Bollinger Squeeze Breakout — cursor_gemini31pro_strategy_082

Thesis: Volatility is cyclical; periods of low volatility (squeeze — BB inside
Keltner Channels) are followed by explosive expansion. Breakouts from the squeeze
predict the direction of the expansion. Exit when close crosses SMA(20).
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


def _ema(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1.0 - alpha) * out[i - 1]
    return out


class Strategy(BaseStrategy):
    name = "gemini31pro_bollinger_squeeze_breakout_v82"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("keltner_mult", default=1.5, low=1.0, high=2.5),
            TunableParam("bb_mult", default=2.0, low=1.5, high=3.0),
            TunableParam("vix_max", default=25.0, low=15.0, high=35.0),
            TunableParam("stop_atr_mult", default=1.5, low=0.8, high=3.0),
            TunableParam("breakeven_pct", default=0.005, low=0.002, high=0.01),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        kelt_mult = params.get("keltner_mult", 1.5)
        bb_mult = params.get("bb_mult", 2.0)
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
        atr20 = _compute_atr(high, low, close, 20)

        # Bollinger Bands: SMA(20) +/- bb_mult * std(20)
        sma20 = np.zeros(n, dtype=np.float64)
        std20 = np.zeros(n, dtype=np.float64)
        for i in range(n):
            start = max(0, i - 19)
            window = close[start:i + 1]
            sma20[i] = np.mean(window)
            std20[i] = np.std(window) if len(window) > 1 else 0.0

        bb_upper = sma20 + bb_mult * std20
        bb_lower = sma20 - bb_mult * std20

        # Keltner Channels: EMA(20) +/- kelt_mult * ATR(20)
        ema20 = _ema(close, 20)
        kelt_upper = ema20 + kelt_mult * atr20
        kelt_lower = ema20 - kelt_mult * atr20

        # Squeeze: BB inside Keltner
        squeeze = (bb_upper < kelt_upper) & (bb_lower > kelt_lower)

        # Volume confirmation: volume > SMA(volume, 20) — surge
        vol_sma20 = np.full(n, 1.0, dtype=np.float64)
        for i in range(19, n):
            vol_sma20[i] = np.mean(volume[i - 19:i + 1])
        vol_sma20 = np.clip(vol_sma20, 1.0, None)
        vol_ok = volume > vol_sma20

        # Filters
        vix_ok = vix < vix_max
        time_ok = (time_mins >= 570) & (time_mins <= 870)  # 09:30-14:30

        # Entry: breakout from squeeze
        long_entry = squeeze & (close > bb_upper) & vol_ok & vix_ok & time_ok
        short_entry = squeeze & (close < bb_lower) & vol_ok & vix_ok & time_ok

        # Signal exit: close crosses SMA(20)
        signal_exit_long = close < sma20
        signal_exit_short = close > sma20

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=sma20,
            use_target_indicator=True,
            stop_loss_atr_mult=stop_atr,
            breakeven_pct=be_pct,
            time_stop_bars=30,
        )
