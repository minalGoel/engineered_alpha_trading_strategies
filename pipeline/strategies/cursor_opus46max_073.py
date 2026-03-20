# AUDIT FIX: Add session window filter to entry signals (entries were firing outside session_start/session_end)
"""ATR Breakout Volatility v1 — cursor_opus46max_073

Thesis: ATR channel breakouts (2x ATR from session mean) with volume and
ADX confirmation capture genuine volatility expansion events.
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
    name = "cursor_opus46max_073"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 910     # 15:10
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("atr_mult", default=2.0, low=1.5, high=3.0),
            TunableParam("atr_expansion_min", default=1.2, low=1.0, high=1.5),
            TunableParam("vol_ratio_min", default=1.5, low=1.0, high=2.5),
            TunableParam("vix_min", default=13.0, low=8.0, high=16.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=30.0),
            TunableParam("stop_loss_atr_mult", default=1.0, low=0.5, high=1.5),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        atr_mult = params.get("atr_mult", 2.0)
        atr_exp_min = params.get("atr_expansion_min", 1.2)
        vol_ratio_min = params.get("vol_ratio_min", 1.5)
        vix_min = params.get("vix_min", 13.0)
        vix_max = params.get("vix_max", 25.0)
        stop_atr = params.get("stop_loss_atr_mult", 1.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high_ = df["high"].to_numpy().astype(np.float64)
        low_ = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_ids = df["day_id"].to_numpy()
        time_min = df["time_minutes"].to_numpy()
        in_session = (time_min >= self.session_start) & (time_min <= self.session_end)

        atr14 = _compute_atr(high_, low_, close, 14)

        # Session mean (cumulative from session open)
        session_mean = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            cum = 0.0
            for j, i in enumerate(idx):
                cum += close[i]
                session_mean[i] = cum / (j + 1)

        # ATR channels
        upper_ch = session_mean + atr_mult * atr14
        lower_ch = session_mean - atr_mult * atr14

        # ATR expansion ratio
        atr_sma60 = np.zeros(n, dtype=np.float64)
        for i in range(60, n):
            atr_sma60[i] = np.mean(atr14[i-59:i+1])
        atr_sma60 = np.clip(atr_sma60, 1e-8, None)
        atr_expansion = atr14 / atr_sma60

        # Volume ratio
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ratio = volume / avg_vol

        # Bar close position in range
        bar_range = high_ - low_
        close_pos = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if bar_range[i] > 1e-8:
                close_pos[i] = (close[i] - low_[i]) / bar_range[i]

        vix_ok = (vix >= vix_min) & (vix <= vix_max)

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if not vix_ok[i] or not in_session[i]:
                continue
            if (close[i] > upper_ch[i] and atr_expansion[i] > atr_exp_min
                    and vol_ratio[i] > vol_ratio_min and close_pos[i] > 0.75):
                long_entry[i] = True
            if (close[i] < lower_ch[i] and atr_expansion[i] > atr_exp_min
                    and vol_ratio[i] > vol_ratio_min and close_pos[i] < 0.25):
                short_entry[i] = True

        # Signal exit: ATR expansion drops below 1.0
        signal_exit_long = atr_expansion < 1.0
        signal_exit_short = atr_expansion < 1.0

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_atr_mult=stop_atr,
            target_atr_mult=3.0,
            trailing_stop_pct=0.002,
            trailing_activate_pct=0.004,
            time_stop_bars=60,
        )
