"""Bollinger Squeeze v1 — cursor_opus46max_072

Thesis: BB width at 120-bar minimum inside Keltner = squeeze. Breakout on
squeeze release with momentum oscillator for direction prediction.
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


def _ema(data, period):
    n = len(data)
    out = np.zeros(n, dtype=np.float64)
    out[0] = data[0]
    alpha = 2.0 / (period + 1)
    for i in range(1, n):
        out[i] = alpha * data[i] + (1 - alpha) * out[i-1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_072"
    is_long_only = False
    session_start = 570   # 09:30
    session_end = 915     # 15:15
    max_trades_per_day = 5

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bb_period", default=20.0, low=15.0, high=30.0),
            TunableParam("squeeze_bars_min", default=10.0, low=5.0, high=20.0),
            TunableParam("kc_mult", default=1.5, low=1.0, high=2.0),
            TunableParam("vix_min", default=12.0, low=8.0, high=16.0),
            TunableParam("vix_max", default=20.0, low=16.0, high=26.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.001, high=0.006),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        bb_period = int(params.get("bb_period", 20.0))
        squeeze_min = int(params.get("squeeze_bars_min", 10.0))
        kc_mult = params.get("kc_mult", 1.5)
        vix_min = params.get("vix_min", 12.0)
        vix_max = params.get("vix_max", 20.0)
        stop_pct = params.get("stop_loss_pct", 0.003)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high_ = df["high"].to_numpy().astype(np.float64)
        low_ = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)

        atr14 = _compute_atr(high_, low_, close, 14)
        atr20 = _compute_atr(high_, low_, close, 20)
        ema20 = _ema(close, 20)

        # BB
        bb_mid = np.zeros(n, dtype=np.float64)
        bb_upper = np.zeros(n, dtype=np.float64)
        bb_lower = np.zeros(n, dtype=np.float64)
        bb_width = np.zeros(n, dtype=np.float64)
        for i in range(bb_period, n):
            seg = close[i - bb_period + 1:i + 1]
            mu = np.mean(seg)
            std = np.std(seg)
            bb_mid[i] = mu
            bb_upper[i] = mu + 2.0 * std
            bb_lower[i] = mu - 2.0 * std
            if mu > 1e-8:
                bb_width[i] = (bb_upper[i] - bb_lower[i]) / mu * 100.0

        # Keltner
        kc_upper = ema20 + kc_mult * atr20
        kc_lower = ema20 - kc_mult * atr20

        # Squeeze: BB inside KC
        squeeze_on = (bb_upper < kc_upper) & (bb_lower > kc_lower)

        # Squeeze duration counter
        squeeze_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if squeeze_on[i]:
                squeeze_count[i] = squeeze_count[i-1] + 1
            else:
                squeeze_count[i] = 0

        # Momentum oscillator
        mom_osc = close - bb_mid

        # Momentum slope (10-bar)
        mom_slope = np.zeros(n, dtype=np.float64)
        for i in range(10, n):
            seg = mom_osc[i-9:i+1]
            x = np.arange(10, dtype=np.float64)
            xm = np.mean(x)
            sm = np.mean(seg)
            num = np.sum((x - xm) * (seg - sm))
            den = np.sum((x - xm) ** 2)
            if den > 1e-12:
                mom_slope[i] = num / den

        # Volume confirmation
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)

        vix_ok = (vix >= vix_min) & (vix <= vix_max)

        # Entry: squeeze release (was on, now off) + breakout
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            was_squeezed = squeeze_count[i-1] >= squeeze_min
            released = was_squeezed and not squeeze_on[i]
            vol_confirm = volume[i] > 1.5 * avg_vol[i]

            if released and vix_ok[i] and vol_confirm:
                if close[i] > bb_upper[i] and mom_slope[i] > 0:
                    long_entry[i] = True
                elif close[i] < bb_lower[i] and mom_slope[i] < 0:
                    short_entry[i] = True

        # Signal exit: BB width contracts again (false breakout)
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if squeeze_on[i] and not squeeze_on[i-1]:
                signal_exit_long[i] = True
                signal_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_atr_mult=2.5,
            trailing_stop_pct=0.002,
            trailing_activate_pct=0.003,
            time_stop_bars=60,
        )
