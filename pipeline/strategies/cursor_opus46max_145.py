"""Ichimoku Cloud — cursor_opus46max_145

Thesis: Adapted Ichimoku (7/22/44 for 1-min) provides multi-factor trend
assessment. Enter on Tenkan-Kijun cross when price is above/below cloud
with all 5 signals aligned.
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


def _rolling_high(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        out[i] = np.max(arr[start:i+1])
    return out


def _rolling_low(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - period + 1)
        out[i] = np.min(arr[start:i+1])
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_145"
    is_long_only = False
    session_start = 570   # 09:30 (need 44 bars for init)
    session_end = 915     # 15:15
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("tenkan_period", default=7.0, low=5.0, high=12.0),
            TunableParam("kijun_period", default=22.0, low=15.0, high=30.0),
            TunableParam("senkou_b_period", default=44.0, low=30.0, high=60.0),
            TunableParam("cloud_thickness_min_bps", default=3.0, low=1.0, high=8.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        t_per = int(params.get("tenkan_period", 7.0))
        k_per = int(params.get("kijun_period", 22.0))
        sb_per = int(params.get("senkou_b_period", 44.0))
        cloud_min = params.get("cloud_thickness_min_bps", 3.0)
        vix_max = params.get("vix_max", 22.0)
        displacement = k_per

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr = _compute_atr(high, low, close, 14)

        # Ichimoku components
        tenkan = (_rolling_high(high, t_per) + _rolling_low(low, t_per)) / 2
        kijun = (_rolling_high(high, k_per) + _rolling_low(low, k_per)) / 2
        senkou_a_raw = (tenkan + kijun) / 2
        senkou_b_raw = (_rolling_high(high, sb_per) + _rolling_low(low, sb_per)) / 2

        # Current senkou spans (displaced forward by k_per, but for current position
        # we use the values from k_per bars ago)
        senkou_a = np.zeros(n, dtype=np.float64)
        senkou_b = np.zeros(n, dtype=np.float64)
        for i in range(displacement, n):
            senkou_a[i] = senkou_a_raw[i - displacement]
            senkou_b[i] = senkou_b_raw[i - displacement]

        # Chikou span: close compared to close[k_per] bars ago
        chikou_above = np.zeros(n, dtype=np.bool_)
        for i in range(displacement, n):
            chikou_above[i] = close[i] > close[i - displacement]

        # Cloud color (future): senkou_a_raw vs senkou_b_raw at current bar (projected forward)
        future_cloud_green = senkou_a_raw > senkou_b_raw
        future_cloud_red = senkou_a_raw < senkou_b_raw

        cloud_top = np.maximum(senkou_a, senkou_b)
        cloud_bot = np.minimum(senkou_a, senkou_b)

        above_cloud = close > cloud_top
        below_cloud = close < cloud_bot
        in_cloud = ~above_cloud & ~below_cloud

        # TK cross
        tk_bullish_cross = np.zeros(n, dtype=np.bool_)
        tk_bearish_cross = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]:
                tk_bullish_cross[i] = True
            if tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]:
                tk_bearish_cross[i] = True

        # Cloud thickness filter
        safe_close = np.clip(close, 1e-10, None)
        cloud_thick_bps = np.abs(senkou_a - senkou_b) / safe_close * 10000.0
        cloud_ok = cloud_thick_bps > cloud_min

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_ok = volume > avg_vol

        vix_ok = (vix >= 12.0) & (vix <= vix_max)
        time_ok = (time_mins >= 570) & (time_mins <= 870)

        # Long: all 5 signals bullish
        long_entry = (
            tk_bullish_cross &
            above_cloud &
            chikou_above &
            future_cloud_green &
            cloud_ok &
            vol_ok &
            vix_ok &
            time_ok
        )

        # Short: all 5 bearish
        chikou_below = np.zeros(n, dtype=np.bool_)
        for i in range(displacement, n):
            chikou_below[i] = close[i] < close[i - displacement]

        short_entry = (
            tk_bearish_cross &
            below_cloud &
            chikou_below &
            future_cloud_red &
            cloud_ok &
            vol_ok &
            vix_ok &
            time_ok
        )

        # Signal exit: price enters cloud or TK reverses
        sig_exit_long = in_cloud | tk_bearish_cross
        sig_exit_short = in_cloud | tk_bullish_cross

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=kijun.copy(),
            target_pct=0.0025,
            stop_loss_pct=0.002,
            time_stop_bars=30,
        )
