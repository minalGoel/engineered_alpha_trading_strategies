"""Elder Impulse System — cursor_opus46max_150

Thesis: EMA(9) for trend + MACD(9,21,7) histogram for momentum.
Color-code each bar: green (both rising), red (both falling), blue (conflict).
Enter on 2-bar confirmed color change with VWAP confirmation.
Never long on red, never short on green.
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


def _ema(arr, period):
    n = len(arr)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    alpha = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, n):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i-1]
    return out


class Strategy(BaseStrategy):
    name = "cursor_opus46max_150"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 915     # 15:15
    max_trades_per_day = 12

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ema_period", default=9.0, low=5.0, high=15.0),
            TunableParam("macd_fast", default=9.0, low=5.0, high=12.0),
            TunableParam("macd_slow", default=21.0, low=15.0, high=30.0),
            TunableParam("macd_signal", default=7.0, low=5.0, high=12.0),
            TunableParam("max_color_changes_30", default=5.0, low=3.0, high=8.0),
            TunableParam("vix_max", default=22.0, low=15.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ema_per = int(params.get("ema_period", 9.0))
        macd_fast = int(params.get("macd_fast", 9.0))
        macd_slow = int(params.get("macd_slow", 21.0))
        macd_sig_per = int(params.get("macd_signal", 7.0))
        max_changes = int(params.get("max_color_changes_30", 5.0))
        vix_max = params.get("vix_max", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        atr = _compute_atr(high, low, close, 14)

        # EMA for trend
        ema9 = _ema(close, ema_per)

        # MACD
        ema_f = _ema(close, macd_fast)
        ema_s = _ema(close, macd_slow)
        macd_line = ema_f - ema_s
        macd_signal = _ema(macd_line, macd_sig_per)
        macd_hist = macd_line - macd_signal

        # Impulse color: green, red, blue
        # green = 1: EMA rising AND histogram rising
        # red = -1: EMA falling AND histogram falling
        # blue = 0: conflict
        impulse = np.zeros(n, dtype=np.int32)  # 0=blue, 1=green, -1=red
        for i in range(1, n):
            ema_rising = ema9[i] > ema9[i-1]
            ema_falling = ema9[i] < ema9[i-1]
            hist_rising = macd_hist[i] > macd_hist[i-1]
            hist_falling = macd_hist[i] < macd_hist[i-1]

            if ema_rising and hist_rising:
                impulse[i] = 1   # green
            elif ema_falling and hist_falling:
                impulse[i] = -1  # red
            else:
                impulse[i] = 0   # blue

        # Consecutive green/red count
        green_count = np.zeros(n, dtype=np.int32)
        red_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if impulse[i] == 1:
                green_count[i] = green_count[i-1] + 1 if impulse[i-1] == 1 else 1
            elif impulse[i] == -1:
                red_count[i] = red_count[i-1] + 1 if impulse[i-1] == -1 else 1

        # Choppiness: count color changes in last 30 bars
        color_change = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if impulse[i] != impulse[i-1]:
                color_change[i] = 1
        cum_changes = np.cumsum(color_change)
        changes_30 = np.zeros(n, dtype=np.int32)
        for i in range(30, n):
            changes_30[i] = cum_changes[i] - cum_changes[i-30]
        not_choppy = changes_30 <= max_changes

        # Volume filter
        avg_vol = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        vix_ok = (vix >= 12.0) & (vix <= vix_max)
        time_ok = (time_mins >= 570) & (time_mins <= 900)

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        for i in range(2, n):
            if not time_ok[i] or not vix_ok[i] or not not_choppy[i]:
                continue

            # Long: 2 consecutive green bars, close > EMA9, close > VWAP,
            # histogram > 0, volume confirms
            if (green_count[i] == 2 and
                    close[i] > ema9[i] and
                    close[i] > vwap[i] and
                    macd_hist[i] > 0 and
                    volume[i] > avg_vol[i]):
                long_entry[i] = True

            # Short: 2 consecutive red bars, close < EMA9, close < VWAP,
            # histogram < 0, volume confirms
            elif (red_count[i] == 2 and
                    close[i] < ema9[i] and
                    close[i] < vwap[i] and
                    macd_hist[i] < 0 and
                    volume[i] > avg_vol[i]):
                short_entry[i] = True

        # Signal exit: impulse color changes to opposite or close crosses EMA9
        # Long exit: impulse turns red or close < EMA9
        # Short exit: impulse turns green or close > EMA9
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if impulse[i] == -1 or close[i] < ema9[i]:
                sig_exit_long[i] = True
            if impulse[i] == 1 or close[i] > ema9[i]:
                sig_exit_short[i] = True

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=0.001,
            target_pct=0.002,
            trailing_stop_pct=0.0005,
            trailing_activate_pct=0.001,
            time_stop_bars=15,
        )
