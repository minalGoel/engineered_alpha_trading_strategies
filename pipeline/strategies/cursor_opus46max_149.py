"""Heikin Ashi Trend — cursor_opus46max_149

Thesis: HA candles smooth price action: HA_close=(O+H+L+C)/4,
HA_open=(prev_HA_open+prev_HA_close)/2. Color change + no-wick strength
+ EMA confirmation identifies clean trend onsets. Signal generation uses
HA candles; execution uses actual prices.
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
    name = "cursor_opus46max_149"
    is_long_only = False
    session_start = 565   # 09:25
    session_end = 915     # 15:15
    max_trades_per_day = 10

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("ema_fast", default=9.0, low=5.0, high=15.0),
            TunableParam("ema_slow", default=21.0, low=15.0, high=30.0),
            TunableParam("wick_tolerance", default=0.20, low=0.05, high=0.35),
            TunableParam("min_prev_trend_bars", default=3.0, low=2.0, high=5.0),
            TunableParam("vix_max", default=25.0, low=18.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        ema_fast_per = int(params.get("ema_fast", 9.0))
        ema_slow_per = int(params.get("ema_slow", 21.0))
        wick_tol = params.get("wick_tolerance", 0.20)
        min_prev = int(params.get("min_prev_trend_bars", 3.0))
        vix_max = params.get("vix_max", 25.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
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

        # Heikin Ashi candles
        ha_close = (open_ + high + low + close) / 4.0
        ha_open = np.zeros(n, dtype=np.float64)
        ha_open[0] = (open_[0] + close[0]) / 2.0
        for i in range(1, n):
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2.0
        ha_high = np.maximum(high, np.maximum(ha_open, ha_close))
        ha_low = np.minimum(low, np.minimum(ha_open, ha_close))

        # HA color: green if ha_close > ha_open
        ha_green = ha_close > ha_open  # True = green, False = red

        # HA body size
        ha_body = np.abs(ha_close - ha_open)
        ha_body_safe = np.clip(ha_body, 1e-10, None)

        # HA strength: strong_up if green and lower wick small relative to body
        # lower wick for green = ha_open - ha_low (since ha_open < ha_close for green)
        # upper wick for red = ha_high - ha_open (since ha_open > ha_close for red)
        lower_wick = np.where(ha_green, np.minimum(ha_open, ha_close) - ha_low, 0.0)
        upper_wick = np.where(~ha_green, ha_high - np.maximum(ha_open, ha_close), 0.0)

        strong_up = ha_green & (lower_wick / ha_body_safe < wick_tol)
        strong_down = (~ha_green) & (upper_wick / ha_body_safe < wick_tol)

        # HA has both wicks (indecision) — for signal exit
        both_wick_ratio = 0.3
        has_both_wicks = (
            ((np.minimum(ha_open, ha_close) - ha_low) / ha_body_safe > both_wick_ratio) &
            ((ha_high - np.maximum(ha_open, ha_close)) / ha_body_safe > both_wick_ratio)
        )

        # Consecutive HA color count
        ha_green_count = np.zeros(n, dtype=np.int32)
        ha_red_count = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if ha_green[i]:
                ha_green_count[i] = ha_green_count[i-1] + 1 if ha_green[i-1] else 1
                ha_red_count[i] = 0
            else:
                ha_red_count[i] = ha_red_count[i-1] + 1 if not ha_green[i-1] else 1
                ha_green_count[i] = 0

        # Previous trend length (before color change)
        prev_trend_len = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            if ha_green[i] and not ha_green[i-1]:
                # Just turned green; previous red trend length
                prev_trend_len[i] = ha_red_count[i-1]
            elif not ha_green[i] and ha_green[i-1]:
                # Just turned red; previous green trend length
                prev_trend_len[i] = ha_green_count[i-1]

        # EMA on actual prices
        ema_fast = _ema(close, ema_fast_per)
        ema_slow = _ema(close, ema_slow_per)

        # Volume filter
        avg_vol = df["volume"].rolling_mean(10).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        vix_ok = (vix >= 12.0) & (vix <= vix_max)
        time_ok = (time_mins >= 570) & (time_mins <= 900)

        # Choppiness filter: count color changes in last 30 bars
        color_changes = np.zeros(n, dtype=np.int32)
        for i in range(1, n):
            color_changes[i] = 1 if ha_green[i] != ha_green[i-1] else 0
        cum_changes = np.cumsum(color_changes)
        changes_30 = np.zeros(n, dtype=np.int32)
        for i in range(30, n):
            changes_30[i] = cum_changes[i] - cum_changes[i-30]
        not_choppy = changes_30 < 8

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        for i in range(2, n):
            if not time_ok[i] or not vix_ok[i] or not not_choppy[i]:
                continue

            # Long: HA turned green, at least 2 consecutive green, strong up on first green,
            # close > VWAP, EMA fast > EMA slow, previous red trend >= min_prev bars
            if (ha_green_count[i] == 2 and
                    strong_up[i] and
                    close[i] > vwap[i] and
                    ema_fast[i] > ema_slow[i] and
                    prev_trend_len[i-1] >= min_prev and
                    volume[i] > avg_vol[i]):
                long_entry[i] = True

            # Short: HA turned red, at least 2 consecutive red, strong down,
            # close < VWAP, EMA fast < EMA slow
            elif (ha_red_count[i] == 2 and
                    strong_down[i] and
                    close[i] < vwap[i] and
                    ema_fast[i] < ema_slow[i] and
                    prev_trend_len[i-1] >= min_prev and
                    volume[i] > avg_vol[i]):
                short_entry[i] = True

        # Signal exit: HA color changes to opposite or HA shows indecision (both wicks)
        # For longs: exit when HA turns red or has both wicks
        # For shorts: exit when HA turns green or has both wicks
        sig_exit_long = (~ha_green) | has_both_wicks
        sig_exit_short = ha_green | has_both_wicks

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
            time_stop_bars=20,
        )
