"""Gap Exhaustion v1 — cursor_opus46max_070

Thesis: After gap open, price extends on declining volume = exhaustion.
Enter reversal when 3+ new extremes on declining volume + MFI divergence.
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


def _compute_rsi(close, period):
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        diff = close[i] - close[i-1]
        if diff > 0:
            gains[i] = diff
        else:
            losses[i] = -diff
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    if avg_loss > 1e-10:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss > 1e-10:
            rsi[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_070"
    is_long_only = False
    session_start = 555   # 09:15
    session_end = 840     # 14:00
    max_trades_per_day = 3

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_min", default=0.005, low=0.003, high=0.01),
            TunableParam("exhaustion_count_min", default=3.0, low=2.0, high=5.0),
            TunableParam("vix_min", default=13.0, low=8.0, high=16.0),
            TunableParam("vix_max", default=22.0, low=18.0, high=28.0),
            TunableParam("stop_loss_pct", default=0.004, low=0.002, high=0.007),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_min = params.get("gap_min", 0.005)
        exh_count_min = int(params.get("exhaustion_count_min", 3.0))
        vix_min = params.get("vix_min", 13.0)
        vix_max = params.get("vix_max", 22.0)
        stop_pct = params.get("stop_loss_pct", 0.004)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        open_ = df["open"].to_numpy().astype(np.float64)
        high_ = df["high"].to_numpy().astype(np.float64)
        low_ = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        day_ids = df["day_id"].to_numpy()
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)

        atr14 = _compute_atr(high_, low_, close, 14)

        # VWAP
        df2 = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume")).cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns(
            (pl.col("_ctv") / pl.col("_cv")).alias("_vwap")
        )
        vwap = df2["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # MFI-like: typical price * volume based RSI
        tp = (high_ + low_ + close) / 3.0
        mf = tp * volume
        mfi = _compute_rsi(mf, 14)

        # Gap per day
        gap_pct = np.zeros(n, dtype=np.float64)
        unique_days = np.unique(day_ids)
        prev_close = np.nan
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) == 0:
                continue
            day_open = open_[idx[0]]
            if not np.isnan(prev_close) and prev_close > 0:
                gap_pct[idx] = (day_open - prev_close) / prev_close
            prev_close = close[idx[-1]]

        # Exhaustion detection: new extremes on declining volume
        # Track per-day running high/low and volume at extremes
        exhaustion_count = np.zeros(n, dtype=np.int32)
        for d in unique_days:
            idx = np.where(day_ids == d)[0]
            if len(idx) < 3:
                continue
            gp = gap_pct[idx[0]]
            running_extreme = close[idx[0]]
            last_vol = volume[idx[0]]
            exh_cnt = 0
            for j in range(1, len(idx)):
                i = idx[j]
                if gp > 0:  # gap up: track new highs
                    if high_[i] > running_extreme:
                        if volume[i] < last_vol:
                            exh_cnt += 1
                        else:
                            exh_cnt = 0
                        last_vol = volume[i]
                        running_extreme = high_[i]
                elif gp < 0:  # gap down: track new lows
                    if low_[i] < running_extreme:
                        if volume[i] < last_vol:
                            exh_cnt += 1
                        else:
                            exh_cnt = 0
                        last_vol = volume[i]
                        running_extreme = low_[i]
                exhaustion_count[i] = exh_cnt

        vix_ok = (vix >= vix_min) & (vix <= vix_max)
        time_ok = (time_mins >= 570) & (time_mins <= 660)  # 09:30-11:00

        # Reversal candle: close above exhaustion bar high (long) or below low (short)
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)
        for i in range(1, n):
            if (gap_pct[i] < -gap_min and exhaustion_count[i] >= exh_count_min
                    and mfi[i] < 25.0 and close[i] > high_[i-1]
                    and vix_ok[i] and time_ok[i]):
                long_entry[i] = True
            if (gap_pct[i] > gap_min and exhaustion_count[i] >= exh_count_min
                    and mfi[i] > 75.0 and close[i] < low_[i-1]
                    and vix_ok[i] and time_ok[i]):
                short_entry[i] = True

        # Signal exit: volume surges 3x in gap direction
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(20, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)

        signal_exit_long = (volume > 3.0 * avg_vol) & (close < open_)
        signal_exit_short = (volume > 3.0 * avg_vol) & (close > open_)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr14,
            target_indicator=vwap,
            use_target_indicator=True,
            stop_loss_atr_mult=0.4,
            time_stop_bars=90,
        )
