"""Camarilla Pivot Points — cursor_opus46max_148

Thesis: Camarilla pivots from previous day's OHLC create S3/R3 mean-reversion
zones (~65% of days) and S4/R4 breakout levels (~35% of days). Bounce at S3/R3
or trade breakout beyond S4/R4.
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
    name = "cursor_opus46max_148"
    is_long_only = False
    session_start = 560   # 09:20
    session_end = 920     # 15:20
    max_trades_per_day = 8

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("bounce_tolerance", default=0.001, low=0.0005, high=0.003),
            TunableParam("breakout_vol_mult", default=1.5, low=1.2, high=2.5),
            TunableParam("vix_max_range", default=22.0, low=15.0, high=28.0),
            TunableParam("breakout_target_bps", default=20.0, low=10.0, high=35.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        bounce_tol = params.get("bounce_tolerance", 0.001)
        brk_vol = params.get("breakout_vol_mult", 1.5)
        vix_max = params.get("vix_max_range", 22.0)
        brk_target = params.get("breakout_target_bps", 20.0)

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
        day_id = df["day_id"].to_numpy()

        atr = _compute_atr(high, low, close, 14)

        # Previous day OHLC
        prev_high = np.zeros(n, dtype=np.float64)
        prev_low = np.zeros(n, dtype=np.float64)
        prev_close_arr = np.zeros(n, dtype=np.float64)

        # Collect per-day stats
        day_starts = [0]
        for i in range(1, n):
            if day_id[i] != day_id[i-1]:
                day_starts.append(i)

        for d in range(len(day_starts)):
            start = day_starts[d]
            end = day_starts[d+1] if d+1 < len(day_starts) else n
            if d == 0:
                # No previous day
                ph = high[start]
                pl_ = low[start]
                pc = close[start]
            else:
                prev_start = day_starts[d-1]
                prev_end = start
                ph = np.max(high[prev_start:prev_end])
                pl_ = np.min(low[prev_start:prev_end])
                pc = close[prev_end - 1]
            for i in range(start, end):
                prev_high[i] = ph
                prev_low[i] = pl_
                prev_close_arr[i] = pc

        # Camarilla levels
        rng = prev_high - prev_low
        cam_r4 = prev_close_arr + rng * 1.1 / 2
        cam_r3 = prev_close_arr + rng * 1.1 / 4
        cam_s3 = prev_close_arr - rng * 1.1 / 4
        cam_s4 = prev_close_arr - rng * 1.1 / 2

        # Volume filter
        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        vix_ok = vix < vix_max
        time_ok = (time_mins >= 565) & (time_mins <= 915)

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        for i in range(2, n):
            if not time_ok[i]:
                continue

            # S3 bounce (long): close near S3 from below, turning up
            if (close[i] >= cam_s3[i] * (1 - bounce_tol) and
                    close[i] <= cam_s3[i] * (1 + bounce_tol) and
                    close[i] > close[i-1] and
                    vix_ok[i]):
                long_entry[i] = True

            # R3 rejection (short): close near R3 from above, turning down
            elif (close[i] >= cam_r3[i] * (1 - bounce_tol) and
                    close[i] <= cam_r3[i] * (1 + bounce_tol) and
                    close[i] < close[i-1] and
                    vix_ok[i]):
                short_entry[i] = True

            # R4 breakout (long): sustained above R4 with volume
            elif (close[i] > cam_r4[i] and close[i-1] > cam_r4[i] and
                    volume[i] > brk_vol * avg_vol[i]):
                long_entry[i] = True

            # S4 breakdown (short): sustained below S4 with volume
            elif (close[i] < cam_s4[i] and close[i-1] < cam_s4[i] and
                    volume[i] > brk_vol * avg_vol[i]):
                short_entry[i] = True

        # Signal exit: S3 bounce targets R3, breakout trails
        # For range trades: target opposite level
        # For simplicity, use fixed target + time stop
        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=np.zeros(n, dtype=np.bool_),
            signal_exit_short=np.zeros(n, dtype=np.bool_),
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=0.003,
            target_pct=brk_target / 10000.0,
            trailing_stop_pct=0.001,
            trailing_activate_pct=0.0015,
            time_stop_bars=45,
        )
