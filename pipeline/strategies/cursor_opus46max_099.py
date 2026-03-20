"""ORB Retest v1 — cursor_opus46max_099

Thesis: After an ORB breakout, the ORB level often gets retested 1-3 hours later
as mid-day institutional flow revisits the morning's price discovery zone. When
the retest holds (price touches ORB level but doesn't break through), it confirms
genuine support/resistance and provides a second entry with higher conviction.
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
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    if n < period + 1:
        return rsi
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period])
    if avg_loss > 0:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi[period] = 100.0
    for i in range(period, len(delta)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss > 0:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        else:
            rsi[i + 1] = 100.0
    return rsi


class Strategy(BaseStrategy):
    name = "cursor_opus46max_099"
    is_long_only = False
    session_start = 630   # 10:30 (retests happen mid-session)
    session_end = 910     # 15:10
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("retest_min_bars", default=60.0, low=30.0, high=90.0),
            TunableParam("retest_max_bars", default=180.0, low=120.0, high=240.0),
            TunableParam("retest_tol_pct", default=0.001, low=0.0005, high=0.002),
            TunableParam("rsi_long_min", default=40.0, low=30.0, high=50.0),
            TunableParam("rsi_short_max", default=60.0, low=50.0, high=70.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_atr_mult", default=2.0, low=1.0, high=3.0),
            TunableParam("vix_low", default=12.0, low=8.0, high=16.0),
            TunableParam("vix_high", default=22.0, low=18.0, high=28.0),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        retest_min = int(params.get("retest_min_bars", 60.0))
        retest_max = int(params.get("retest_max_bars", 180.0))
        retest_tol = params.get("retest_tol_pct", 0.001)
        rsi_long_min = params.get("rsi_long_min", 40.0)
        rsi_short_max = params.get("rsi_short_max", 60.0)
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_atr_mult = params.get("target_atr_mult", 2.0)
        vix_lo = params.get("vix_low", 12.0)
        vix_hi = params.get("vix_high", 22.0)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        vix = df["vix"].to_numpy().astype(np.float64)
        vix = np.nan_to_num(vix, nan=99.0)
        time_min = df["time_minutes"].to_numpy()
        day_id = df["day_id"].to_numpy()

        # -- VWAP --
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # -- ORB 15-min --
        orb_high = np.zeros(n, dtype=np.float64)
        orb_low = np.zeros(n, dtype=np.float64)
        orb_computed = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if i == 0 or day_id[i] != day_id[i - 1]:
                cur_h = high[i]
                cur_l = low[i]
            if time_min[i] < 570:
                cur_h = max(cur_h, high[i])
                cur_l = min(cur_l, low[i])
            orb_high[i] = cur_h
            orb_low[i] = cur_l
            if time_min[i] >= 570:
                orb_computed[i] = True

        # -- Detect initial breakouts (09:30-10:30 window) --
        initial_breakout_up_bar = np.full(n, -1, dtype=np.int64)
        initial_breakout_down_bar = np.full(n, -1, dtype=np.int64)
        for i in range(n):
            if not orb_computed[i]:
                continue
            if i > 0 and day_id[i] == day_id[i - 1]:
                initial_breakout_up_bar[i] = initial_breakout_up_bar[i - 1]
                initial_breakout_down_bar[i] = initial_breakout_down_bar[i - 1]

            if time_min[i] >= 570 and time_min[i] <= 630:
                if close[i] > orb_high[i] and initial_breakout_up_bar[i] < 0:
                    initial_breakout_up_bar[i] = i
                if close[i] < orb_low[i] and initial_breakout_down_bar[i] < 0:
                    initial_breakout_down_bar[i] = i

        rsi = _compute_rsi(close, 14)
        vix_ok = (vix >= vix_lo) & (vix <= vix_hi)

        # -- Detect retest: price touches ORB level and bounces --
        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        avg_vol = df["volume"].rolling_mean(20).to_numpy().astype(np.float64)
        avg_vol = np.nan_to_num(avg_vol, nan=1.0)
        avg_vol = np.clip(avg_vol, 1.0, None)

        for i in range(1, n):
            if not orb_computed[i] or not vix_ok[i]:
                continue
            if day_id[i] != day_id[i - 1]:
                continue

            orb_h = orb_high[i]
            orb_l = orb_low[i]
            orb_h_safe = orb_h if orb_h > 0 else 1.0
            orb_l_safe = orb_l if orb_l > 0 else 1.0

            # Long retest: initial up breakout occurred, now low touches ORB_high, close stays above
            if initial_breakout_up_bar[i] > 0:
                bars_since = i - initial_breakout_up_bar[i]
                if retest_min <= bars_since <= retest_max:
                    # Low within tolerance of ORB_high (touched and bounced)
                    touched = abs(low[i] - orb_h) / orb_h_safe <= retest_tol
                    close_above = close[i] > orb_h
                    bar_range = high[i] - low[i]
                    upper_half = bar_range > 0 and (close[i] - low[i]) / bar_range > 0.5
                    rsi_ok = rsi[i] > rsi_long_min
                    vwap_ok = close[i] > vwap[i]
                    # Retest volume should be lower than breakout volume (healthy pullback)
                    if touched and close_above and upper_half and rsi_ok and vwap_ok:
                        long_entry[i] = True

            # Short retest: initial down breakout, high touches ORB_low, close stays below
            if initial_breakout_down_bar[i] > 0:
                bars_since = i - initial_breakout_down_bar[i]
                if retest_min <= bars_since <= retest_max:
                    touched = abs(high[i] - orb_l) / orb_l_safe <= retest_tol
                    close_below = close[i] < orb_l
                    bar_range = high[i] - low[i]
                    lower_half = bar_range > 0 and (close[i] - low[i]) / bar_range < 0.5
                    rsi_ok = rsi[i] < rsi_short_max
                    vwap_ok = close[i] < vwap[i]
                    if touched and close_below and lower_half and rsi_ok and vwap_ok:
                        short_entry[i] = True

        # -- Signal exit: close breaks ORB level against position by > 0.15% --
        signal_exit_long = np.zeros(n, dtype=np.bool_)
        signal_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(n):
            if orb_computed[i] and orb_high[i] > 0:
                if close[i] < orb_high[i] * (1.0 - 0.0015):
                    signal_exit_long[i] = True
            if orb_computed[i] and orb_low[i] > 0:
                if close[i] > orb_low[i] * (1.0 + 0.0015):
                    signal_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=signal_exit_long,
            signal_exit_short=signal_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_atr_mult=target_atr_mult,
            time_stop_bars=120,
        )
