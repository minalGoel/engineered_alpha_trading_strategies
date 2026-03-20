"""New Listing Momentum v1 — cursor_opus46max_200

Thesis: IPO listing day offers unique opportunities. Stocks that list at
significant premium and maintain it through first 30 minutes attract FOMO
buying. Adapted: detect first-day-like conditions (extreme volume surge,
wide range) and trade momentum continuation after initial 30-bar confirmation.
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


def _compute_ema(data, period):
    n = len(data)
    ema = np.zeros(n, dtype=np.float64)
    if n == 0:
        return ema
    ema[0] = data[0]
    k = 2.0 / (period + 1.0)
    for i in range(1, n):
        ema[i] = data[i] * k + ema[i-1] * (1.0 - k)
    return ema


class Strategy(BaseStrategy):
    name = "cursor_opus46max_200"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 840     # 14:00 (early close for listing day)
    max_trades_per_day = 2

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_thresh_pct", default=2.0, low=1.0, high=5.0),
            TunableParam("vol_surge_mult", default=4.0, low=2.0, high=8.0),
            TunableParam("wait_bars", default=30.0, low=15.0, high=45.0),
            TunableParam("premium_persist_pct", default=80.0, low=60.0, high=95.0),
            TunableParam("stop_loss_pct", default=0.0035, low=0.002, high=0.005),
            TunableParam("target_pct", default=0.006, low=0.004, high=0.01),
            TunableParam("trailing_activate_pct", default=0.004, low=0.002, high=0.006),
            TunableParam("trailing_stop_pct", default=0.002, low=0.001, high=0.003),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_th = params.get("gap_thresh_pct", 2.0)
        vol_mult = params.get("vol_surge_mult", 4.0)
        wait = int(params.get("wait_bars", 30.0))
        persist_pct = params.get("premium_persist_pct", 80.0) / 100.0
        stop_pct = params.get("stop_loss_pct", 0.0035)
        target_pct = params.get("target_pct", 0.006)
        trail_act = params.get("trailing_activate_pct", 0.004)
        trail_pct = params.get("trailing_stop_pct", 0.002)

        n = len(df)
        close = df["close"].to_numpy().astype(np.float64)
        opn = df["open"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)
        volume = np.nan_to_num(volume, nan=1.0)
        time_mins = df["time_minutes"].to_numpy().astype(np.int32)
        day_id = df["day_id"].to_numpy()

        # VWAP
        df_v = df.with_columns([
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
            .cum_sum().over("day_id").alias("_ctv"),
            pl.col("volume").cum_sum().over("day_id").alias("_cv"),
        ]).with_columns((pl.col("_ctv") / pl.col("_cv")).alias("_vwap"))
        vwap = df_v["_vwap"].to_numpy().astype(np.float64)
        vwap = np.nan_to_num(vwap, nan=0.0)

        # Detect extreme gap + extreme volume (listing-like day proxy)
        gap_pct = np.zeros(n, dtype=np.float64)
        bars_since_open = np.zeros(n, dtype=np.int32)
        day_open = np.zeros(n, dtype=np.float64)
        day_open_price = np.zeros(n, dtype=np.float64)
        session_high = np.zeros(n, dtype=np.float64)
        session_low = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                bars_since_open[i] = 0
                if i > 0 and close[i-1] > 0:
                    gap_pct[i] = (opn[i] - close[i-1]) / close[i-1] * 100
                day_open_price[i] = opn[i]
                session_high[i] = high[i]
                session_low[i] = low[i]
            else:
                bars_since_open[i] = bars_since_open[i-1] + 1
                gap_pct[i] = gap_pct[i - bars_since_open[i]]
                day_open_price[i] = day_open_price[i-1]
                session_high[i] = max(session_high[i-1], high[i])
                session_low[i] = min(session_low[i-1], low[i])

        # Volume surge per day
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)
        vol_surge = volume > (vol_mult * avg_vol)

        # Premium persistence: how much of initial gap is maintained
        premium_persist = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if abs(gap_pct[i]) > 0.1 and day_open_price[i] > 0:
                # Current premium vs initial gap
                if bars_since_open[i] > 0:
                    initial_gap = gap_pct[i]
                    prev_close_approx = day_open_price[i] / (1 + initial_gap / 100.0) if abs(initial_gap) > 0 else day_open_price[i]
                    if prev_close_approx > 0:
                        current_prem = (close[i] - prev_close_approx) / prev_close_approx * 100
                        if abs(initial_gap) > 0.01:
                            premium_persist[i] = current_prem / initial_gap

        # First 30-bar VWAP
        first30_vwap = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if bars_since_open[i] == wait:
                first30_vwap[i] = vwap[i]
            elif bars_since_open[i] > wait:
                first30_vwap[i] = first30_vwap[i-1]

        time_ok = (time_mins >= 585) & (time_mins <= 840)

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        for i in range(n):
            bso = bars_since_open[i]
            if bso < wait or bso > 90:
                continue
            if not time_ok[i]:
                continue

            g = gap_pct[i]

            # Strong gap up maintained = momentum long
            if g > gap_th and premium_persist[i] > persist_pct:
                if close[i] > first30_vwap[i] and close[i] > vwap[i]:
                    # New high after wait period
                    if close[i] >= session_high[i] * 0.999:
                        long_entry[i] = True

            # Gap down or premium lost = momentum short
            if g > 0 and premium_persist[i] < 0.5 and bso >= wait:
                if close[i] < first30_vwap[i] and close[i] < vwap[i]:
                    if close[i] <= session_low[i] * 1.001:
                        short_entry[i] = True
            elif g < -gap_th / 2:
                if close[i] < first30_vwap[i] and close[i] < vwap[i]:
                    if close[i] <= session_low[i] * 1.001:
                        short_entry[i] = True

        # Signal exit: premium persistence drops for longs
        sig_exit_long = premium_persist < 0.6
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(2, n):
            if close[i] > close[i-1] and close[i-1] > close[i-2]:
                sig_exit_short[i] = True

        atr = _compute_atr(high, low, close, 14)

        return StrategySignals(
            long_entry=long_entry,
            short_entry=short_entry,
            signal_exit_long=sig_exit_long,
            signal_exit_short=sig_exit_short,
            atr_arr=atr,
            target_indicator=np.zeros(n, dtype=np.float64),
            stop_loss_pct=stop_pct,
            target_pct=target_pct,
            trailing_activate_pct=trail_act,
            trailing_stop_pct=trail_pct,
            time_stop_bars=90,
        )
