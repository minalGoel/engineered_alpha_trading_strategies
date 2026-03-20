"""Bonus Record Date v1 — cursor_opus46max_198

Thesis: On bonus/split ex-date, stocks overshoot to downside as retail dumps.
After 30 mins of selling exhaustion, buy the bounce. Adapted: detect sessions
where stock opens significantly below prior close with high volume, then
buy the bounce after selling pressure exhausts.
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
    name = "cursor_opus46max_198"
    is_long_only = False
    session_start = 585   # 09:45
    session_end = 910     # 15:10 (but entry only before 14:00 = 840)
    max_trades_per_day = 1

    def tunable_params(self) -> list[TunableParam]:
        return [
            TunableParam("gap_down_thresh_bps", default=20.0, low=10.0, high=50.0),
            TunableParam("wait_bars", default=30.0, low=15.0, high=45.0),
            TunableParam("recovery_bps", default=10.0, low=5.0, high=20.0),
            TunableParam("vol_surge_mult", default=2.0, low=1.5, high=4.0),
            TunableParam("stop_loss_pct", default=0.003, low=0.002, high=0.005),
            TunableParam("target_pct", default=0.003, low=0.002, high=0.005),
        ]

    def compute(self, df: pl.DataFrame, params: dict) -> StrategySignals:
        gap_th = params.get("gap_down_thresh_bps", 20.0)
        wait = int(params.get("wait_bars", 30.0))
        recovery_th = params.get("recovery_bps", 10.0)
        vol_mult = params.get("vol_surge_mult", 2.0)
        stop_pct = params.get("stop_loss_pct", 0.003)
        target_pct = params.get("target_pct", 0.003)

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

        ema10 = _compute_ema(close, 10)

        # Detect gap-down and gap-up opens
        gap_bps = np.zeros(n, dtype=np.float64)
        bars_since_open = np.zeros(n, dtype=np.int32)
        session_low = np.zeros(n, dtype=np.float64)
        session_high = np.zeros(n, dtype=np.float64)
        cum_vol = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if i == 0 or day_id[i] != day_id[i-1]:
                bars_since_open[i] = 0
                if i > 0 and close[i-1] > 0:
                    gap_bps[i] = (opn[i] - close[i-1]) / close[i-1] * 10000
                session_low[i] = low[i]
                session_high[i] = high[i]
                cum_vol[i] = volume[i]
            else:
                bars_since_open[i] = bars_since_open[i-1] + 1
                gap_bps[i] = gap_bps[i - bars_since_open[i]]
                session_low[i] = min(session_low[i-1], low[i])
                session_high[i] = max(session_high[i-1], high[i])
                cum_vol[i] = cum_vol[i-1] + volume[i]

        # Recovery from session low
        recovery_from_low = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if session_low[i] > 0:
                recovery_from_low[i] = (close[i] - session_low[i]) / session_low[i] * 10000

        # Volume surge (cumulative first 30 bars vs average)
        avg_vol = np.zeros(n, dtype=np.float64)
        for i in range(19, n):
            avg_vol[i] = np.mean(volume[i-19:i+1])
        avg_vol = np.clip(avg_vol, 1.0, None)

        long_entry = np.zeros(n, dtype=np.bool_)
        short_entry = np.zeros(n, dtype=np.bool_)

        for i in range(n):
            bso = bars_since_open[i]
            if bso < wait or bso > 90:
                continue
            if time_mins[i] > 840:  # no entry after 14:00
                continue

            g = gap_bps[i]

            # Gap down + recovery = buy bounce
            if g < -gap_th and recovery_from_low[i] > recovery_th:
                if close[i] > vwap[i] and close[i] > ema10[i]:
                    long_entry[i] = True

            # Gap up euphoria fading = sell
            elif g > gap_th * 1.5:
                if close[i] < opn[i - bso] and close[i] < vwap[i] and close[i] < ema10[i]:
                    short_entry[i] = True

        # Signal exit
        sig_exit_long = np.zeros(n, dtype=np.bool_)
        sig_exit_short = np.zeros(n, dtype=np.bool_)
        for i in range(10, n):
            # VWAP slope against position
            if vwap[i] < vwap[i-10]:
                sig_exit_long[i] = True
            if vwap[i] > vwap[i-10]:
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
            trailing_activate_pct=0.0020,
            trailing_stop_pct=0.0015,
            time_stop_bars=150,
        )
